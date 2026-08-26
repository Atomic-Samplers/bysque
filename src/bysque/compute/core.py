"""Compute derived quantities from Lobster coefficient matrices.

The central object is [LobsterComputable][bysque.compute.core.LobsterComputable],
which holds projected wavefunction coefficients together with the occupations,
k-points and k-weights needed to form density matrices and related descriptors.
[COBIComputable][bysque.compute.core.COBIComputable] specialises it for crystal
orbital bond index (COBI/ICOBI) quantities.

Shape and symbol conventions
----------------------------
The einsum patterns in this module use single-letter axis labels:

s : spin channels
k : k-points
n : bands
i : orbital (row) index of a matrix
j : orbital (column) index of a matrix
t : real-space translations
b : energy bins

The stored arrays therefore have the following canonical shapes:

coefficients : (s, k, i, n)   projection coefficients c_skin
occupations  : (s, k, n)      band occupations f_skn
eigenvalues  : (s, k, n)      band eigenvalues
k_points     : (k, 3)         fractional k-point coordinates
k_weights    : (k,)           k-point weights
"""

from __future__ import annotations

import re
from math import factorial, pi, sqrt
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self

import numpy as np

from bysque.compute.numpy_ import contract
from bysque.protocols import (
    ArrayNamespace,
    ContractFunction,
    LobsterMatricesLike,
    NumericArray,
    VasprunLike,
)
from bysque.utils import check_dimensions, get_cycle, get_invariant_permutations

if TYPE_CHECKING:
    from collections.abc import Sequence


class LobsterComputable[ArrayType: NumericArray]:
    """Container of projected coefficients that computes density matrices.

    Holds the quantities extracted from a Lobster coefficient calculation and
    contracts them into density matrices and derived working quantities. Every
    contraction is delegated to `contract_function`, so a single instance works
    with any array backend that satisfies
    [NumericArray][bysque.protocols.NumericArray].

    Parameters
    ----------
    coefficients : ArrayType
        Projection coefficients c_skin, shape (s, k, i, n).
    occupations : ArrayType
        Band occupations f_skn, shape (s, k, n).
    k_points : ArrayType
        Fractional k-point coordinates, shape (k, 3).
    k_weights : ArrayType
        k-point weights, shape (k,).
    eigenvalues : ArrayType or None
        Band eigenvalues, shape (s, k, n), or None when unavailable.
    use_time_reversal : bool, default True
        Whether to assume time-reversal symmetry. When set, the weight of every
        non-TRIM k-point is doubled and the real part of Hermitian results is
        returned.
    contract_function : ContractFunction, default contract
        Einsum-like contraction used for every tensor product.

    Attributes
    ----------
    xp : ArrayNamespace
        Array namespace used for elementwise operations. Defaults to numpy.

    Generic
    -------
    ArrayType : NumericArray
        The array type of every stored quantity and every returned value.
        Parameterising the class lets one implementation serve numpy, jax and
        torch without branching on the backend.
    """

    xp: ClassVar[ArrayNamespace[Any]] = np

    def __init__(
        self,
        coefficients: ArrayType,
        occupations: ArrayType,
        k_points: ArrayType,
        k_weights: ArrayType,
        eigenvalues: ArrayType | None,
        *,
        use_time_reversal: bool = True,
        contract_function: ContractFunction[ArrayType] = contract,
    ) -> None:
        self.coefficients = coefficients
        self.occupations = occupations

        self.k_points = k_points
        self.k_weights = k_weights

        self.eigenvalues = eigenvalues

        self.use_time_reversal = use_time_reversal

        self.contract_function = contract_function

    @classmethod
    def from_vasp_pymatgen_objects(
        cls,
        vasprun: VasprunLike,
        lobster_matrices: LobsterMatricesLike,
        **kwargs: Any,
    ) -> Self:
        """Build a `LobsterComputable` from pymatgen VASP and Lobster objects.

        Parameters
        ----------
        vasprun : VasprunLike
            Parsed vasprun providing eigenvalues, k-points and k-weights.
        lobster_matrices : LobsterMatricesLike
            Parsed Lobster matrices; `matrix_type` must be "coefficient".
        **kwargs
            Forwarded to the constructor (use_time_reversal, contract_function).

        Returns
        -------
        Self
            A new instance whose occupations and eigenvalues are truncated to
            the number of projected orbitals.

        Raises
        ------
        ValueError
            If the matrices are not coefficient matrices, the vasprun carries no
            eigenvalues, or the extracted arrays are mutually inconsistent in
            shape.
        """
        if lobster_matrices.matrix_type != "coefficient":
            raise ValueError(
                f"Expected coefficient matrices, got matrix_type={lobster_matrices.matrix_type!r}"
            )

        if (eigenvalues := vasprun.eigenvalues) is None:
            raise ValueError("`vasprun.xml` has no eigenvalues")

        if (efermi := vasprun.efermi) is None:
            raise ValueError("`vasprun.xml` has no Fermi energy")

        coefficients = lobster_matrices.matrices

        occupations = np.array([value[:, :, 1] for value in eigenvalues.values()])  # (s, k, n)
        check_dimensions(occupations, 3, "occupations")

        eigenvalues = np.array([value[:, :, 0] for value in eigenvalues.values()])  # (s, k, n)
        check_dimensions(eigenvalues, 3, "eigenvalues")

        k_points = np.array(vasprun.actual_kpoints)  # (k, 3)
        check_dimensions(k_points, 2, "k-points")

        if not np.allclose(lobster_matrices.kpoints, k_points):
            raise ValueError(
                "Lobster matrices and vasprun disagree on the k-points; they must "
                "describe the same k-point grid"
            )

        k_weights = np.array(vasprun.actual_kpoints_weights)  # (k,)
        check_dimensions(k_weights, 1, "k-weights")

        c_ns, c_nk, c_nu, c_nj = coefficients.shape
        o_ns, o_nk, o_nj = occupations.shape
        k_nk, _ = k_points.shape
        (kw_nk,) = k_weights.shape

        if c_ns != o_ns:
            raise ValueError(
                "coefficients and occupations disagree on the number of spin "
                f"channels ({c_ns} vs {o_ns})"
            )

        if c_nk != o_nk or o_nk != k_nk or k_nk != kw_nk:
            raise ValueError(
                "coefficients, occupations, k-points and k-weights disagree on "
                f"the number of k-points ({c_nk}, {o_nk}, {k_nk}, {kw_nk})"
            )

        if c_nu != c_nj:
            raise ValueError(
                f"coefficient matrices must be square, got {c_nu} orbitals and {c_nj} bands"
            )

        if o_nj < c_nj:
            raise ValueError(
                f"occupations cover fewer bands than the coefficient matrices ({o_nj} < {c_nj})"
            )

        if eigenvalues.shape != occupations.shape:
            raise ValueError(
                "eigenvalues and occupations must share a shape, got "
                f"{eigenvalues.shape} and {occupations.shape}"
            )

        occupations = occupations[..., :c_nj]
        eigenvalues = eigenvalues[..., :c_nj] - efermi

        return cls(coefficients, occupations, k_points, k_weights, eigenvalues, **kwargs)

    def get_density_matrix(
        self,
        translations: ArrayType | None = None,
        *,
        i_indices: ArrayType | slice = slice(None),
        j_indices: ArrayType | slice = slice(None),
    ) -> ArrayType:
        """Return the one-particle density matrix D_sij.

        D_sij = sum_kn w_k f_skn c_skin conj(c_skjn)

        Parameters
        ----------
        translations : ArrayType or None, optional
            Fractional real-space translations, shape (t, 3). When given, the
            density matrix is resolved per translation through a Bloch phase
            factor exp(-2 pi i sum_j k_j t_j) and gains a translation axis.
        i_indices, j_indices : ArrayType or slice, optional
            Orbital selections for the row (i) and column (j) axes. The default
            selects every orbital.

        Returns
        -------
        ArrayType
            Density matrix of shape (s, i, j), or (s, t, i, j) when translations
            are given. The real part is returned when use_time_reversal is set.
        """
        arguments = (
            self.k_weights,
            self.occupations,
            self.coefficients[:, :, i_indices, :],
            self.coefficients[:, :, j_indices, :].conj(),
        )

        if translations is None:
            density_matrix = self.contract_function("k,skn,skin,skjn->sij", *arguments)

            return density_matrix.real if self.use_time_reversal else density_matrix

        phase_factor = self.get_bloch_phase_factor(translations)

        density_matrix = self.contract_function(
            "k,skn,skin,skjn,tk->stij", *arguments, phase_factor
        )

        return density_matrix.real if self.use_time_reversal else density_matrix

    def get_gaussian_smeared_eigenvalues(
        self, energies: np.ndarray, sigma: float = 0.01
    ) -> np.ndarray:
        """Return Gaussian weights distributing each band over an energy grid.

        For every band the weight is exp(-((E - eps) / sigma) ** 2 / 2), an
        unnormalised Gaussian centred on the band eigenvalue eps and evaluated at
        each grid energy E.

        Parameters
        ----------
        energies : np.ndarray
            Energy grid, shape (b,).
        sigma : float, optional
            Standard deviation of the Gaussian, in the units of `energies`.

        Returns
        -------
        np.ndarray
            Bin weights, shape (s, b, k, n).

        Raises
        ------
        ValueError
            If no eigenvalues are stored on the instance.
        """
        if self.eigenvalues is None:
            raise ValueError(
                "eigenvalues are required for Gaussian smearing, but this instance has none"
            )

        return (
            1
            / (sigma * sqrt(pi))
            * self.xp.exp(
                -(((energies[None, :, None, None] - self.eigenvalues[:, None, :, :]) / sigma) ** 2)
                / 1
            )
        )  # (s, b, k, n)

    def get_binned_density_matrix(
        self,
        bins: ArrayType,
        translations: ArrayType | None = None,
        *,
        i_indices: ArrayType | Sequence[int] | slice = slice(None),
        j_indices: ArrayType | Sequence[int] | slice = slice(None),
        use_occupations: bool = False,
    ) -> ArrayType:
        """Return the density matrix resolved over energy bins.

        Like [get_density_matrix][bysque.compute.core.LobsterComputable.get_density_matrix],
        but every band n is distributed over energy bins b through the weights
        in `bins` (typically Gaussian-smeared occupations), yielding a spectral
        density matrix.

        Parameters
        ----------
        bins : ArrayType
            Per-band bin weights, shape (s, b, k, n).
        translations : ArrayType or None, optional
            Fractional real-space translations, shape (t, 3). Adds a leading
            translation axis when given.
        i_indices, j_indices : ArrayType or slice, optional
            Orbital selections for the row (i) and column (j) axes.
        use_occupations : bool, default False
            Whether to additionally weight each band by its occupation.

        Returns
        -------
        ArrayType
            Binned density matrix of shape (s, b, i, j), or (s, b, t, i, j) when
            translations are given.
        """
        arguments = [
            self.k_weights,
            bins,
            self.coefficients[:, :, i_indices, :],
            self.coefficients[:, :, j_indices, :].conj(),
        ]

        input_pattern = "k,sbkn,skin,skjn"

        if use_occupations:
            input_pattern += ",skn"
            arguments.append(self.occupations)

        if translations is None:
            return self.contract_function(f"{input_pattern}->sbij", *arguments)

        phase_factor = self.get_bloch_phase_factor(translations)

        binned_density_matrix = self.contract_function(
            f"{input_pattern},tk->sbtij", *arguments, phase_factor
        )

        return binned_density_matrix.real if self.use_time_reversal else binned_density_matrix

    def get_working_quantity_at(
        self,
        translations: ArrayType | None = None,
        spin_indices: ArrayType | slice = slice(None),
        k_points_indices: ArrayType | slice = slice(None),
        i_indices: ArrayType | slice = slice(None),
        j_indices: ArrayType | slice = slice(None),
        bands_indices: ArrayType | slice = slice(None),
        sum_over: tuple[Literal["s", "t", "k", "i", "j", "n"], ...] | None = None,
        *,
        use_occupations: bool = True,
    ) -> ArrayType:
        """Return a flexibly reduced coefficient product.

        Selects sub-ranges along each axis, forms the product
        c_skin conj(c_skjn) (optionally weighted by occupations and a Bloch
        phase), and removes the axes named in `sum_over` from the output.

        Parameters
        ----------
        translations : ArrayType or None, optional
            Fractional real-space translations, shape (t, 3).
        spin_indices, k_points_indices, bands_indices : ArrayType or slice, optional
            Selections along the spin (s), k-point (k) and band (n) axes.
        i_indices, j_indices : ArrayType or slice, optional
            Selections along the row (i) and column (j) orbital axes.
        sum_over : tuple of {"s", "t", "k", "i", "j", "n"} or None, optional
            Axis labels to contract away from the output. None keeps every axis.
        use_occupations : bool, default True
            Whether to weight each band by its occupation.

        Returns
        -------
        ArrayType
            The contracted quantity with the requested axes summed out.

        Raises
        ------
        ValueError
            If "t" is requested in `sum_over` while no translations are given.
        """
        input_pattern = "k,skin,skjn"
        output_pattern = "" if sum_over is None else "|".join(sum_over)

        coefficients = self.coefficients[spin_indices][:, :, k_points_indices][
            :, :, :, bands_indices
        ]

        i_coefficients = coefficients[:, i_indices, :, :]
        j_coefficients = coefficients[:, j_indices, :, :]

        arguments = [self.k_weights, i_coefficients, j_coefficients.conj()]

        if use_occupations:
            input_pattern += ",skn"

            occupations = self.occupations[spin_indices][:, k_points_indices][:, :, bands_indices]
            arguments.append(occupations)

        if translations is None:
            if "t" in output_pattern:
                raise ValueError(
                    "Cannot sum over the translation axis 't' when no translations are given"
                )

            return self.contract_function(
                f"{input_pattern}->{re.sub(output_pattern, '', 'skijn')}", *arguments
            )

        phase_factor = self.get_bloch_phase_factor(translations)

        return self.contract_function(
            f"{input_pattern}->{re.sub(output_pattern, '', 'skijn')}", *arguments, phase_factor
        )

    def get_real_density_matrix(
        self, translations: ArrayType | None, tolerance: float = 1.0e-4, **kwargs: Any
    ) -> ArrayType:
        """Return the real part of the density matrix, asserting it is real.

        Parameters
        ----------
        translations : np.ndarray or None
            Fractional real-space translations, shape (t, 3).
        tolerance : float, optional
            Maximum tolerated magnitude of any imaginary element.

        Returns
        -------
        np.ndarray
            The real part of the density matrix.

        Raises
        ------
        ValueError
            If any imaginary element exceeds `tolerance`.
        """
        density_matrix = self.get_density_matrix(translations, **kwargs)

        if (abs(density_matrix.imag) > tolerance).any():
            raise ValueError(f"The density matrix contains imaginary elements above {tolerance}")

        return density_matrix.real

    def get_bloch_phase_factor(self, translations: ArrayType) -> ArrayType:
        """Return the Bloch phase factor exp(-2 pi i sum_j k_j t_j).

        Resolves the density matrix per real-space translation: each k-point
        contributes the phase acquired over the translation.

        Parameters
        ----------
        translations : ArrayType
            Fractional real-space translations, shape (t, 3).

        Returns
        -------
        ArrayType
            Phase factor for every translation and k-point, shape (t, k).
        """
        return self.xp.exp(
            -2j * pi * self.contract_function("kj,tj->tk", self.k_points, translations)
        )


class COBIComputable(LobsterComputable[np.ndarray]):
    """Crystal orbital bond index (COBI/ICOBI) computations.

    Specialises [LobsterComputable][bysque.compute.core.LobsterComputable] for
    numpy arrays and adds the multi-centre bond indices obtained from products
    of real density-matrix elements.
    """

    def get_icobi_between(self, *indices: int, cells: np.ndarray | None = None) -> np.ndarray:
        """Return the integrated crystal orbital bond index between orbitals.

        Forms the product of off-diagonal density-matrix elements over the
        orbital pairs drawn from `indices`, scaled by factorial(len(indices)).
        Two indices use the ordered pairs (i, j) and (j, i); more than two use
        every unordered pair.

        Parameters
        ----------
        *indices : int
            Two or more orbital indices defining the interaction.
        cells : np.ndarray or None, optional
            Integer lattice cell of each index's periodic image, shape
            (len(indices), 3). Defaults to the home cell (all zeros) for every
            index. Pairwise cell differences set the real-space translations.

        Returns
        -------
        np.ndarray
            ICOBI values of shape (s,).
        """
        n_sites = len(indices)

        cells = np.zeros((n_sites, 3), int) if cells is None else np.asarray(cells)

        i_up, j_up = np.triu_indices(n_sites, 1)
        translations, inverse = np.unique(cells[j_up] - cells[i_up], axis=0, return_inverse=True)

        density_matrix = self.get_real_density_matrix(
            translations, i_indices=indices, j_indices=indices
        )

        n_spin = density_matrix.shape[0]

        tmp = np.zeros((n_spin, n_sites, n_sites), dtype=np.float64)

        tmp[..., i_up, j_up] = density_matrix[..., inverse, i_up, j_up]
        tmp[..., j_up, i_up] = tmp[..., i_up, j_up]

        pairs = get_cycle(range(n_sites))

        return np.prod([tmp[..., a, b] for a, b in pairs], axis=0) * factorial(n_sites)

    def get_cobi_between(
        self, *indices: int, energies: np.ndarray, cells: np.ndarray | None = None
    ) -> np.ndarray:
        """Return the energy-resolved crystal orbital bond index between orbitals.

        Like [get_icobi_between][bysque.compute.core.COBIComputable.get_icobi_between],
        but the final orbital pair is taken from the energy-binned density
        matrix, so the result is resolved over specified energies.

        Parameters
        ----------
        *indices : int
            Two or more orbital indices defining the interaction.
        energies : np.ndarray
            Per-band bin weights forwarded to
            [get_binned_density_matrix][bysque.compute.core.LobsterComputable.get_binned_density_matrix],
            shape (s, b, k, n).
        cells : np.ndarray or None, optional
            Integer lattice cell of each index's periodic image, shape
            (len(indices), 3). Defaults to the home cell (all zeros) for every
            index. Pairwise cell differences set the real-space translations.

        Returns
        -------
        np.ndarray
            COBI values resolved over energy bins, shape (s, b).
        """
        n_sites = len(indices)

        cells = np.zeros((n_sites, 3), int) if cells is None else np.asarray(cells)

        i_up, j_up = np.triu_indices(n_sites, 1)
        translations, inverse = np.unique(cells[j_up] - cells[i_up], axis=0, return_inverse=True)

        density_matrix = self.get_real_density_matrix(
            translations, i_indices=indices, j_indices=indices
        )
        binned_density_matrix = self.get_binned_density_matrix(
            energies, translations, i_indices=indices, j_indices=indices
        )

        n_spin, n_bins, _, _, _ = binned_density_matrix.shape

        tmp = np.zeros((n_spin, n_sites, n_sites), dtype=np.float64)

        tmp[..., i_up, j_up] = density_matrix[..., inverse, i_up, j_up]
        tmp[..., j_up, i_up] = tmp[..., i_up, j_up]

        tmp_ = np.zeros((n_spin, n_bins, n_sites, n_sites), dtype=np.float64)

        tmp_[..., i_up, j_up] = binned_density_matrix[..., inverse, i_up, j_up]
        tmp_[..., j_up, i_up] = tmp_[..., i_up, j_up]

        *pairs, last = get_cycle(range(n_sites))

        scalar = np.prod([tmp[..., a, b] for a, b in pairs], axis=0)

        return scalar[..., None] * tmp_[..., last[0], last[1]]

    def get_invariant_cobi_between(
        self, *indices: int, energies: np.ndarray, cells: np.ndarray | None = None
    ) -> np.ndarray:
        """Return the permutation-invariant energy-resolved crystal orbital bond index.

        Like [get_cobi_between][bysque.compute.core.COBIComputable.get_cobi_between],
        but sums the energy-binned contribution over every endpoint permutation
        from [get_invariant_permutations][bysque.utils.get_invariant_permutations]
        and scales by 2 factorial(len(indices) - 2), so the result no longer
        depends on the order of `indices`.

        Parameters
        ----------
        *indices : int
            Two or more orbital indices defining the interaction.
        energies : np.ndarray
            Per-band bin weights forwarded to
            [get_binned_density_matrix][bysque.compute.core.LobsterComputable.get_binned_density_matrix],
            shape (s, b, k, n).
        cells : np.ndarray or None, optional
            Integer lattice cell of each index's periodic image, shape
            (len(indices), 3). Defaults to the home cell (all zeros) for every
            index. Pairwise cell differences set the real-space translations.

        Returns
        -------
        np.ndarray
            COBI values resolved over energy bins, shape (s, b).
        """
        n_sites = len(indices)

        cells = np.zeros((n_sites, 3), int) if cells is None else np.asarray(cells)

        i_up, j_up = np.triu_indices(n_sites, 1)
        translations, inverse = np.unique(cells[j_up] - cells[i_up], axis=0, return_inverse=True)

        density_matrix = self.get_real_density_matrix(
            translations, i_indices=indices, j_indices=indices
        )
        binned_density_matrix = self.get_binned_density_matrix(
            energies, translations, i_indices=indices, j_indices=indices
        )

        n_spin, n_bins, _, _, _ = binned_density_matrix.shape

        tmp = np.zeros((n_spin, n_sites, n_sites), dtype=np.float64)

        tmp[..., i_up, j_up] = density_matrix[..., inverse, i_up, j_up]
        tmp[..., j_up, i_up] = tmp[..., i_up, j_up]

        tmp_ = np.zeros((n_spin, n_bins, n_sites, n_sites), dtype=np.float64)

        tmp_[..., i_up, j_up] = binned_density_matrix[..., inverse, i_up, j_up]
        tmp_[..., j_up, i_up] = tmp_[..., i_up, j_up]

        result = []
        for permutation in get_invariant_permutations(n_sites):
            *pairs, last = get_cycle(permutation)
            scalar = np.prod([tmp[..., a, b] for a, b in pairs], axis=0)
            result.append(scalar[..., None] * tmp_[..., last[0], last[1]])

        return np.sum(result, axis=0) * 2.0 * factorial(n_sites - 2)
