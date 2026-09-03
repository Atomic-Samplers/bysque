"""Numpy backend for the tensor contractions used across `bysque.compute`."""

from __future__ import annotations

from itertools import product
from typing import Any, Literal, overload

import numpy as np

from bysque.protocols import NumericArray


def contract[ArrayType: NumericArray](pattern: str, *arrays: ArrayType) -> ArrayType:
    """
    Contract `arrays` according to an einsum `pattern` using numpy.

    Thin wrapper around numpy.einsum with optimal contraction ordering. It is the default
    `contract_function` of [LobsterComputable][bysque.compute.core.LobsterComputable] .

    Parameters
    ----------
    pattern : str
        Einsum subscript specification, for example "skin,skjn->sij".
    *arrays : ArrayType
        Operands whose axes match `pattern`.

    Returns
    -------
    ArrayType
        The contracted array.

    Generic
    -------
    ArrayType : NumericArray The array type of the operands and of the returned value.
    """
    return np.einsum(pattern, *arrays, optimize="optimal")


def compute_icoxx(
    pairs: list[tuple[int, int]],
    matrix: np.ndarray,
    last_matrix: np.ndarray,
    translation_indices: np.ndarray,
) -> np.ndarray:
    """
    Return the integrated multi-centre product over a cycle of orbital pairs.

    Symmetrises `matrix` and `last_matrix` by mirroring the upper-triangle elements selected at
    their per-pair translations, then multiplies the `matrix` element of every pair but the last by
    the `last_matrix` element of the last pair. This is the integrated bond index backing
    [get_icobi_between][bysque.compute.core.COBIComputable.get_icobi_between] and
    [get_icohp_between][bysque.compute.core.COHPComputable.get_icohp_between].

    Parameters
    ----------
    pairs : list of tuple of int
        Ordered (row, column) orbital pairs forming the interaction cycle. The last pair draws its
        factor from `last_matrix`, the rest from `matrix`.
    matrix : np.ndarray
        Translation-resolved matrix, shape (s, t, i, j).
    last_matrix : np.ndarray
        Translation-resolved matrix supplying the last pair's factor, shape (s, t, i, j).
    translation_indices : np.ndarray
        Translation index of each upper-triangle element, shape (i (i - 1) / 2,).

    Returns
    -------
    np.ndarray
        Integrated product, shape (s,).
    """
    s, _, i, j = matrix.shape

    i_up, j_up = np.triu_indices(i, 1)

    symmetric_matrix = np.zeros((s, i, j), dtype=np.float64)  # (s, i, j)
    symmetric_matrix[..., i_up, j_up] = matrix[..., translation_indices, i_up, j_up]
    symmetric_matrix[..., j_up, i_up] = symmetric_matrix[..., i_up, j_up]

    *pairs, last = pairs

    symmetric_last_matrix = np.zeros((s, i, j), dtype=np.float64)  # (s, i, j)
    symmetric_last_matrix[..., i_up, j_up] = last_matrix[..., translation_indices, i_up, j_up]
    symmetric_last_matrix[..., j_up, i_up] = symmetric_last_matrix[..., i_up, j_up]

    factors = [symmetric_matrix[..., row, column] for row, column in pairs]  # each (s,)

    return np.prod(factors, axis=0) * symmetric_last_matrix[..., *last]


def compute_k_resolved_coxx(coefficients: np.ndarray, hamiltonians: np.ndarray) -> np.ndarray:
    """This is more complicated than it looks, not implemented for now."""
    raise NotImplementedError
    return np.einsum("skin,skjn,skij->skn", coefficients.conj(), coefficients, hamiltonians)



def compute_coxx(
    pairs: list[tuple[int, int]],
    matrix: np.ndarray,
    binned_matrix: np.ndarray,
    translation_indices: np.ndarray,
) -> np.ndarray:
    """
    Return the energy-resolved multi-centre product over a cycle of orbital pairs.

    Like [compute_icoxx][bysque.compute.numpy_.compute_icoxx], but the last pair's factor is drawn
    from the energy-binned matrix, so the product gains an energy-bin axis. This backs
    [get_cobi_between][bysque.compute.core.COBIComputable.get_cobi_between] and
    [get_cohp_between][bysque.compute.core.COHPComputable.get_cohp_between].

    Parameters
    ----------
    pairs : list of tuple of int
        Ordered (row, column) orbital pairs forming the interaction cycle. The last pair draws its
        factor from `binned_matrix`, the rest from `matrix`.
    matrix : np.ndarray
        Translation-resolved matrix, shape (s, t, i, j).
    binned_matrix : np.ndarray
        Translation-resolved energy-binned matrix supplying the last pair's factor, shape
        (s, b, t, i, j).
    translation_indices : np.ndarray
        Translation index of each upper-triangle element, shape (i (i - 1) / 2,).

    Returns
    -------
    np.ndarray
        Energy-resolved product, shape (s, b).
    """
    s, b, _, i, j = binned_matrix.shape

    i_up, j_up = np.triu_indices(i, 1)

    symmetric_matrix = np.zeros((s, i, j), dtype=np.float64)  # (s, i, j)
    symmetric_matrix[..., i_up, j_up] = matrix[..., translation_indices, i_up, j_up]
    symmetric_matrix[..., j_up, i_up] = symmetric_matrix[..., i_up, j_up]

    symmetric_binned_matrix = np.zeros((s, b, i, j), dtype=np.float64)  # (s, b, i, j)
    symmetric_binned_matrix[..., i_up, j_up] = binned_matrix[..., translation_indices, i_up, j_up]
    symmetric_binned_matrix[..., j_up, i_up] = symmetric_binned_matrix[..., i_up, j_up]

    *pairs, last = pairs

    factors = [symmetric_matrix[..., row, column] for row, column in pairs]  # each (s,)
    scalar = np.prod(factors, axis=0)  # (s,)

    return scalar[..., None] * symmetric_binned_matrix[..., *last]


@overload
def get_translations(
    cells: np.ndarray, *, unique: Literal[True] = True
) -> tuple[np.ndarray, np.ndarray]: ...


@overload
def get_translations(cells: np.ndarray, *, unique: Literal[False]) -> np.ndarray: ...


def get_translations(cells: np.ndarray, *, unique: bool = True) -> Any:
    """
    Return the unique pairwise translations between lattice cells.

    For every upper-triangle pair (a, b) of `cells`, forms the difference cells[b] - cells[a] and
    returns the unique differences. With `unique` set, also returns the index mapping each pair to
    its unique translation.

    Parameters
    ----------
    cells : np.ndarray
        Integer lattice cells, shape (n, 3).
    unique : bool, default True
        Whether to also return the inverse indices mapping each pair to its unique translation.

    Returns
    -------
    np.ndarray
        Unique translations, shape (t, 3).
    np.ndarray
        Inverse index of each pair, shape (n (n - 1) / 2,). Returned only when `unique` is set.
    """
    i_up, j_up = np.triu_indices(len(cells), 1)

    return np.unique(cells[j_up] - cells[i_up], axis=0, return_inverse=unique)


def get_all_translations_k_mesh(k_mesh: tuple[int, int, int]) -> np.ndarray:
    """
    Return every integer translation spanned by a k-point mesh.

    Enumerates the Cartesian product range(k_mesh[0]) x range(k_mesh[1]) x range(k_mesh[2]).

    Parameters
    ----------
    k_mesh : tuple of int
        Number of k-points along each reciprocal axis.

    Returns
    -------
    np.ndarray
        Integer translations, shape (prod(k_mesh), 3).
    """
    return np.array(list(product(*[range(i) for i in k_mesh])))


def get_inphase_translations_and_weights(
    k_mesh: tuple[int, int, int], lattice: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the minimum-image translations of a k-point mesh and their weights.

    For every translation from
    [get_all_translations_k_mesh][bysque.compute.numpy_.get_all_translations_k_mesh], considers its
    periodic images shifted by integer multiples of `k_mesh` (each multiplier in range(-2, 3)),
    keeps those whose Cartesian length under `lattice` is minimal, and splits a unit weight equally
    among the degenerate minima. The weights therefore sum to prod(k_mesh).

    Parameters
    ----------
    k_mesh : tuple of int
        Number of k-points along each reciprocal axis.
    lattice : np.ndarray
        Real-space lattice vectors as rows, shape (3, 3).

    Returns
    -------
    np.ndarray
        Minimum-image translations, shape (m, 3).
    np.ndarray
        Weight of each translation, shape (m,).
    """
    all_shifts = np.array(list(product(range(-2, 3), repeat=3)))  # (125, 3)

    mesh = np.array(k_mesh)  # (3,)

    in_phase_translations, weights = [], []
    for translation in get_all_translations_k_mesh(k_mesh):
        candidates = translation + all_shifts * mesh  # (125, 3)

        distances = np.linalg.norm(candidates @ lattice, axis=1)  # (125,)
        minima = np.min(distances)

        selected = np.abs(distances - minima) < 1e-5 * max(minima, 1.0)

        for candidate in candidates[selected]:
            in_phase_translations.append(candidate)
            weights.append(1 / np.sum(selected))

    if not np.isclose(np.sum(weights), np.prod(k_mesh)):
        raise ValueError(
            "in-phase weights sum to "
            f"{np.sum(weights)}, expected the number of k-points {np.prod(k_mesh)}"
        )

    return np.array(in_phase_translations), np.array(weights)
