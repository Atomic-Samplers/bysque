from itertools import combinations
from math import factorial
from typing import Any

import numpy as np
from numpy.testing import assert_allclose
from pymatgen.io.lobster.future.outputs.coxxcar import COBICAR, COHPCAR
from pymatgen.io.lobster.future.outputs.icoxxlist import ICOHPLIST, NcICOBILIST
from pymatgen.io.lobster.future.outputs.misc import LobsterMatrices
from pymatgen.io.vasp import Vasprun
from scipy.integrate import trapezoid

from bysque.compute.core import COBIComputable, COHPComputable, LobsterComputable
from bysque.utils import (
    get_invariant_permutations,
)


def test_from_vasp_pymatgen_objects(
    carborane_lobster_matrices: LobsterMatrices,
    carborane_vasprun: Vasprun,
    carborane_raw_matrices: list[str],
):

    lobster_computable = LobsterComputable.from_vasp_pymatgen_objects(
        carborane_vasprun, carborane_lobster_matrices
    )

    assert lobster_computable.coefficients.shape == (2, 1, 60, 60)
    assert lobster_computable.occupations.shape == (2, 1, 60)
    assert lobster_computable.k_points.shape == (1, 3)
    assert lobster_computable.k_weights.shape == (1,)

    assert_allclose(lobster_computable.k_points, [[0, 0, 0]])

    assert_allclose(lobster_computable.occupations[:, 0, :25], np.ones((2, 25)))
    assert_allclose(lobster_computable.occupations[:, 0, 25:], np.zeros((2, 35)))

    data_spin_up = np.genfromtxt(
        carborane_raw_matrices, skip_header=3, skip_footer=189, usecols=list(range(1, 61))
    )

    assert_allclose(lobster_computable.coefficients[0, 0, :, :].real, data_spin_up)


def test_density_matrix_computation_gamma(
    carborane_lobster_matrices: LobsterMatrices, carborane_vasprun: Vasprun
):
    lobster_computable = LobsterComputable[np.ndarray].from_vasp_pymatgen_objects(
        carborane_vasprun, carborane_lobster_matrices
    )

    density_matrix = lobster_computable.get_density_matrix()

    assert_allclose(density_matrix, density_matrix.swapaxes(2, 1))

    assert_allclose(np.trace(density_matrix[0, :, :]), 25)
    assert_allclose(np.trace(density_matrix[1, :, :]), 25)

    assert_allclose(
        density_matrix[0, :, :] @ density_matrix[0, :, :], density_matrix[0, :, :], atol=1.0e-3
    )
    assert_allclose(
        density_matrix[1, :, :] @ density_matrix[1, :, :], density_matrix[1, :, :], atol=1.0e-3
    )

    spectrum = np.linalg.eigvalsh(density_matrix[0, :, :])
    assert_allclose(np.sort(spectrum), [0.0] * 35 + [1.0] * 25, atol=1e-3)

    spectrum = np.linalg.eigvalsh(density_matrix[1, :, :])
    assert_allclose(np.sort(spectrum), [0.0] * 35 + [1.0] * 25, atol=1e-3)

    translated_density_matrix = lobster_computable.get_density_matrix(
        translations=np.array([[0, 0, 0], [1, 0, 0], [1, 1, -2], [100, -2133, 50]])
    )

    assert_allclose(
        translated_density_matrix[0, :, :] - density_matrix[0, :, :][None], 0.0, atol=1e-12
    )


def test_density_matrix_computation(gete_lobster_matrices: LobsterMatrices, gete_vasprun: Vasprun):
    lobster_computable = LobsterComputable[np.ndarray].from_vasp_pymatgen_objects(
        gete_vasprun, gete_lobster_matrices
    )

    translations = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 1], [6, 0, 0], [-4, 0, 0], [-1, -1, -1]])

    density_matrix = lobster_computable.get_density_matrix(translations=translations)

    assert density_matrix.shape == (2, 6, 39, 39)

    assert_allclose(density_matrix[:, 2, :, :], density_matrix[:, -1, :, :].swapaxes(-2, -1))

    assert_allclose(density_matrix[:, 1, :, :], density_matrix[:, 3, :, :])
    assert_allclose(density_matrix[:, 1, :, :], density_matrix[:, -2, :, :])

    assert_allclose(np.trace(density_matrix[0, 0, :, :]), 30)
    assert_allclose(np.trace(density_matrix[1, 0, :, :]), 30)

    lobster_computable.occupations = np.ones_like(lobster_computable.occupations)


def test_icobi_computation_gamma(
    carborane_lobster_matrices: LobsterMatrices,
    carborane_vasprun: Vasprun,
    carborane_nc_icobilist: NcICOBILIST,
):
    icobi_computable = COBIComputable.from_vasp_pymatgen_objects(
        carborane_vasprun, carborane_lobster_matrices
    )

    for interaction in carborane_nc_icobilist.interactions:
        if not any(interaction["orbitals"]):
            continue

        their_icobi = np.array(list(interaction["icoxx"].values()))

        indices = [
            carborane_lobster_matrices.basis_functions.index(f"{i}_{j}")
            for i, j in zip(interaction["centers"], interaction["orbitals"], strict=True)
        ]

        our_icobi = icobi_computable.get_icobi_between(*indices)
        assert_allclose(our_icobi, their_icobi, atol=1.0e-3, rtol=5.0e-2)


def test_cobi_computation_gamma(
    f2xe_lobster_matrices: LobsterMatrices,
    f2xe_vasprun: Vasprun,
    f2xe_cobicar: COBICAR,
):
    cobi_computable = COBIComputable.from_vasp_pymatgen_objects(
        f2xe_vasprun, f2xe_lobster_matrices
    )

    energies = cobi_computable.get_gaussian_smeared_eigenvalues(f2xe_cobicar.energies, sigma=0.1)

    for interaction in f2xe_cobicar.interactions:
        if not any(interaction["orbitals"]):
            continue

        their_cobi = np.array(list(interaction["coxx"].values()))

        indices = [
            f2xe_lobster_matrices.basis_functions.index(f"{i}_{j}")
            for i, j in zip(interaction["centers"], interaction["orbitals"], strict=True)
        ]

        our_cobi = cobi_computable.get_invariant_cobi_between(*indices, energies=energies)

        assert_allclose(our_cobi, their_cobi, atol=1.0e-1, rtol=1.0e-2)


def test_icobi_computation(
    gete_lobster_matrices: LobsterMatrices,
    gete_vasprun: Vasprun,
    gete_nc_icobilist: NcICOBILIST,
):
    icobi_computable = COBIComputable.from_vasp_pymatgen_objects(
        gete_vasprun, gete_lobster_matrices
    )

    current_cells = None
    for interaction in gete_nc_icobilist.interactions:
        if not any(interaction["orbitals"]):
            current_cells = np.array(interaction["cells"])
            continue

        their_icobi = np.array(list(interaction["icoxx"].values()))

        indices = [
            gete_lobster_matrices.basis_functions.index(f"{i}_{j}")
            for i, j in zip(interaction["centers"], interaction["orbitals"], strict=True)
        ]

        our_icobi = icobi_computable.get_icobi_between(*indices, cells=current_cells)

        assert_allclose(our_icobi, their_icobi, atol=1.0e-6)


def test_cobi_computation(
    gete_lobster_matrices: LobsterMatrices,
    gete_vasprun: Vasprun,
    gete_cobicar: COBICAR,
    gete_nc_icobilist: NcICOBILIST,
):
    cobi_computable = COBIComputable.from_vasp_pymatgen_objects(
        gete_vasprun, gete_lobster_matrices
    )

    energies = cobi_computable.get_gaussian_smeared_eigenvalues(gete_cobicar.energies, sigma=0.01)

    energy_range = gete_cobicar.energies
    (below_fermi,) = np.where(energy_range < 0)

    current_cells = None

    cobi_interactions = gete_cobicar.interactions
    icobi_interactions = gete_nc_icobilist.interactions

    for cobi_interaction, icobi_interaction in zip(
        cobi_interactions, icobi_interactions, strict=True
    ):
        if not any(cobi_interaction["orbitals"]):
            current_cells = np.array(icobi_interaction["cells"])
            continue

        assert current_cells is not None
        assert_allclose(current_cells, cobi_interaction["cells"])

        assert cobi_interaction["centers"] == icobi_interaction["centers"]
        assert cobi_interaction["orbitals"] == icobi_interaction["orbitals"]

        indices = [
            gete_lobster_matrices.basis_functions.index(f"{i}_{j}")
            for i, j in zip(cobi_interaction["centers"], cobi_interaction["orbitals"], strict=True)
        ]

        our_icobi = cobi_computable.get_icobi_between(*indices, cells=current_cells)
        our_cobi = cobi_computable.get_invariant_cobi_between(
            *indices, energies=energies, cells=current_cells
        )

        integrated_icobi = trapezoid(our_cobi[:, below_fermi], energy_range[below_fermi])

        assert_allclose(our_icobi, integrated_icobi, atol=1.0e-3, rtol=1.0e-2)


def test_variant_invariant_computation(
    gete_lobster_matrices: LobsterMatrices,
    gete_vasprun: Vasprun,
):
    cobi_computable = COBIComputable.from_vasp_pymatgen_objects(
        gete_vasprun, gete_lobster_matrices
    )

    n_sites = cobi_computable.coefficients.shape[-1]

    to_compute: list[Any] = list(combinations(range(n_sites), 2))[::20]
    to_compute.extend(list(combinations(range(n_sites), 3))[::200])

    energies = cobi_computable.get_gaussian_smeared_eigenvalues(
        np.arange(-20, 10, 0.01), sigma=0.01
    )

    for indices in to_compute:
        our_invariant_cobi = cobi_computable.get_invariant_cobi_between(
            *indices, energies=energies
        )

        our_cobi = []
        for permutation in get_invariant_permutations(len(indices)):
            new_indices = [indices[idx] for idx in permutation]
            our_cobi.append(cobi_computable.get_cobi_between(*new_indices, energies=energies))

        our_cobi = np.sum(our_cobi, axis=0) * factorial(len(indices)) / len(our_cobi)

        assert_allclose(our_cobi, our_invariant_cobi, atol=1.0e-12)


def test_from_vasp_pymatgen_objects_hamiltonian(
    f2xe_lobster_matrices: LobsterMatrices,
    f2xe_hamilton_matrices: LobsterMatrices,
    f2xe_vasprun: Vasprun,
):

    lobster_computable = COHPComputable.from_vasp_pymatgen_objects(
        f2xe_vasprun, f2xe_lobster_matrices, hamilton_matrices=f2xe_hamilton_matrices
    )

    assert lobster_computable.k_mesh == (1, 1, 1)
    assert_allclose(lobster_computable.lattice, np.eye(3) * 15)


def test_icohp_computation_gamma(
    f2xe_lobster_matrices: LobsterMatrices,
    f2xe_hamilton_matrices: LobsterMatrices,
    f2xe_vasprun: Vasprun,
    f2xe_icohplist: ICOHPLIST,
):
    icohp_computable = COHPComputable.from_vasp_pymatgen_objects(
        f2xe_vasprun, f2xe_lobster_matrices, hamilton_matrices=f2xe_hamilton_matrices
    )

    for interaction in f2xe_icohplist.interactions:
        if not any(interaction["orbitals"]):
            continue

        their_icohp = np.array(list(interaction["icoxx"].values()))

        indices = [
            f2xe_lobster_matrices.basis_functions.index(f"{i}_{j}")
            for i, j in zip(interaction["centers"], interaction["orbitals"], strict=True)
        ]

        our_icohp = icohp_computable.get_icohp_between(*indices)

        assert_allclose(our_icohp, their_icohp, atol=1.0e-3, rtol=5.0e-2)


def test_icohp_computation(
    gete_lobster_matrices: LobsterMatrices,
    gete_hamilton_matrices: LobsterMatrices,
    gete_vasprun: Vasprun,
    gete_icohplist: ICOHPLIST,
):
    icohp_computable = COHPComputable.from_vasp_pymatgen_objects(
        gete_vasprun, gete_lobster_matrices, hamilton_matrices=gete_hamilton_matrices
    )

    current_cells = None
    for interaction in gete_icohplist.interactions:
        if not any(interaction["orbitals"]):
            current_cells = np.array(interaction["cells"])
            continue

        their_icohp = np.array(list(interaction["icoxx"].values()))

        indices = [
            gete_lobster_matrices.basis_functions.index(f"{i}_{j}")
            for i, j in zip(interaction["centers"], interaction["orbitals"], strict=True)
        ]

        our_icohp = icohp_computable.get_icohp_between(*indices, cells=current_cells)

        assert_allclose(our_icohp, their_icohp, atol=1.0e-3, rtol=5.0e-2)


def test_cohp_computation_gamma(
    f2xe_lobster_matrices: LobsterMatrices,
    f2xe_hamilton_matrices: LobsterMatrices,
    f2xe_vasprun: Vasprun,
    f2xe_cohpcar: COHPCAR,
):
    cobi_computable = COHPComputable.from_vasp_pymatgen_objects(
        f2xe_vasprun, f2xe_lobster_matrices, hamilton_matrices=f2xe_hamilton_matrices
    )

    energy_range = f2xe_cohpcar.energies
    (below_fermi,) = np.where(energy_range < 0)

    energies = cobi_computable.get_gaussian_smeared_eigenvalues(f2xe_cohpcar.energies, sigma=0.1)

    for interaction in f2xe_cohpcar.interactions:
        if not any(interaction["orbitals"]):
            continue

        their_cobi = np.array(list(interaction["coxx"].values()))

        indices = [
            f2xe_lobster_matrices.basis_functions.index(f"{i}_{j}")
            for i, j in zip(interaction["centers"], interaction["orbitals"], strict=True)
        ]

        our_cobi = cobi_computable.get_cohp_between(*indices, energies=energies)

        our_icobis = trapezoid(our_cobi[:, below_fermi], energy_range[below_fermi])
        their_icobis = trapezoid(their_cobi[:, below_fermi], energy_range[below_fermi])

        assert_allclose(our_cobi, their_cobi, atol=1.0e-2, rtol=0.25)
        assert_allclose(our_icobis, their_icobis, atol=1.0e-4, rtol=1.0e-5)


def test_cohp_computation(
    gete_lobster_matrices: LobsterMatrices,
    gete_hamilton_matrices: LobsterMatrices,
    gete_vasprun: Vasprun,
    gete_cohpcar: COHPCAR,
    gete_icohplist: ICOHPLIST,
):
    cohp_computable = COHPComputable.from_vasp_pymatgen_objects(
        gete_vasprun, gete_lobster_matrices, hamilton_matrices=gete_hamilton_matrices
    )

    energies = cohp_computable.get_gaussian_smeared_eigenvalues(gete_cohpcar.energies, sigma=0.1)

    energy_range = gete_cohpcar.energies

    # Gaussian smearing will make curve spills above Fermi,
    # while LOBSTER tetrahedron integration will not, let's put "Fermi" at 0.5 eV.
    (below_fermi,) = np.where(energy_range < 0.5)

    current_cells = None

    cohp_interactions = gete_cohpcar.interactions
    icohp_interactions = gete_icohplist.interactions

    for cohp_interaction, icohp_interaction in zip(
        cohp_interactions[1:], icohp_interactions, strict=True
    ):
        if not any(cohp_interaction["orbitals"]):
            current_cells = np.array(icohp_interaction["cells"])
            continue

        assert current_cells is not None

        assert cohp_interaction["centers"] == icohp_interaction["centers"]
        assert cohp_interaction["orbitals"] == icohp_interaction["orbitals"]

        indices = [
            gete_lobster_matrices.basis_functions.index(f"{i}_{j}")
            for i, j in zip(cohp_interaction["centers"], cohp_interaction["orbitals"], strict=True)
        ]

        our_cohp = cohp_computable.get_cohp_between(
            *indices, energies=energies, cells=current_cells
        )
        their_cohp = np.array(list(cohp_interaction["coxx"].values()))

        our_icobis = trapezoid(our_cohp[:, below_fermi], energy_range[below_fermi])
        their_icobis = trapezoid(their_cohp[:, below_fermi], energy_range[below_fermi])

        assert_allclose(our_icobis, their_icobis, atol=1.0e-5, rtol=1.0e-6)


def test_hamiltonian_interpolation(
    gete_lobster_matrices: LobsterMatrices,
    gete_hamilton_matrices: LobsterMatrices,
    gete_vasprun: Vasprun,
):
    cohp_computable = COHPComputable.from_vasp_pymatgen_objects(
        gete_vasprun, gete_lobster_matrices, hamilton_matrices=gete_hamilton_matrices
    )

    new_hamilton_matrices = cohp_computable.get_hamiltonian_at(cohp_computable.k_points)

    assert_allclose(new_hamilton_matrices.real, cohp_computable.hamiltonians.real, atol=1.0e-12)

    # Suspiciously high tolerance needed here.
    assert_allclose(new_hamilton_matrices, cohp_computable.hamiltonians, atol=1.0e-4)

    random_k_point = np.array([[0.137, -0.061, 0.213]])
    tmp = cohp_computable.get_hamiltonian_at(random_k_point)
    tmp_ = cohp_computable.get_hamiltonian_at(-random_k_point)
    assert_allclose(tmp, tmp_.conj(), atol=1e-12)


def test_band_interpolation(
    gete_lobster_matrices: LobsterMatrices,
    gete_hamilton_matrices: LobsterMatrices,
    gete_vasprun: Vasprun,
):
    cohp_computable = COHPComputable.from_vasp_pymatgen_objects(
        gete_vasprun, gete_lobster_matrices, hamilton_matrices=gete_hamilton_matrices
    )

    eigenvalues, coefficients = cohp_computable.get_bands_at(cohp_computable.k_points)

    old_eigenvalues = cohp_computable.eigenvalues
    assert old_eigenvalues is not None

    assert_allclose(eigenvalues - gete_vasprun.efermi, old_eigenvalues, atol=1.0e-6)

    density_matrix = cohp_computable.get_real_density_matrix(None)
    cohp_computable.coefficients = coefficients
    assert_allclose(cohp_computable.get_real_density_matrix(None), density_matrix, atol=1.0e-6)
