"""Global `conftest.py` configuration for pytest."""

from __future__ import annotations

from pathlib import Path

import pytest
from monty.io import zopen
from pymatgen.io.lobster.future.outputs.coxxcar import COBICAR
from pymatgen.io.lobster.future.outputs.icoxxlist import ICOBILIST, NcICOBILIST
from pymatgen.io.lobster.future.outputs.misc import LobsterMatrices
from pymatgen.io.vasp import Vasprun

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(name="carborane_lobster_matrices")
def carborane_lobster_matrices() -> LobsterMatrices:
    return LobsterMatrices(
        DATA_DIR / "pubchem-17750431-c2h12b10/coefficientMatricesLSO2.lobster.gz"
    )


@pytest.fixture(name="f2xe_lobster_matrices")
def f2xe_lobster_matrices() -> LobsterMatrices:
    return LobsterMatrices(DATA_DIR / "pubchem-83674-f2xe/coefficientMatricesLSO2.lobster.gz")


@pytest.fixture(name="carborane_raw_matrices")
def carborane_raw_matrices() -> list[str]:
    with zopen(
        DATA_DIR / "pubchem-17750431-c2h12b10/coefficientMatricesLSO2.lobster.gz", "rt"
    ) as f:
        return f.readlines()


@pytest.fixture(name="carborane_vasprun")
def carborane_vasprun() -> Vasprun:
    return Vasprun(DATA_DIR / "pubchem-17750431-c2h12b10/vasprun.xml.gz")


@pytest.fixture(name="f2xe_vasprun")
def f2xe_vasprun() -> Vasprun:
    return Vasprun(DATA_DIR / "pubchem-83674-f2xe/vasprun.xml.gz")


@pytest.fixture(name="carborane_cobicar")
def carborane_cobicar() -> COBICAR:
    return COBICAR(DATA_DIR / "pubchem-17750431-c2h12b10/COBICAR.lobster.gz")


@pytest.fixture(name="carborane_icobilist")
def carborane_icobilist() -> ICOBILIST:
    return ICOBILIST(DATA_DIR / "pubchem-17750431-c2h12b10/ICOBILIST.lobster.gz")


@pytest.fixture(name="carborane_nc_icobilist")
def carborane_nc_icobilist() -> NcICOBILIST:
    return NcICOBILIST(DATA_DIR / "pubchem-17750431-c2h12b10/NcICOBILIST.lobster.gz")


@pytest.fixture(name="f2xe_cobicar")
def f2xe_cobicar() -> COBICAR:
    return COBICAR(DATA_DIR / "pubchem-83674-f2xe/COBICAR.lobster.gz")


@pytest.fixture(name="f2xe_icobilist")
def f2xe_icobilist() -> ICOBILIST:
    return ICOBILIST(DATA_DIR / "pubchem-83674-f2xe/ICOBILIST.lobster.gz")


@pytest.fixture(name="f2xe_nc_icobilist")
def f2xe_nc_icobilist() -> NcICOBILIST:
    return NcICOBILIST(DATA_DIR / "pubchem-83674-f2xe/NcICOBILIST.lobster.gz")


@pytest.fixture(name="gete_cobicar")
def gete_cobicar() -> COBICAR:
    return COBICAR(DATA_DIR / "mp-938-ge3te3/COBICAR.lobster.gz")


@pytest.fixture(name="gete_lobster_matrices")
def gete_lobster_matrices() -> LobsterMatrices:
    return LobsterMatrices(DATA_DIR / "mp-938-ge3te3/coefficientMatricesLSO2.lobster.gz")


@pytest.fixture(name="gete_icobilist")
def gete_icobilist() -> ICOBILIST:
    return ICOBILIST(DATA_DIR / "mp-938-ge3te3/ICOBILIST.lobster.gz")

@pytest.fixture(name="gete_vasprun")
def gete_vasprun() -> Vasprun:
    return Vasprun(DATA_DIR / "mp-938-ge3te3/vasprun.xml.gz")


@pytest.fixture(name="gete_nc_icobilist")
def gete_nc_icobilist() -> NcICOBILIST:
    return NcICOBILIST(DATA_DIR / "mp-938-ge3te3/NcICOBILIST.lobster.gz")
