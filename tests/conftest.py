"""Global `conftest.py` configuration for pytest."""

from __future__ import annotations

from pathlib import Path

import pytest
from monty.io import zopen
from pymatgen.io.lobster.future.outputs.coxxcar import COBICAR, COHPCAR
from pymatgen.io.lobster.future.outputs.icoxxlist import ICOBILIST, ICOHPLIST, NcICOBILIST
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


@pytest.fixture(name="f2xe_cohpcar")
def f2xe_cohpcar() -> COHPCAR:
    return COHPCAR(DATA_DIR / "pubchem-83674-f2xe/COHPCAR.lobster.gz")


@pytest.fixture(name="f2xe_icobilist")
def f2xe_icobilist() -> ICOBILIST:
    return ICOBILIST(DATA_DIR / "pubchem-83674-f2xe/ICOBILIST.lobster.gz")


@pytest.fixture(name="f2xe_icohplist")
def f2xe_icohplist() -> ICOHPLIST:
    return ICOHPLIST(DATA_DIR / "pubchem-83674-f2xe/ICOHPLIST.lobster.gz")


@pytest.fixture(name="f2xe_hamilton_matrices")
def f2xe_hamilton_matrices() -> LobsterMatrices:
    return LobsterMatrices(DATA_DIR / "pubchem-83674-f2xe/hamiltonMatrices.lobster.gz", efermi=0.0)


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


@pytest.fixture(name="gete_cohpcar")
def gete_cohpcar() -> COHPCAR:
    return COHPCAR(DATA_DIR / "mp-938-ge3te3/COHPCAR.lobster.gz")


@pytest.fixture(name="gete_icohplist")
def gete_icohplist() -> ICOHPLIST:
    return ICOHPLIST(DATA_DIR / "mp-938-ge3te3/ICOHPLIST.lobster.gz")


@pytest.fixture(name="gete_hamilton_matrices")
def gete_hamilton_matrices() -> LobsterMatrices:
    return LobsterMatrices(DATA_DIR / "mp-938-ge3te3/hamiltonMatrices.lobster.gz", efermi=0.0)


@pytest.fixture(name="b2mg_cobicar")
def b2mg_cobicar() -> COBICAR:
    return COBICAR(DATA_DIR / "mp-763-b2mg/COBICAR.lobster.gz")


@pytest.fixture(name="b2mg_lobster_matrices")
def b2mg_lobster_matrices() -> LobsterMatrices:
    return LobsterMatrices(DATA_DIR / "mp-763-b2mg/coefficientMatricesLSO2.lobster.gz")


@pytest.fixture(name="b2mg_icobilist")
def b2mg_icobilist() -> ICOBILIST:
    return ICOBILIST(DATA_DIR / "mp-763-b2mg/ICOBILIST.lobster.gz")


@pytest.fixture(name="b2mg_vasprun")
def b2mg_vasprun() -> Vasprun:
    return Vasprun(DATA_DIR / "mp-763-b2mg/vasprun.xml.gz")


@pytest.fixture(name="b2mg_nc_icobilist")
def b2mg_nc_icobilist() -> NcICOBILIST:
    return NcICOBILIST(DATA_DIR / "mp-763-b2mg/NcICOBILIST.lobster.gz")


@pytest.fixture(name="b2mg_cohpcar")
def b2mg_cohpcar() -> COHPCAR:
    return COHPCAR(DATA_DIR / "mp-763-b2mg/COHPCAR.lobster.gz")


@pytest.fixture(name="b2mg_icohplist")
def b2mg_icohplist() -> ICOHPLIST:
    return ICOHPLIST(DATA_DIR / "mp-763-b2mg/ICOHPLIST.lobster.gz")


@pytest.fixture(name="b2mg_hamilton_matrices")
def b2mg_hamilton_matrices() -> LobsterMatrices:
    return LobsterMatrices(DATA_DIR / "mp-763-b2mg/hamiltonMatrices.lobster.gz", efermi=0.0)
