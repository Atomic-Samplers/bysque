"""Numpy backend for the tensor contractions used across `bysque.compute`."""

from __future__ import annotations

import numpy as np

from bysque.protocols import NumericArray


def contract[ArrayType: NumericArray](pattern: str, *arrays: ArrayType) -> ArrayType:
    """Contract `arrays` according to an einsum `pattern` using numpy.

    Thin wrapper around numpy.einsum with optimal contraction ordering. It is
    the default `contract_function` of
    [LobsterComputable][bysque.compute.core.LobsterComputable].

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
    ArrayType : NumericArray
        The array type of the operands and of the returned value.
    """
    return np.einsum(pattern, *arrays, optimize="optimal")
