"""
Structural types describing the array and pymatgen objects `bysque` consumes.

Two families of protocols live here. The `*Like` protocols capture the subset of pymatgen's
`Vasprun` and `LobsterMatrices` attributes that
[LobsterComputable][bysque.compute.core.LobsterComputable] reads, so any object exposing those
attributes is accepted. The array protocols capture the portable intersection of numpy.ndarray, jax
Array and torch Tensor, letting a single implementation serve every backend without branching on
its concrete type.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, Self, SupportsInt, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    import numpy as np


class VasprunLike(Protocol):
    """The pymatgen `Vasprun` attributes and methods `bysque` requires."""

    @property
    def eigenvalues(self) -> Mapping[SupportsInt, np.ndarray] | None: ...

    efermi: float | None
    actual_kpoints: np.ndarray | None
    actual_kpoints_weights: np.ndarray | None

    @property
    def is_spin(self) -> bool: ...


class LobsterMatricesLike(Protocol):
    """The pymatgen `LobsterMatrices` attributes read when building a computable."""

    @property
    def spins(self) -> Sequence[SupportsInt] | None: ...

    matrix_type: str
    matrices: np.ndarray

    basis_functions: list[str]

    kpoints: list[tuple[float, ...]]


# --------------------------------------------------------------------------
# Core attributes
# --------------------------------------------------------------------------
@runtime_checkable
class HasDType(Protocol):
    """Array exposing a `dtype`."""

    @property
    def dtype(self) -> Any:
        """np.dtype | torch.dtype | jnp dtype — mutually unrelated types."""
        ...


@runtime_checkable
class HasShape(Protocol):
    """Array exposing `shape` and `ndim`."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Torch returns ``torch.Size``, which subclasses ``tuple[int, ...]``."""
        ...

    @property
    def ndim(self) -> int: ...


@runtime_checkable
class HasDevice(Protocol):
    """
    Array exposing a `device` .

    numpy >= 2.0, jax >= 0.4.27 (property, was a method before), torch any.
    """

    @property
    def device(self) -> Any: ...


@runtime_checkable
class HasItemSize(Protocol):
    """
    Array exposing `itemsize` and `nbytes` .

    torch >= 2.1 only — both were added there for array-API parity.
    """

    @property
    def itemsize(self) -> int: ...

    @property
    def nbytes(self) -> int: ...


# --------------------------------------------------------------------------
# Interop / conversion
# --------------------------------------------------------------------------
@runtime_checkable
class SupportsArrayInterop(Protocol):
    """Array convertible to a numpy array or to a nested list."""

    def __array__(self, *args: Any, **kwargs: Any) -> Any:
        """Signature varies (numpy 2 passes ``copy=``), so stay permissive."""
        ...

    def tolist(self) -> Any:
        """Nested list; a bare scalar for 0-d input."""
        ...


@runtime_checkable
class SupportsScalarConversion(Protocol):
    """
    Array convertible to a Python scalar.

    All raise for arrays with more than one element.
    """

    def item(self) -> Any: ...
    def __bool__(self) -> bool: ...
    def __int__(self) -> int: ...
    def __float__(self) -> float: ...
    def __complex__(self) -> complex: ...
    def __index__(self) -> int: ...


@runtime_checkable
class SupportsContainerOps(Protocol):
    """Array supporting length, iteration and indexing."""

    def __len__(self) -> int:
        """Return the length; raises `TypeError`/`IndexError` on 0-d in all three."""
        ...

    def __iter__(self) -> Iterator[Any]: ...

    def __getitem__(self, key: Any, /) -> Any: ...


# --------------------------------------------------------------------------
# Operators
# --------------------------------------------------------------------------
@runtime_checkable
class SupportsArithmetic(Protocol):
    """Array supporting the elementwise arithmetic operators."""

    def __add__(self, other: Any, /) -> Self: ...
    def __radd__(self, other: Any, /) -> Self: ...
    def __sub__(self, other: Any, /) -> Self: ...
    def __rsub__(self, other: Any, /) -> Self: ...
    def __mul__(self, other: Any, /) -> Self: ...
    def __rmul__(self, other: Any, /) -> Self: ...
    def __truediv__(self, other: Any, /) -> Self: ...
    def __rtruediv__(self, other: Any, /) -> Self: ...
    def __floordiv__(self, other: Any, /) -> Self: ...
    def __rfloordiv__(self, other: Any, /) -> Self: ...
    def __mod__(self, other: Any, /) -> Self: ...
    def __rmod__(self, other: Any, /) -> Self: ...
    def __pow__(self, other: Any, /) -> Self: ...
    def __rpow__(self, other: Any, /) -> Self: ...
    def __matmul__(self, other: Any, /) -> Self: ...
    def __rmatmul__(self, other: Any, /) -> Self: ...
    def __neg__(self) -> Self: ...
    def __pos__(self) -> Self: ...
    def __abs__(self) -> Self: ...


@runtime_checkable
class SupportsBitwise(Protocol):
    """
    Array supporting the elementwise bitwise operators.

    Integer/bool dtypes only; float dtypes raise in all three.
    """

    def __and__(self, other: Any, /) -> Self: ...
    def __or__(self, other: Any, /) -> Self: ...
    def __xor__(self, other: Any, /) -> Self: ...
    def __lshift__(self, other: Any, /) -> Self: ...
    def __rshift__(self, other: Any, /) -> Self: ...
    def __invert__(self) -> Self: ...


@runtime_checkable
class SupportsComparison(Protocol):
    """Array supporting the elementwise comparison operators."""

    def __lt__(self, other: Any, /) -> Self: ...
    def __le__(self, other: Any, /) -> Self: ...
    def __gt__(self, other: Any, /) -> Self: ...
    def __ge__(self, other: Any, /) -> Self: ...


@runtime_checkable
class SupportsComplex(Protocol):
    """Array exposing real and imaginary parts and conjugation."""

    @property
    def real(self) -> Self:
        """Return the real part of the array."""
        ...

    @property
    def imag(self) -> Self:
        """Return the imaginary part of the array."""
        ...

    def conj(self) -> Self: ...


# --------------------------------------------------------------------------
# Aggregate
# --------------------------------------------------------------------------
@runtime_checkable
class NumericArray(
    HasDType,
    HasShape,
    HasDevice,
    SupportsArrayInterop,
    SupportsScalarConversion,
    SupportsContainerOps,
    SupportsArithmetic,
    SupportsComparison,
    SupportsComplex,
    Protocol,
):
    """
    The portable intersection of ndarray, jax Array and torch Tensor.

    Deliberately omits [SupportsBitwise][bysque.protocols.SupportsBitwise] (dtype-conditional) and
    [HasItemSize][bysque.protocols.HasItemSize] (torch >= 2.1). Mix them in explicitly if you need
    them::

    class MyArray(NumericArray, SupportsBitwise, Protocol): ...
    """


class ContractFunction[ArrayType: NumericArray](Protocol):
    """
    Callable performing an einsum-style contraction over arrays.

    Generic
    -------
    ArrayType : NumericArray The array type of the operands and of the returned value; a single
    contraction signature then serves every backend.
    """

    def __call__(self, pattern: str, *args: ArrayType) -> ArrayType: ...


class ArrayNamespace[ArrayType: NumericArray](Protocol):
    """
    Namespace of elementwise array operations, such as the numpy module.

    Generic
    -------
    ArrayType : NumericArray The array type accepted and returned by every operation, so the
    namespace stays bound to a single backend.
    """

    def abs(self, x: ArrayType, /) -> ArrayType: ...
    def round(self, x: ArrayType, /) -> ArrayType: ...
    def ones_like(self, x: ArrayType, /) -> ArrayType: ...
    def all(self, x: ArrayType, axis: int, /) -> ArrayType: ...

    def exp(self, x: ArrayType, /) -> ArrayType: ...
