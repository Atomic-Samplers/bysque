from __future__ import annotations

from collections.abc import Mapping, Iterator, Sequence
from typing import Any, Protocol, Self, SupportsInt, runtime_checkable

import numpy as np
from numpy import complexfloating
from numpy.typing import NDArray
from numpy.typing import ArrayLike

LobsterMatrixData: dict[str, dict[SupportsInt | None, NDArray[complexfloating]]]


class VasprunLike(Protocol):

    @property
    def eigenvalues(self) -> Mapping[SupportsInt, np.ndarray] | None:
        ...

    efermi: float | None
    actual_kpoints: np.ndarray | None
    actual_kpoints_weights: np.ndarray | None

    @property
    def is_spin(self) -> bool:
        ...


class LobsterMatricesLike(Protocol):

    @property
    def spins(self) -> Sequence[SupportsInt] | None:
        ...

    matrix_type: str
    matrices: np.ndarray

    basis_functions: list[str]

    kpoints: list[tuple[float, ...]]


@runtime_checkable
class Shaped(Protocol):
    """Protocol for objects with a shape attribute."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Tuple of dimensions of the object."""
        ...


@runtime_checkable
class SliceIndexable(Shaped, Protocol):
    """Protocol for objects that can be indexed with slices."""

    def __getitem__(self, key: tuple[slice, ...] | slice, /) -> Self:
        """Return the item at the given index."""
        ...


@runtime_checkable
class NumPyConvertible(Protocol):
    """
    Protocol for objects compatible with NumPy's ``ndarray``.

    Extends [`MutableArrayLike`][] with `__array__` to allow
    transparent interoperation with the NumPy C-API.

    Parameters
    ----------
    T : type
        The scalar element type of the array.
    """

    def __array__(self, *args: Any, **kwargs: Any) -> Any:
        """Return a NumPy ``ndarray`` representation of this object."""
        ...


#@runtime_checkable
#class NumericArray(SliceIndexable, Protocol):
#    @property
#    def dtype(self) -> Any:
#        """Data type of the array elements."""
#        ...
#
#    @property
#    def real(self) -> Self:
#        """Return the real part of the array."""
#        ...
#
#    @property
#    def imag(self) -> Self:
#        """Return the imaginary part of the array."""
#        ...
#
#    def tolist(self) -> list[Any]:
#        """Convert the array to a (nested) list."""
#        ...


# --------------------------------------------------------------------------
# Core attributes
# --------------------------------------------------------------------------
@runtime_checkable
class HasDType(Protocol):
    @property
    def dtype(self) -> Any:
        """np.dtype | torch.dtype | jnp dtype — mutually unrelated types."""
        ...


@runtime_checkable
class HasShape(Protocol):
    @property
    def shape(self) -> tuple[int, ...]:
        """torch returns ``torch.Size``, which subclasses ``tuple[int, ...]``."""
        ...

    @property
    def ndim(self) -> int: ...


@runtime_checkable
class HasDevice(Protocol):
    """numpy >= 2.0, jax >= 0.4.27 (property, was a method before), torch any."""

    @property
    def device(self) -> Any: ...


@runtime_checkable
class HasItemSize(Protocol):
    """torch >= 2.1 only — both were added there for array-API parity."""

    @property
    def itemsize(self) -> int: ...

    @property
    def nbytes(self) -> int: ...


# --------------------------------------------------------------------------
# Interop / conversion
# --------------------------------------------------------------------------
@runtime_checkable
class SupportsArrayInterop(Protocol):
    def __array__(self, *args: Any, **kwargs: Any) -> Any:
        """Signature varies (numpy 2 passes ``copy=``), so stay permissive."""
        ...

    def tolist(self) -> Any:
        """Nested list; a bare scalar for 0-d input."""
        ...


@runtime_checkable
class SupportsScalarConversion(Protocol):
    """All raise for arrays with more than one element."""

    def item(self) -> Any: ...
    def __bool__(self) -> bool: ...
    def __int__(self) -> int: ...
    def __float__(self) -> float: ...
    def __complex__(self) -> complex: ...
    def __index__(self) -> int: ...


@runtime_checkable
class SupportsContainerOps(Protocol):
    def __len__(self) -> int:
        """Raises ``TypeError``/``IndexError`` on 0-d in all three."""
        ...

    def __iter__(self) -> Iterator[Any]: ...

    def __getitem__(self, key: Any, /) -> Any: ...


# --------------------------------------------------------------------------
# Operators
# --------------------------------------------------------------------------
@runtime_checkable
class SupportsArithmetic(Protocol):
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
    """Integer/bool dtypes only; float dtypes raise in all three."""

    def __and__(self, other: Any, /) -> Self: ...
    def __or__(self, other: Any, /) -> Self: ...
    def __xor__(self, other: Any, /) -> Self: ...
    def __lshift__(self, other: Any, /) -> Self: ...
    def __rshift__(self, other: Any, /) -> Self: ...
    def __invert__(self) -> Self: ...


@runtime_checkable
class SupportsComparison(Protocol):
    def __lt__(self, other: Any, /) -> Self: ...
    def __le__(self, other: Any, /) -> Self: ...
    def __gt__(self, other: Any, /) -> Self: ...
    def __ge__(self, other: Any, /) -> Self: ...

@runtime_checkable
class SupportsComplex(Protocol):
    @property
    def real(self) -> Self:
        """Return the real part of the array."""
        ...

    @property
    def imag(self) -> Self:
        """Return the imaginary part of the array."""
        ...

    def conj(self) -> Self:
        ...

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
    """The portable intersection of ndarray / Array / Tensor.

    Deliberately omits ``SupportsBitwise`` (dtype-conditional) and
    ``HasItemSize`` (torch >= 2.1). Mix them in explicitly if you need them::

        class MyArray(NumericArray, SupportsBitwise, Protocol): ...
    """

class ContractFunction[ArrayType: NumericArray](Protocol):
    def __call__(self, pattern: str, *args: ArrayType) -> ArrayType: ...


class ArrayNamespace[ArrayType: NumericArray](Protocol):

    def abs(self, x: ArrayType, /) -> ArrayType: ...
    def round(self, x: ArrayType, /) -> ArrayType: ...
    def ones_like(self, x: ArrayType, /) -> ArrayType: ...
    def all(self, x: ArrayType, axis: int, /) -> ArrayType: ...

    def exp(self, x: ArrayType, /) -> ArrayType: ...
