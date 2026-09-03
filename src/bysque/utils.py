"""Helpers to prepare VASP quantities for bond-index computations."""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    import numpy as np


def get_cycle(indices: Sequence[int]) -> list[tuple[int, int]]:
    """
    Return the consecutive ordered pairs of a closed cycle over `indices` .

    Pairs each index with its successor, wrapping the last index back to the first, so that n
    indices yield n pairs.

    Parameters
    ----------
    indices : Sequence[int]
        The indices to walk through, in cycle order.

    Returns
    -------
    list of tuple of int
        The pairs (indices[i], indices[i + 1 mod n]) for i in range(n).

    Examples
    --------
    >>> get_cycle([0, 1, 2])
    [(0, 1), (1, 2), (2, 0)]
    """
    size = len(indices)

    return [(indices[i], indices[(i + 1) % size]) for i in range(size)]


def get_invariant_permutations(count: int) -> Generator[list[int], None, None]:
    """
    Yield the orderings of range(count) anchored on each pair of endpoints.

    For every unordered pair (first, last) drawn from range(count), yields the ordering that places
    first at the front and last at the back, keeping the remaining indices in their natural order
    in between. The number of yielded orderings is therefore the number of pairs, count * (count -
    1) / 2.

    Parameters
    ----------
    count : int
        The number of indices to permute.

    Yields
    ------
    list of int
        An ordering [first, ...remaining..., last] for each endpoint pair.

    Examples
    --------
    >>> list(get_invariant_permutations(3))
    [[0, 2, 1], [0, 1, 2], [1, 0, 2]]
    """
    indices = list(range(count))

    for pair in combinations(range(count), 2):
        first, last = pair

        yield [
            indices[first],
            *[idx for idx in indices if idx not in pair],
            indices[last],
        ]


def check_dimensions(array: np.ndarray, ndim: int, name: str) -> None:
    """Raise a ValueError if `array` does not have exactly `ndim` axes."""
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional, got {array.ndim} dimensions")
