"""
This module intends to provide a way to express chains of transformation
elements (i.e. data and operations) in such a way that a query might use it
to check if a given transformation expression contains all of them in the
specified order. Anything that can act as a transformation element (e.g. Type,
Operation) should inherit from Chain.
"""

from __future__ import annotations

from abc import ABC
from typing import Union, Any


"""
`None` represents a skip, which may alternatively be written as an ellipsis,
that is, `...`.
"""
Skip = Union[None, type(Ellipsis)]  # type: Any


class Chain(ABC):
    """
    A chain captures some relevant aspects of a conceptual process, in terms of
    the sequence of elements that must occur in it.
    """
    # The same can also be described in terms of semantic linear time logic
    # formulae (SLTL), but since we will be using SPARQL to search through
    # workflows, the approach chosen here makes for a straightforward
    # translation.

    def __and__(self, other: Chain) -> Parallel:
        x = self if isinstance(self, Parallel) else Parallel(self)
        x.branches.append(other)
        return x

    def __rshift__(self, other: Union[Chain, Skip]) -> Serial:
        x = self if isinstance(self, Serial) else Serial(self)
        y = None if other == ... else other
        assert y is None or isinstance(y, Chain)
        x.sequence.append(y)
        return x


class Serial(Chain):
    """
    Describes which transformation elements must occur, in what order. A `None`
    value (or `...` ellipsis) indicates that we may skip zero or more steps.
    """

    def __init__(self, *sequence: Union[Chain, None]):
        self.sequence = list(sequence)


class Parallel(Chain):
    """
    Describes which transformation elements must occur conjunctively.
    """

    def __init__(self, *branches: Chain):
        self.branches = list(branches)
