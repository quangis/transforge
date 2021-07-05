"""
This module intends to provide a way to express chains of transformation
elements (i.e. data and operations) in such a way that a query might use it to
check if a given transformation expression contains all of them in the
specified order. Anything that can act as a single transformation element (e.g.
data or operations) should inherit from Unit.
"""

from __future__ import annotations

from abc import ABC
from typing import Union, Any


"""
`None` represents a skip, which may alternatively be written as an ellipsis,
that is, `...`.
"""
Skip = Union[None, type(Ellipsis)]  # type: Any


class Flow(ABC):
    """
    A flow captures some relevant aspects of a conceptual process, in terms of
    the sequence of elements that must occur in it. For example, the following
    flow holds that there must be datatypes A and B that are fed to an
    operation f that eventually results in a datatype C:

    C << ... << f << (A & B)
    """
    # The same can also be described in terms of semantic linear time logic
    # formulae (SLTL), but since we will be using SPARQL to search through
    # workflows, the approach chosen here makes for a straightforward
    # translation.

    def __and__(self, other: Flow) -> Parallel:
        if isinstance(self, Parallel):
            self.branches.append(other)
            return self
        else:
            return Parallel(self, other)

    def __lshift__(self, other: Union[Flow, Skip]) -> Serial:
        x = None if other == ... else other
        if isinstance(self, Serial):
            self.sequence.append(x)
            return self
        else:
            return Serial(self, x)


class Unit(Flow, ABC):
    """
    A single element of a Flow.
    """
    pass


class Serial(Flow):
    """
    Describes which transformation elements must occur, in what order. A `None`
    value (or `...` ellipsis) indicates that we may skip zero or more steps.
    """

    def __init__(self, *sequence: Union[Flow, None]):
        # assert len(list(x for x in sequence if x)) > 1
        self.sequence = list(sequence)

    # def pairs(self) -> Iterator[Tuple[Optional[Flow], bool, Optional[Flow]]]:
    #     """
    #     Return a pairwise iterator of connections between units in the series,
    #     with a boolean indicating whether the connection is direct or not.
    #     """


class Parallel(Flow):
    """
    Describes which transformation elements must occur conjunctively.
    """

    def __init__(self, *branches: Flow):
        # assert len(list(x for x in branches if x)) > 1
        self.branches = list(branches)
