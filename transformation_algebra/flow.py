"""
Defines the generic `Flow` class.
"""

from __future__ import annotations

from typing import Iterator, TypeVar, Generic, Union

T = TypeVar('T')


class Flow(Generic[T]):
    """
    A flow is used to capture a sequence of steps, possibly skipping over any
    number of them and branching into disjunctions or conjunctions.

    The sequence is 'reversed' (that is, specified from end to beginning). This
    allows for a convenient tree-like notation, but it may trip you up.

    A shorthand notation allows lists to be interpreted as `SERIES` and tuples
    as `LINKED` series.

    For example, the following flow of type `Flow[int]` describes a flow ending
    in 9, in which step 3 directly combines the outputs from steps 1 and 2.

        [9, (3, AND(1, 2))]
    """

    def __init__(self, *items: FlowShorthand[T]):
        if len(items) < 2:
            raise RuntimeError("Expected at least two items.")
        self.items: list[Flow1[T]] = [Flow.shorthand(x) for x in items]

    def __iter__(self) -> Iterator[Flow1[T]]:
        return iter(self.items)

    @staticmethod
    def shorthand(value: FlowShorthand[T]) -> Flow1[T]:
        """
        Translate shorthand data structures (list for `SERIES`, tuples for
        `LINKED`) to real flows.
        """
        if isinstance(value, list):
            return SERIES(*value)
        elif isinstance(value, tuple):
            return LINKED(*value)
        else:
            return value


class LINKED(Flow[T]):
    """
    A series linked in a chain, where each node must *immediately* follow the
    one after.
    """
    pass


class SERIES(Flow[T]):
    """
    A sequence where each node must follow the one after, but not necessarily
    immediately.
    """
    pass


class AND(Flow[T]):
    """
    Indicate which paths must occur conjunctively. That is, every path must
    occur somewhere --- possibly on distinct, parallel branches, possibly on
    the same branch.
    """
    pass


class OR(Flow[T]):
    """
    Indicate which paths can occur disjunctively. That is, at least one path
    must occur somewhere.
    """
    pass


"""
A shorthand for specifying `Flow`s.
"""
Flow1 = Union[T, Flow[T]]
FlowShorthand = Union[Flow1[T], list[Flow1[T]], tuple[Flow1[T]]]