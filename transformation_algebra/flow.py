"""
Defines the generic `Flow` class.
"""

from __future__ import annotations

from itertools import chain, product
from typing import Iterator, TypeVar, Generic, Union

T = TypeVar('T')


class Flow(Generic[T]):
    """
    A flow is used to capture a sequence of steps, possibly skipping over any
    number of them and branching into disjunctions or conjunctions.

    The sequence is 'reversed' (that is, specified from end to beginning). This
    allows for a convenient tree-like notation, but it may trip you up.

    A shorthand notation allows lists to be interpreted as `JUMP` sequences.

    For example, the following flow of type `Flow[int]` describes a flow ending
    in 9, in which step 3 directly combines the outputs from steps 1 and 2.

        [9, STEP(3, AND(1, 2))]
    """

    def __init__(self, *items: FlowShorthand[T]):
        if len(items) < 2:
            raise RuntimeError("Expected at least two items.")
        self.items: list[Flow1[T]] = [Flow.shorthand(x) for x in items]

    def __iter__(self) -> Iterator[Flow1[T]]:
        return iter(self.items)

    @staticmethod
    def bags(value: Flow1[T]) -> list[set[T]]:
        """
        Determine the possible sets items occurring in the flow. Returns a
        list where the first item contains the set of items in all branches,
        and the remainder is the disjunction of sets of items occurring in at
        least one branch.
        """
        # TODO efficiency: for example, bags might pointlessly contain the same
        # items
        def _bags(v: Flow1[T]) -> list[set[T]]:
            if isinstance(v, Flow):
                if isinstance(v, OR):
                    return list(chain.from_iterable(
                        _bags(i) for i in v.items))
                else:
                    assert isinstance(v, (AND, STEP, JUMP))
                    return list(set(chain.from_iterable(p))
                        for p in product(*(_bags(i) for i in v.items)))
            else:
                return [{v}]
        everything = _bags(value)
        intersection = set.intersection(*everything)
        disjunctions = list(a for s in everything if (a := s - intersection))
        return [intersection] + disjunctions

    @staticmethod
    def shorthand(value: FlowShorthand[T]) -> Flow1[T]:
        """
        Translate shorthand data structures (list for `SKIP`) to real flows.
        """
        if isinstance(value, list):
            if len(value) == 1:
                return Flow.shorthand(value[0])
            else:
                return JUMP(*value)
        else:
            return value


class STEP(Flow[T]):
    """
    A series linked in a chain, where each node must *immediately* follow the
    one after.
    """
    pass


class JUMP(Flow[T]):
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
FlowShorthand = Union[Flow1[T], list[Flow1[T]]]
