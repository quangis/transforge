"""
This module intends to provide a way to express transformation trees (i.e.
chains of types and operations) in such a way that a query might use it to
check if a given transformation expression contains types and operations in the
specified order.
"""

from __future__ import annotations

from abc import ABC
from transformation_algebra import Operation, Type
from typing import TYPE_CHECKING, Protocol, TypeVar, Iterator, Any, Union, \
    overload

# We use '...' to indicate that steps may be skipped. This workaround allows us
# to refer to the ellipsis' type. See github.com/python/typing/issues/684
if TYPE_CHECKING:
    from builtins import ellipsis
else:
    ellipsis = type(Ellipsis)

# For convenience, we allow nested sequences in `Flow.serial()`. The following
# temporary solution to the recursive type thus introduced has been lifted from
# github.com/python/mypy/issues/731
_T_co = TypeVar("_T_co")


class _RecursiveSequence(Protocol[_T_co]):
    def __len__(self) -> int:
        ...

    @overload
    def __getitem__(self, __index: int) -> _T_co | _RecursiveSequence[_T_co]:
        ...

    @overload
    def __getitem__(self, __index: slice) -> _RecursiveSequence[_T_co]:
        ...

    def __contains__(self, __x: object) -> bool:
        ...

    def __iter__(self) -> Iterator[_T_co | _RecursiveSequence[_T_co]]:
        ...

    def __reversed__(self) -> Iterator[_T_co | _RecursiveSequence[_T_co]]:
        ...

    def count(self, __value: Any) -> int:
        ...

    def index(self, __value: Any, __start: int = ...,
            __stop: int = ...) -> int:
        ...


Unit = Union[Type, Operation, ellipsis]


class Flow(ABC):
    """
    A flow captures some relevant aspects of a conceptual process, in terms of
    the sequence of elements that must occur in it. For example, the following
    flow holds that there must be datatypes A and B that are fed to an
    operation f that eventually results in a datatype C:

    Flow.serial(C, ..., f, [A, B])
    """
    # The same can also be described in terms of semantic linear time logic
    # formulae (SLTL), but since we will be using SPARQL to search through
    # workflows, the approach chosen here makes for a straightforward
    # translation.

    @staticmethod
    def serial(*steps: Unit | _RecursiveSequence[Unit]) -> Serial:
        """
        A convenience function for more readable queries. Tuples indicate
        sequences, lists indicate branches, the ellipsis indicates we may skip
        any number of steps.
        """
        args: list[Type | Operation | None | Parallel] = []
        for step in steps:
            if step == ...:
                args.append(None)
            elif isinstance(step, (Type, Operation)):
                args.append(step)
            elif isinstance(step, tuple):
                args.extend(Flow.serial(*step).sequence)
            elif isinstance(step, list):
                args.append(Flow.parallel(*step))
            else:
                raise ValueError

        return Serial(*args)

    @staticmethod
    def parallel(*steps: Unit | _RecursiveSequence[Unit]) -> Parallel:
        """
        Counterpart to `serial`.
        """
        args: list[Type | Operation | Serial] = []
        for step in steps:
            if isinstance(step, (Type, Operation)):
                args.append(step)
            elif isinstance(step, list):
                args.extend(Flow.parallel(*step).branches)
            elif isinstance(step, tuple):
                args.append(Flow.serial(*step))
            else:
                raise ValueError

        return Parallel(*args)


class Serial(Flow):
    """
    Describes which transformation elements must occur, in what order.
    """

    def __init__(self, *sequence: Type | Operation | None | Parallel):
        self.sequence = list(sequence)


class Parallel(Flow):
    """
    Describes which transformation paths must occur conjunctively.
    """

    def __init__(self, *branches: Type | Operation | Serial):
        self.branches = list(branches)
