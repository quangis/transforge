"""
This module intends to provide a way to express transformation trees (i.e.
chains of types and operations) in such a way that a query might use it to
check if a given transformation expression contains types and operations in the
specified order.
"""

from __future__ import annotations

from abc import ABC
from transformation_algebra.type import Type, TypeInstance
from transformation_algebra.expr import Operation
from typing import TYPE_CHECKING, Protocol, TypeVar, Iterator, Any, Union, \
    overload, Optional

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


class Nested(Protocol[_T_co]):
    def __len__(self) -> int:
        ...

    @overload
    def __getitem__(self, __index: int) -> _T_co | Nested[_T_co]:
        ...

    @overload
    def __getitem__(self, __index: slice) -> Nested[_T_co]:
        ...

    def __contains__(self, __x: object) -> bool:
        ...

    def __iter__(self) -> Iterator[_T_co | Nested[_T_co]]:
        ...

    def __reversed__(self) -> Iterator[_T_co | Nested[_T_co]]:
        ...

    def count(self, __value: Any) -> int:
        ...

    def index(self, __value: Any, __start: int = ...,
            __stop: int = ...) -> int:
        ...


Element = Union[Type, Operation, ellipsis, 'TransformationFlow']


class TransformationFlow(ABC):
    """
    A flow captures some relevant aspects of a conceptual process, in terms of
    the sequence of elements that must occur in it. For example, the following
    flow holds that there must be datatypes `A` and `B` that are fed to an
    operation f that eventually results in a datatype `C`:

    Serial(C, ..., f, [A, B])

    Note that the flow is 'reversed' (from output to input). This allows for a
    convenient tree-like notation, but it may trip you up.

    Furthermore, for succinct notation, nested tuples are interpreted as
    `Serial` and lists as `Parallel` transformation flows. The ellipsis
    indicates we may skip any number of steps.
    """
    # The same can also be described in terms of semantic linear time logic
    # formulae (SLTL), but since we will be using SPARQL to search through
    # workflows, the approach chosen here makes for a straightforward
    # translation.

    def __init__(self, *items: Element | Nested[Element]):
        # Does a skip occur after? (e.g. *before* in the reversed sequence)
        self.skip = False
        self.items: list[TransformationFlow] = [
            TransformationFlow.shorthand(x) for x in items]

    @staticmethod
    def shorthand(value: Element | Nested[Element]) \
            -> TransformationFlow:
        """
        Translate shorthand data structures (ellipsis for skips, tuples for
        serials, lists for parallels) to real flows.
        """
        assert value != ..., "ellipses may only occur in serials"
        if isinstance(value, tuple):
            return Serial(*value)
        elif isinstance(value, list):
            return Parallel(*value)
        elif isinstance(value, Type):
            return Unit(type=value)
        elif isinstance(value, Operation):
            return Unit(operation=value)
        elif isinstance(value, TransformationFlow):
            return value
        else:
            raise ValueError(
                f"{value} cannot be interpreted as a TransformationFlow")


class Unit(TransformationFlow):
    """
    A unit represents a single data instance in the flow.
    """

    def __init__(self, type: Optional[Type] = None,
            operation: Optional[Operation] = None):
        self.via = operation
        self.immediate = False
        self.type = type
        super().__init__()


class Serial(TransformationFlow):
    """
    Indicate the order in which transformation elements must occur.
    """

    def __init__(self, *items: Element | Nested[Element]):
        super().__init__()

        skip: bool = False
        then: Type | Operation | None = None  # 'previous', in reversed flow
        for current in items:
            if current == ...:
                skip = True
                continue
            if isinstance(current, Operation) and isinstance(then, Type):
                assert isinstance(self.items[-1], Unit)
                self.items[-1].via = current
                self.items[-1].immediate = skip
            else:
                item = TransformationFlow.shorthand(current)
                item.skip = skip
                self.items.append(item)
            skip = False
            if isinstance(current, (Type, Operation)):
                then = current


class Parallel(TransformationFlow):
    """
    Indicate which transformation paths must occur conjunctively.
    """
    pass


class Choice(TransformationFlow):
    """
    Indicate which transformation paths can occur disjunctively.
    """
    pass
