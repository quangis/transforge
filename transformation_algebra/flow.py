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


Unit = Union[Type, Operation, ellipsis, 'TransformationFlow']


class TransformationFlow(ABC):
    """
    A flow captures some relevant aspects of a conceptual process, in terms of
    the sequence of elements that must occur in it. For example, the following
    flow holds that there must be datatypes A and B that are fed to an
    operation f that eventually results in a datatype C:

    Serial(C, ..., f, [A, B])

    Note that, for succinct notation, nested tuples are interpreted as `Serial`
    and lists as `Parallel` transformation flows. The ellipsis indicates we may
    skip any number of steps.

    """
    # The same can also be described in terms of semantic linear time logic
    # formulae (SLTL), but since we will be using SPARQL to search through
    # workflows, the approach chosen here makes for a straightforward
    # translation.

    def __init__(self, *items: Unit | Nested[Unit]):
        self.items: list[Type | Operation | TransformationFlow] = [
            TransformationFlow.shorthand(x) for x in items]

    @staticmethod
    def shorthand(value: Unit | Nested[Unit]) \
            -> Union[Type, Operation, TransformationFlow]:
        """
        Translate shorthand data structures (ellipsis for skips, tuples for
        serials, lists for parallels) to real flows.
        """
        if value == ...:
            return Skip()
        elif isinstance(value, tuple):
            return Serial(*value)
        elif isinstance(value, list):
            return Parallel(*value)
        elif isinstance(value, (Type, Operation, TransformationFlow)):
            return value
        else:
            raise ValueError(
                f"{value} cannot be interpreted as a TransformationFlow")


class Skip(TransformationFlow):
    """
    Indicate that any number of steps may be skipped over.
    """

    def __init__(self):
        pass


class Serial(TransformationFlow):
    """
    Indicate the order in which transformation elements must occur.
    """
    pass


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
