"""
This module intends to provide a way to express transformation trees (i.e.
chains of types and operations) in such a way that a query might use it to
check if a given transformation expression contains types and operations in the
specified order.
"""

from __future__ import annotations

from abc import ABC
from itertools import count, chain
from transformation_algebra.type import Type, TypeOperation, \
    Function, TypeVariable
from transformation_algebra.expr import Operator
from transformation_algebra.graph import TransformationNamespace, TA
from typing import TYPE_CHECKING, Iterator, Any, Union, \
    overload, Optional, TypeVar
# from rdflib.plugins import sparql
from rdflib.namespace import RDF, RDFS

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol


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


Element = Union[Type, Operator, ellipsis, 'TransformationFlow']
NestedNotation = Union[Element, Nested[Element]]


class TransformationQuery(object):  # TODO subclass rdflib.Query?
    def __init__(self, *items: NestedNotation,
            namespace: TransformationNamespace):
        self.flow = Serial(*items)
        self.namespace = namespace

    def sparql_type(self, name: str, type: Type,
            name_generator: Iterator[str],
            index: Optional[int] = None) -> Iterator[str]:
        """
        Produce SPARQL constraints for the given (non-function) type.
        """

        t = type.instance()

        if isinstance(t, TypeVariable):
            # If a type in a trace query contains variables, it must be a
            # wildcard --- because we don't do anything with it
            assert t.wildcard
        else:
            assert isinstance(t, TypeOperation) and t.operator != Function

            pred = "rdf:type" if index is None else f"rdf:_{index}"
            if t.params:
                bnode = next(name_generator)
                yield f"?{name} {pred} ?{bnode}."
                yield f"?{bnode} rdfs:subClassOf <{self.namespace[t._operator]}>."
                for i, param in enumerate(t.params, start=1):
                    yield from self.sparql_type(bnode, param, name_generator,
                        index=i)
            else:
                yield f"?{name} {pred}/(rdfs:subClassOf*) <{self.namespace[t._operator]}>."

    def trace(self, flow: TransformationFlow,
            previous: Optional[tuple[str, TransformationFlow]] = None,
            name_generator: Optional[Iterator[str]] = None,
            current: str = "",
            ) -> Iterator[str]:
        """
        Trace the paths between each node in a chain to produce SPARQL
        constraints.
        """
        # See rdflib.paths: (~TA.feeds * OneOrMore).n3(g.namespace_manager)

        name_generator = name_generator or iter(f"n{i}" for i in count(start=1))
        current = current or next(name_generator)

        if isinstance(flow, Unit):
            if previous:
                assert isinstance(previous[1], Unit)
                if flow.skip:
                    if previous[1].type and flow.operation:
                        modifier = '*'
                    else:
                        modifier = '+'
                else:
                    modifier = ''

                yield (
                    f"?{previous[0]} "
                    f"(^ta:feeds{modifier}) "
                    f"?{current}.")

            if flow.type:
                yield from self.sparql_type(current, flow.type, name_generator)
            else:
                assert flow.operation
                yield f"?{current} ta:via <{self.namespace[flow.operation]}>."

        elif isinstance(flow, Parallel):
            for sub in flow.items:
                yield from self.trace(sub, previous, name_generator)

        else:
            assert isinstance(flow, Serial)

            # TODO remove this assumption when possible
            assert all(isinstance(x, Unit) for x in flow.items[:-1])

            for name, sub in zip(chain([current], name_generator), flow.items):
                yield from self.trace(sub, previous, name_generator, name)
                previous = name, sub

    def sparql(self) -> str:
        """
        Convert this Flow to a SPARQL query.
        """

        query = [
            "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>",
            "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>",
            "PREFIX ta: <https://github.com/quangis/transformation-algebra#>",
            "SELECT ?workflow ?description WHERE {",
            "?workflow rdf:type ta:Transformation.",
            "?workflow rdfs:comment ?description.",
            "?workflow ta:result ?n0.",
        ]
        query.extend(self.trace(self.flow, current="n0"))
        query.append("} GROUP BY ?workflow ?description")
        return "\n".join(query)

        # TODO return sparql.Query instead, to avoid parsing at all. This does
        # not avoid parsing:
        # return sparql.prepareQuery("\n".join(query),
        #         initNs={'ta': TA, 'rdf': RDF, 'rdfs': RDFS}
        # )


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

    def __init__(self, *items: Element | Nested[Element], skip: bool = False):
        self.items: list[TransformationFlow] = [
            TransformationFlow.shorthand(x) for x in items]

    def set_skip(self) -> None:
        for x in self.items:
            x.set_skip()

    @staticmethod
    def shorthand(value: Element | Nested[Element]) \
            -> TransformationFlow:
        """
        Translate shorthand data structures (ellipsis for skips, tuples for
        serials, lists for parallels) to real flows.
        """
        assert value != ..., "ellipses may only occur in serials"
        if isinstance(value, (Type, Operator)):
            return Unit(value)
        elif isinstance(value, tuple):
            return Serial(*value)
        elif isinstance(value, list):
            return Parallel(*value)
        elif isinstance(value, TransformationFlow):
            return value
        else:
            raise ValueError(
                f"{value} cannot be interpreted as a TransformationFlow")


class Unit(TransformationFlow):
    """
    A unit represents a single data instance in the flow.
    """

    def __init__(self, value: Type | Operator):
        if isinstance(value, Operator) and not value.is_primitive():
            raise ValueError("any operation in a flow query must be primitive")
        self.value = value

        # Does a skip occur after this unit? (i.e. is `...` specified *before*
        # this unit in the reversed sequence)
        self.skip = False

        super().__init__()

    @property
    def type(self) -> Optional[Type]:
        return self.value if isinstance(self.value, Type) else None

    @property
    def operation(self) -> Optional[Operator]:
        return self.value if isinstance(self.value, Operator) else None

    def set_skip(self) -> None:
        self.skip = True


class Serial(TransformationFlow):
    """
    Indicate the order in which transformation elements must occur.
    """

    def __init__(self, *items: Element | Nested[Element]):
        super().__init__()

        skip: bool = False
        for current in items:
            if current == ...:
                skip = True
            else:
                item = TransformationFlow.shorthand(current)
                if skip:
                    item.set_skip()
                self.items.append(item)
                skip = False

    def set_skip(self) -> None:
        self.items[0].set_skip()


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
