"""
This module intends to provide a way to express transformation trees (i.e.
chains of types and operations) in such a way that a query might use it to
check if a given transformation expression contains types and operations in the
specified order.
"""

from __future__ import annotations

from abc import ABC
from itertools import count, chain
from transformation_algebra.type import Type, TypeInstance, TypeOperation, \
    Function, TypeVar
from transformation_algebra.expr import Operation
from transformation_algebra.rdf import AlgebraNamespace, TA
import typing
from typing import TYPE_CHECKING, Protocol, Iterator, Any, Union, \
    overload, Optional
# from rdflib.plugins import sparql
from rdflib.namespace import RDF, RDFS

# We use '...' to indicate that steps may be skipped. This workaround allows us
# to refer to the ellipsis' type. See github.com/python/typing/issues/684
if TYPE_CHECKING:
    from builtins import ellipsis
else:
    ellipsis = type(Ellipsis)

# For convenience, we allow nested sequences in `Flow.serial()`. The following
# temporary solution to the recursive type thus introduced has been lifted from
# github.com/python/mypy/issues/731
_T_co = typing.TypeVar("_T_co")


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
NestedNotation = Union[Element, Nested[Element]]


class TransformationQuery(object):  # TODO subclass rdflib.Query?
    def __init__(self, *items: NestedNotation, namespace: AlgebraNamespace):
        self.flow = Serial(*items)
        self.namespace = namespace

    def sparql_type(self, name: str, type: Type,
            name_generator: Iterator[str],
            index: Optional[int] = None) -> Iterator[str]:
        """
        Produce SPARQL constraints for the given (non-function) type.
        """

        t = type.instance()

        if isinstance(t, TypeVar):
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

    def trace(self,
            f: TransformationFlow,
            current: str,
            previous: Optional[str] = None,
            name_generator: Optional[Iterator[str]] = None) -> \
            Iterator[str]:
        """
        Trace the paths between each node in a chain to produce SPARQL
        constraints.
        """
        # See rdflib.paths: (~TA.feeds * OneOrMore).n3(g.namespace_manager)

        name_generator = name_generator or iter(f"n{i}" for i in count())

        if isinstance(f, Unit):
            if previous:
                yield (
                    f"?{previous} "
                    f"({'^ta:feeds' + ('+' if f.skip else '')}) "
                    f"?{current}.")

            if f.via:
                assert f.via.is_primitive(), \
                    "operation in a flow query must be primitive"
                yield f"?{current} ta:via <{self.namespace[f.via]}>."
            if f.type:
                yield from self.sparql_type(current, f.type, name_generator)

        elif isinstance(f, Parallel):
            for sub in f.items:
                yield from self.trace(sub, next(name_generator), previous,
                    name_generator)

        else:
            assert isinstance(f, Serial)

            # TODO remove this assumption when possible
            assert all(isinstance(x, Unit) for x in f.items[:-1])

            for n, x in zip(chain([current], name_generator), f.items):
                yield from self.trace(x, n, previous, name_generator)
                previous = n

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
            "?workflow ta:result ?output_node.",
        ]
        query.extend(self.trace(self.flow, "output_node"))
        query.append("} GROUP BY ?workflow")
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
        # Does the operation immediately output the type? (i.e. is `...`
        # specified before the type and after the operation?)
        self.immediate = False
        self.type = type
        # Does a skip occur after this unit? (i.e. is `...` specified *before*
        # this unit in the reversed sequence)
        self.skip = False
        super().__init__()

    def set_skip(self) -> None:
        self.skip = True


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
                if skip:
                    item.set_skip()
                self.items.append(item)
            skip = False
            if isinstance(current, (Type, Operation)):
                then = current

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
