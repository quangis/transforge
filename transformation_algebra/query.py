"""
This module intends to provide a way to express transformation trees (i.e.
chains of types and operations) in such a way that a query might use it to
check if a given transformation expression contains types and operations in the
specified order.
"""

from __future__ import annotations

import rdflib
from rdflib import Graph
from rdflib.paths import Path, ZeroOrMore, OneOrMore
from rdflib.term import Node
from rdflib.namespace import Namespace, NamespaceManager, RDFS, RDF
from abc import ABC, abstractmethod
from itertools import count, product
from collections import defaultdict
from transformation_algebra.type import Type, TypeOperation, \
    Function, TypeVariable
from transformation_algebra.expr import Operator
from transformation_algebra.graph import TA
from typing import TYPE_CHECKING, Protocol, Iterator, Any, Union, \
    overload, Optional, TypeVar

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
NestedFlow = Union[Element, Nested[Element]]


class QueryStatements(object):
    def __init__(self):
        self.entrances: list[tuple[bool, rdflib.Variable]] = []
        self.statements: list[str] = []
        self.exits: list[tuple[rdflib.Variable, bool]] = []


class TransformationQuery(object):  # TODO subclass rdflib.Query?
    def __init__(self, items: NestedFlow, namespace: Namespace):
        self.flow = TransformationFlow.shorthand(items)
        self.namespace = namespace
        self.prefix = "n"
        self.nsm = NamespaceManager(Graph())

        self.nsm.bind("ta", TA)
        self.nsm.bind(self.prefix, self.namespace)

        self.generator = iter(rdflib.Variable(f"n{i}") for i in count(start=1))

    def n3(self, *items: Path | Node) -> str:
        result = []
        for item in items:
            try:
                result.append(item.n3(self.nsm))
            except TypeError:
                result.append(item.n3())

        return " ".join(result) + "."
        # assert all(isinstance(item, (Path, Node)) for item in items)
        # return " ".join(item.n3(self.nsm) for item in items) + "."

    def sparql_type(self, variable: rdflib.Variable, type: Type,
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

            pred = RDF.type if index is None else RDF[f"_{index}"]
            if t.params:
                bnode = next(self.generator)
                yield self.n3(variable, pred, bnode)
                yield self.n3(bnode, RDFS.subClassOf,
                        self.namespace[t._operator.name])
                for i, param in enumerate(t.params, start=1):
                    yield from self.sparql_type(bnode, param, index=i)
            else:
                yield self.n3(
                    variable,
                    pred / (RDFS.subClassOf * ZeroOrMore),
                    self.namespace[t._operator.name])

    def trace(self, current: TransformationFlow) -> QueryStatements:
        """
        As a side effect, produce SPARQL constraints by tracing the paths
        between each unit. Return the variables assigned to the units.
        """

        result = QueryStatements()

        if isinstance(current, Unit):
            incoming = next(self.generator)
            outgoing = incoming

            # Deal with unit itself
            if current.type:
                result.statements.extend(
                    self.sparql_type(incoming, current.type))

            if current.operator:
                if current.type and not current.direct:
                    outgoing = next(self.generator)
                    result.statements.append(self.n3(
                        incoming,
                        ~TA.feeds * ZeroOrMore,
                        outgoing))

                result.statements.append(self.n3(
                    outgoing,
                    TA.via,
                    self.namespace[current.operator.name]))

            result.entrances.append((False, incoming))
            result.exits.append((outgoing, False))

        elif isinstance(current, AllOf):
            for item in current.items:
                t = self.trace(item)
                result.entrances.extend(t.entrances)
                result.statements.extend(t.statements)
                result.exits.extend(t.exits)

        elif isinstance(current, AnyOf):
            # result.entrances.append(?)
            result.statements.append(" UNION ".join(
                "{" + "\n".join(self.trace(item).statements) + "}"
                for item in current.items
            ))
            # result.exits.append(?)

        else:
            assert isinstance(current, Sequence)

            items = [self.trace(item) for item in current.items]

            for i in range(len(items)):
                result.statements.extend(items[i].statements)

                try:
                    lefts = items[i].exits
                    rights = items[i + 1].entrances
                except IndexError:
                    pass
                else:
                    for (l, lskip), (rskip, r) in product(lefts, rights):
                        skip = lskip or rskip or current.skips[i + 1]
                        result.statements.append(self.n3(
                            l,
                            ~TA.feeds * OneOrMore if skip else ~TA.feeds,
                            r
                        ))

            skip_entrance = current.skips[0]
            result.entrances = [(skip or skip_entrance, var)
                for skip, var in items[0].entrances]
            skip_exit = current.skips[-1]
            result.exits = [(var, skip or skip_exit)
                for skip, var in items[-1].entrances]

        return result

    def sparql(self) -> str:
        """
        Convert this Flow to a SPARQL query.
        """

        subquery = self.trace(self.flow)
        query = [
            "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>",
            "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>",
            "PREFIX ta: <https://github.com/quangis/transformation-algebra#>",
            f"PREFIX {self.prefix}: <{self.namespace}>",
            "SELECT ?workflow ?description WHERE {",
            "?workflow rdf:type ta:Transformation.",
            "OPTIONAL {?workflow rdfs:comment ?description}",
        ]
        query.extend((
            self.n3(
                rdflib.Variable("workflow"),
                TA.result / (~TA.feeds * ZeroOrMore) if skip else TA.result,
                var)
            for skip, var in subquery.entrances
        ))
        query.extend(subquery.statements)
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

    [C, ..., f, AllOf(A, B)]

    Note that the flow is 'reversed' (from output to input). This allows for a
    convenient tree-like notation, but it may trip you up.

    Furthermore, for succinct notation, lists are interpreted as `Sequence`
    transformation flows and the ellipsis indicates we may skip any number of
    steps.
    """
    # The same can also be described in terms of semantic linear time logic
    # formulae (SLTL), but since we will be using SPARQL to search through
    # workflows, the approach chosen here makes for a straightforward
    # translation.

    @staticmethod
    def shorthand(value: NestedFlow) -> TransformationFlow:
        """
        Translate shorthand data structures (ellipsis for skips, lists for
        sequences) to real flows.
        """
        assert value != ..., "ellipses may only occur in sequences"
        if isinstance(value, Operator):
            return Unit(operator=value)
        elif isinstance(value, Type):
            return Unit(type=value)
        elif isinstance(value, list):
            return Sequence(*value)
        elif isinstance(value, TransformationFlow):
            return value
        else:
            raise ValueError(
                f"{value} cannot be interpreted as a TransformationFlow")


class Unit(TransformationFlow):
    """
    A unit represents a single data instance in the flow.
    """

    def __init__(self,
            operator: Optional[Operator] = None,
            direct: bool = False,
            type: Optional[Type] = None):

        if operator and operator.definition:
            raise ValueError("any operation in a flow query must be primitive")

        self.operator = operator
        self.direct = direct
        self.type = type


class Sequence(TransformationFlow):
    """
    Indicate the order in which transformation elements must occur.
    """

    def __init__(self, *items: NestedFlow):
        assert items

        self.items: list[TransformationFlow] = []
        self.skips: list[bool] = []

        skip: bool = False
        previous: Type | Operator | None = None
        for current in items:
            if current == ...:
                skip = True
                continue

            if isinstance(previous, Type) and isinstance(current, Operator):
                result = self.items[-1]
                assert isinstance(result, Unit)
                assert not current.definition
                result.operator = current
                result.direct = not skip
            else:
                self.skips.append(skip)
                self.items.append(TransformationFlow.shorthand(current))

            if isinstance(current, (Operator, Type)):
                previous = current
            skip = False

        self.skips.append(skip)

        assert len(self.items) == len(self.skips) - 1


class AllOf(TransformationFlow):
    """
    Indicate which transformation paths must occur conjunctively. That is,
    every path must occur somewhere --- possibly on distinct, parallel
    branches, possibly on the same branch.
    """

    def __init__(self, *items: NestedFlow):
        assert items
        self.items = [TransformationFlow.shorthand(x) for x in items]


class AnyOf(TransformationFlow):
    """
    Indicate which transformation paths can occur disjunctively. That is, at
    least one path must occur somewhere.
    """

    def __init__(self, *items: NestedFlow):
        assert items
        self.items = [TransformationFlow.shorthand(x) for x in items]
