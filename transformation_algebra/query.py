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
from abc import ABC
from itertools import count, product, chain
from transformation_algebra.type import Type, TypeOperation, \
    Function, TypeVariable
from transformation_algebra.expr import Operator
from transformation_algebra.graph import TA
from typing import TYPE_CHECKING, Iterator, Union, Optional

# We use '...' to indicate that steps may be skipped. This workaround allows us
# to refer to the ellipsis' type. See github.com/python/typing/issues/684
if TYPE_CHECKING:
    from builtins import ellipsis
else:
    ellipsis = type(Ellipsis)

Element = Union[Type, Operator, ellipsis, 'Flow']
NestedFlow = Union[Element, list[Element]]

Triple = tuple[Node, Union[Path, Node], Node]


generator = iter(rdflib.Variable(f"n{i}") for i in count(start=1))


def n3(*items: Path | Node) -> str:
    result = []
    for item in items:
        try:
            result.append(item.n3())
        except TypeError:
            result.append(str(item))

    return " ".join(result) + "."


class Query(object):

    def strings(self) -> Iterator[str]:
        if isinstance(self, QueryUNIT):
            yield from (n3(*s) for s in chain(*(self.prologue, self.statements,
                self.epilogue)))
        elif isinstance(self, (QuerySEQ, QueryOR, QueryAND)):
            yield from chain(*(item.strings() for item in self.items))

    def entrances(self) -> Iterator[tuple[bool, QueryUNIT]]:
        if isinstance(self, QueryUNIT):
            yield (False, self)
        elif isinstance(self, QuerySEQ):
            yield from ((skip or self.flow.skips[0], unit)
                for skip, unit in self.items[0].entrances())
        elif isinstance(self, (QueryOR, QueryAND)):
            yield from chain(*(item.entrances() for item in self.items))

    def exits(self) -> Iterator[tuple[QueryUNIT, bool]]:
        if isinstance(self, QueryUNIT):
            yield (self, False)
        elif isinstance(self, QuerySEQ):
            yield from ((unit, skip or self.flow.skips[-1])
                for unit, skip in self.items[-1].exits())
        elif isinstance(self, (QueryOR, QueryAND)):
            yield from chain(*(item.exits() for item in self.items))

    @staticmethod
    def create(flow: Flow, ns: Namespace) -> Query:
        if isinstance(flow, Unit):
            return QueryUNIT(flow, ns)
        elif isinstance(flow, Sequence):
            return QuerySEQ(flow, ns)
        elif isinstance(flow, OR):
            return QueryOR(flow, ns)
        else:
            assert isinstance(flow, AND)
            return QueryAND(flow, ns)


class QueryUNIT(Query):
    def __init__(self, unit: Unit, ns: Namespace):
        self.unit = unit
        self.namespace = ns
        self.incoming = next(generator)
        self.outgoing = self.incoming

        self.prologue: list[Triple] = []
        self.statements: list[Triple] = []
        self.epilogue: list[Triple] = []

        if unit.type:
            self.type(self.incoming, unit.type)

        if unit.operator:
            if unit.type and not unit.direct:
                self.outgoing = next(generator)
                self.add(self.incoming, ~TA.feeds * ZeroOrMore, self.outgoing)

            self.add(self.outgoing, TA.via, self.namespace[unit.operator.name])

    def add(self, *other: Node | Path, at_start: bool = False) -> None:
        if at_start:
            self.statements.insert(0, other)  # type: ignore
        else:
            self.statements.append(other)  # type: ignore

    def type(self, variable: rdflib.Variable, type: Type,
            index: Optional[int] = None) -> None:
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
                bnode = next(generator)
                self.add(variable, pred, bnode)
                self.add(bnode, RDFS.subClassOf, self.namespace[t._operator.name])
                for i, param in enumerate(t.params, start=1):
                    self.type(bnode, param, index=i)
            else:
                self.add(
                    variable,
                    pred / (RDFS.subClassOf * ZeroOrMore),  # type: ignore
                    self.namespace[t._operator.name])

    def connect_left(self, *units: tuple[QueryUNIT, bool], skip=False):
        for unit, skipL in units:
            self.prologue.append((
                unit.outgoing,
                ~TA.feeds * OneOrMore if skipL or skip else ~TA.feeds,
                self.incoming
            ))

    def connect_right(self, *units: tuple[bool, QueryUNIT], skip=False):
        for skipR, unit in units:
            self.epilogue.append((
                self.outgoing,
                ~TA.feeds * OneOrMore if skip or skipR else ~TA.feeds,
                unit.incoming
            ))

    def connect_unit(self, unit: QueryUNIT, skip=False):
        unit.prologue.append((
            self.outgoing,
            ~TA.feeds * OneOrMore if skip else ~TA.feeds,
            unit.incoming))


class QuerySEQ(Query):
    def __init__(self, flow: Sequence, ns: Namespace):
        self.flow = flow
        self.items: list[Query] = [Query.create(o, ns) for o in flow.items]

        # Add connections
        for i in range(len(self.items)):
            try:
                lefts = self.items[i].exits()
                rights = self.items[i + 1].entrances()
            except IndexError:
                pass
            else:
                for left, lskip in lefts:
                    for rskip, right in rights:
                        left.connect_unit(right,
                            skip=lskip or rskip or flow.skips[i + 1])


class QueryOR(Query):
    def __init__(self, flow: OR, ns: Namespace):
        self.flow = flow
        self.items: list[Query] = [Query.create(o, ns) for o in flow.items]


class QueryAND(Query):
    def __init__(self, flow: AND, ns: Namespace):
        self.flow = flow
        self.items: list[Query] = [Query.create(o, ns) for o in flow.items]


class TransformationQuery(object):  # TODO subclass rdflib.Query?
    def __init__(self, items: NestedFlow, namespace: Namespace):
        self.flow = Flow.shorthand(items)
        self.namespace = namespace
        self.prefix = "n"
        self.nsm = NamespaceManager(Graph())

        self.nsm.bind("ta", TA)
        self.nsm.bind(self.prefix, self.namespace)

        self.generator = iter(rdflib.Variable(f"n{i}") for i in count(start=1))

    def sparql(self) -> str:
        """
        Convert this Flow to a SPARQL query.
        """

        subquery = Query.create(self.flow, self.namespace)
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
            n3(
                rdflib.Variable("workflow"),
                TA.result / (~TA.feeds * ZeroOrMore) if skip else TA.result,
                unit.incoming)
            for skip, unit in subquery.entrances()
        ))
        query.extend(subquery.strings())
        query.append("} GROUP BY ?workflow ?description")
        return "\n".join(query)

        # TODO return sparql.Query instead, to avoid parsing at all. This does
        # not avoid parsing:
        # return sparql.prepareQuery("\n".join(query),
        #         initNs={'ta': TA, 'rdf': RDF, 'rdfs': RDFS}
        # )


class Flow(ABC):
    """
    A flow captures some relevant aspects of a conceptual process, in terms of
    the sequence of elements that must occur in it. For example, the following
    flow holds that there must be datatypes `A` and `B` that are fed to an
    operation f that eventually results in a datatype `C`:

    [C, ..., f, AND(A, B)]

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
    def shorthand(value: NestedFlow) -> Flow:
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
        elif isinstance(value, Flow):
            return value
        else:
            raise ValueError(
                f"{value} cannot be interpreted as a Flow")


class Unit(Flow):
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


class Sequence(Flow):
    """
    Indicate the order in which transformation elements must occur.
    """

    def __init__(self, *items: NestedFlow):
        assert items

        self.items: list[Flow] = []
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
                self.items.append(Flow.shorthand(current))

            if isinstance(current, (Operator, Type)):
                previous = current
            skip = False

        self.skips.append(skip)

        assert len(self.items) == len(self.skips) - 1


class Branch(Flow):
    def __init__(self, *items: NestedFlow):
        assert items
        self.items = [Flow.shorthand(x) for x in items]


class AND(Branch):
    """
    Indicate which transformation paths must occur conjunctively. That is,
    every path must occur somewhere --- possibly on distinct, parallel
    branches, possibly on the same branch.
    """
    pass


class OR(Branch):
    """
    Indicate which transformation paths can occur disjunctively. That is, at
    least one path must occur somewhere.
    """
    pass
