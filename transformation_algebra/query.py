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
from itertools import count, chain
from typing import Iterator, Union, Optional

from transformation_algebra.flow import Flow, Sequence, AND, OR, ellipsis
from transformation_algebra.type import Type, TypeOperation, \
    Function, TypeVariable
from transformation_algebra.expr import Operator
from transformation_algebra.graph import TA


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
            yield from ((skip or self.skips[0], unit)
                for skip, unit in self.items[0].entrances())
        elif isinstance(self, (QueryOR, QueryAND)):
            yield from chain(*(item.entrances() for item in self.items))

    def exits(self) -> Iterator[tuple[QueryUNIT, bool]]:
        if isinstance(self, QueryUNIT):
            yield (self, False)
        elif isinstance(self, QuerySEQ):
            yield from ((unit, skip or self.skips[-1])
                for unit, skip in self.items[-1].exits())
        elif isinstance(self, (QueryOR, QueryAND)):
            yield from chain(*(item.exits() for item in self.items))

    @staticmethod
    def create(flow: Flow | Type | Operator, ns: Namespace) -> Query:
        if isinstance(flow, Type):
            q = QueryUNIT(ns)
            q.typed = flow
            return q
        elif isinstance(flow, Operator):
            q = QueryUNIT(ns)
            q.operator = flow
            return q
        elif isinstance(flow, Sequence):
            return QuerySEQ(flow, ns)
        elif isinstance(flow, OR):
            return QueryOR(flow, ns)
        else:
            assert isinstance(flow, AND)
            return QueryAND(flow, ns)


class QueryUNIT(Query):
    def __init__(self, ns: Namespace):
        self.operator: Operator | None = None
        self.direct: bool = False
        self.typed: Type | None = None
        self.namespace = ns
        self.incoming = next(generator)
        self.outgoing = self.incoming

        self.prologue: list[Triple] = []
        self.statements: list[Triple] = []
        self.epilogue: list[Triple] = []

    def init(self):
        if self.typed:
            self.type(self.incoming, self.typed)

        if self.operator:
            if self.type and not self.direct:
                self.outgoing = next(generator)
                self.add(self.incoming, ~TA.feeds * ZeroOrMore, self.outgoing)

            self.add(self.outgoing, TA.via, self.namespace[self.operator.name])

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
        self.items: list[Query] = []
        self.skips: list[bool] = []

        previous: Type | Operator | None = None
        for lskip, current, rskip in flow:
            if isinstance(current, Operator) and isinstance(previous, Type):
                result = self.items[-1]
                assert isinstance(result, QueryUNIT) and not result.operator
                assert not current.definition
                result.operator = current
                result.direct = not lskip
            else:
                self.items.append(Query.create(current, ns))
                self.skips.append(lskip)
            if isinstance(current, (Operator, Type)):
                previous = current
        self.skips.append(flow.skips[-1])

        # Add connections
        for i in range(len(self.items)):
            item = self.items[i]
            if isinstance(item, QueryUNIT):
                item.init()
            try:
                lefts = self.items[i].exits()
                rights = self.items[i + 1].entrances()
            except IndexError:
                pass
            else:
                for left, lskip in lefts:
                    for rskip, right in rights:
                        left.connect_unit(right,
                            skip=lskip or rskip or self.skips[i + 1])


class QueryOR(Query):
    def __init__(self, flow: OR, ns: Namespace):
        self.flow = flow
        self.items: list[Query] = [Query.create(o, ns) for o in flow.items]


class QueryAND(Query):
    def __init__(self, flow: AND, ns: Namespace):
        self.flow = flow
        self.items: list[Query] = [Query.create(o, ns) for o in flow.items]


class TransformationQuery(object):  # TODO subclass rdflib.Query?
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

    def __init__(self, items: Flow[Type | Operator], namespace: Namespace):
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

