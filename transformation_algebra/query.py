"""
This module intends to provide a way to express transformation trees (i.e.
chains of types and operations) in such a way that a query might use it to
check if a given transformation expression contains types and operations in the
specified order.
"""

from __future__ import annotations

import rdflib
from rdflib.paths import Path, ZeroOrMore, OneOrMore
from rdflib.term import Node, Variable
from rdflib.namespace import RDFS, RDF
from itertools import count, chain, product, starmap
from typing import Iterator, Iterable, Union, TypeVar
from collections import defaultdict

from transformation_algebra.flow import Flow, Flow1, SKIP, LINK, AND, OR, \
    FlowShorthand
from transformation_algebra.type import Type, TypeOperation, \
    Function, TypeVariable
from transformation_algebra.expr import Operator
from transformation_algebra.lang import Language
from transformation_algebra.graph import TA

Operators = Flow[Union[Type, Operator]]  # really: OR[Operator]
Triple = tuple[Node, Union[Path, Node], Node]


A = TypeVar('A')


def flatten(xs: Iterable[Iterable[A]]) -> list[A]:
    return list(chain.from_iterable(xs))


def union(*xs: str) -> Iterator[str]:
    if len(xs) > 1:
        yield "{{{}}}".format("\n} UNION {\n".join(xs))
    else:
        yield from xs


class Query(object):  # TODO subclass rdflib.Query?
    """
    A flow captures some relevant aspects of a conceptual process, in terms of
    the sequence of elements that must occur in it. For example, the following
    flow holds that there must be datatypes `A` and `B` that are fed to an
    operation f that eventually results in a datatype `C`:

    [C, f, AND(A, B)]

    Note that the flow is 'reversed' (from output to input). This allows for a
    convenient tree-like notation, but it may trip you up.
    """

    def __init__(self, lang: Language, flow: FlowShorthand[Type | Operator],
            by_output: bool = False,
            by_types: bool = False,
            # by_operators: bool = True,
            by_order: bool = True):

        self.language = lang
        self.flow: Flow1[Type | Operator] = Flow.shorthand(flow)

        # Connect each node to a disjunction of conjunction of nodes
        self.conns: dict[Variable, list[list[Variable]]] = defaultdict(list)
        self.skips: dict[Variable, bool] = defaultdict(bool)
        self.attr_via: dict[Variable, list[Operator]] = defaultdict(list)
        self.attr_type: dict[Variable, list[Type]] = defaultdict(list)

        self.cache: dict[Type, Variable] = dict()
        self.generator = iter(Variable(f"_{i}") for i in count())

        # Filter by...
        self.by_output = by_output
        self.by_types = by_types
        # self.by_operators = by_operators
        self.by_order = by_order

        self.workflow = rdflib.Variable("workflow")

        entrances, _ = self.add_flow(self.flow)
        assert len(entrances) == 1 and len(entrances[0]) == 1
        self.output = entrances[0][0]

        # if self.by_output:
        #     for item in Flow.leaves(self.flow, targets=True):
        #         if isinstance(item, Type):
        #             self.set_type(self.output, item)

        # for item in Flow.leaves(self.flow):
        #     # if self.by_operators and isinstance(item, Operator):
        #     #     self.set_operator(self.root, item, TA.member)

        #     if self.by_types and isinstance(item, Type):
        #         self.set_type(self.root, item, TA.member)

        # if self.by_order:
        #     self.connect(self.output, "start", False, self.flow)

    def sparql(self) -> str:
        """
        Convert the flow to a full SPARQL query.
        """
        return "\n".join(chain([
            "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>",
            "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>",
            "PREFIX ta: <https://github.com/quangis/transformation-algebra#>",
            "SELECT ?workflow WHERE {"],
            self.chronology(self.output),
            ["} GROUP BY ?workflow"]

        ))

    def chronology(self, target: Variable,
            entrance: Variable | None = None) -> Iterator[str]:

        # Connecting to target node
        if entrance:
            if self.attr_type[entrance] and not self.attr_type[target]:
                if self.skips[entrance]:
                    yield self.triple(entrance, ~TA.feeds * ZeroOrMore, target)
                else:
                    yield f"BIND({entrance.n3()} AS {target.n3()})"
            else:
                if self.skips[entrance]:
                    yield self.triple(entrance, ~TA.feeds * OneOrMore, target)
                else:
                    yield self.triple(entrance, ~TA.feeds, target)
        else:
            # assert entrance == self.output, f"{entrance} != {self.output}"
            assert target == self.output
            yield self.triple(self.workflow, TA.output, self.output)

        # Node's own attributes
        yield from union(*("\n".join(self.set_type(target, t)) for t in
            self.attr_type[target]))
        yield from union(*("\n".join(self.set_operator(target, o)) for o in
            self.attr_via[target]))

        # Connecting to rest of the tree
        connections = self.conns[target]
        if len(connections) > 1:
            yield "{"
            yield "\n} UNION {\n".join(
                "\n".join(chain.from_iterable(
                    self.chronology(conj, target) for conj in disj
                ))
                for disj in connections
            )
            yield "}"
        elif connections:
            for conj in connections[0]:
                yield from self.chronology(conj, target)

    def triple(self, *items: Node | Path) -> str:
        result = []
        for item in items:
            try:
                result.append(item.n3())
            except TypeError:
                result.append(str(item))
        return " ".join(result) + "."

    def set_operator(self, variable: rdflib.Variable, op: Operator,
            predicate: Node | Path = TA.via) -> Iterator[str]:

        if op.definition:
            from warnings import warn
            warn(f"query used a non-primitive operation {op.name}")
        yield self.triple(variable, predicate, self.language.namespace[op])

    def set_type(self, variable: rdflib.Variable, type: Type,
            predicate: Node | Path = RDF.type) -> Iterator[str]:
        """
        Produce SPARQL constraints for the given (non-function) type.
        """

        t = type.instance().normalize()

        taxonomy = self.language.taxonomy

        if t not in taxonomy:
            from warnings import warn
            warn(f"type {t} in query is non-canonical")

        if isinstance(t, TypeVariable):
            # If a type in a trace query contains variables, it must be a
            # wildcard --- because we don't do anything with it
            assert t.wildcard
        elif t in taxonomy:
            yield self.triple(variable,
                predicate / RDFS.subClassOf,  # type: ignore
                self.language.namespace[t])
        else:
            assert isinstance(t, TypeOperation) and t.operator != Function
            if t.params:  # and t not in self.cache:
                self.cache[t] = bnode = next(self.generator)
                yield self.triple(variable, predicate, bnode)
                yield self.triple(bnode, RDFS.subClassOf,
                    self.language.namespace[t._operator])
                for i, param in enumerate(t.params, start=1):
                    yield from self.set_type(bnode, param, predicate=RDF[f"_{i}"])
            else:
                if not t.params:
                    node = self.language.namespace[t._operator]
                # else:
                #     node = self.cache[t]
                yield self.triple(
                    variable,
                    predicate / RDFS.subClassOf,  # type: ignore
                    node)

    def add_flow(self, item: Flow1[Type | Operator]) \
            -> tuple[list[list[Variable]], list[Variable]]:

        # Unit flows
        if isinstance(item, Type):
            current = next(self.generator)
            self.attr_type[current].append(item)
            return [[current]], [current]

        elif isinstance(item, Operator):
            current = next(self.generator)
            self.attr_via[current].append(item)
            return [[current]], [current]

        elif isinstance(item, OR) and all(
                isinstance(i, (Type, Operator)) for i in item.items):
            current = next(self.generator)
            for i in item.items:
                if isinstance(i, Operator):
                    self.attr_via[current].append(i)
                elif isinstance(i, Type):
                    self.attr_type[current].append(i)
            return [[current]], [current]

        # Sequential flows
        elif isinstance(item, (SKIP, LINK)):
            skip = not isinstance(item, LINK)
            subs = [self.add_flow(i) for i in item.items]
            for (_, exits), (entrances, _) in zip(subs, subs[1:]):
                for exit_point in exits:
                    assert not self.conns[exit_point]
                    self.skips[exit_point] = skip
                    self.conns[exit_point] = entrances
            return subs[0][0], subs[-1][-1]

        # Branching flows
        else:
            assert isinstance(item, (OR, AND)), type(item)
            subs = [self.add_flow(i) for i in item.items]
            # Exit points are just the concatenation of final nodes. Entry
            # points differ depending on whether we should consider them
            # conjunctively or disjunctively.
            exits = flatten(s[-1] for s in subs)
            entrances = [flatten(p) for p in product(*(s[0] for s in subs))] \
                if isinstance(item, AND) else flatten(s[0] for s in subs)
            return entrances, exits
