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
from itertools import count, chain, product
from typing import Iterator, Iterable, Union, TypeVar
from collections import defaultdict

from transformation_algebra.flow import Flow, Flow1, FlowShorthand, \
    STEPS, JUMPS, AND, OR
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


def union(xs: list[Iterable[str]]) -> Iterator[str]:
    if len(xs) > 1:
        yield "{{{}}}".format("\n} UNION {\n".join("\n".join(x) for x in xs))
    elif xs:
        yield from xs[0]


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
            allow_noncanonical: bool = False):

        self.language = lang
        self.flow: Flow1[Type | Operator] = Flow.shorthand(flow)
        self.generator = iter(Variable(f"_{i}") for i in count())

        self.allow_noncanonical = allow_noncanonical

        # Remember which types occur in the flow in its totality
        self.bags: list[set[Type | Operator]] = Flow.bags(self.flow)

        # Connect each node to a disjunction of conjunction of nodes
        self.conns: dict[Variable, list[list[Variable]]] = defaultdict(list)
        self.skips: dict[Variable, bool] = defaultdict(bool)
        self.via: dict[Variable, list[Operator]] = defaultdict(list)
        self.type: dict[Variable, list[Type]] = defaultdict(list)

        self.cache: dict[Type, Variable] = dict()

        self.workflow = rdflib.Variable("workflow")

        entrances, _ = self.add_flow(self.flow)
        assert len(entrances) == 1 and len(entrances[0]) == 1
        self.output = entrances[0][0]

    def sparql(self,
            by_output: bool = True,
            by_types: bool = True,
            by_operators: bool = True,
            by_chronology: bool = True) -> str:
        """
        Convert the flow to a full SPARQL query.
        """
        return "\n".join(chain([
            "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>",
            "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>",
            "PREFIX ta: <https://github.com/quangis/transformation-algebra#>",
            "SELECT ?workflow WHERE {"],
            self.output_type() if by_output else (),
            self.bag(operators=True) if by_operators else (),
            self.bag(types=True) if by_types else (),
            self.chronology(self.output) if by_chronology else (),
            ["} GROUP BY ?workflow"]

        ))

    def output_type(self) -> Iterator[str]:
        for t in self.type[self.output]:
            yield from self.attr_type(self.workflow, t,
               TA.output / RDF.type)

    def bag(self, types: bool = False, operators: bool = False) \
            -> Iterator[str]:

        def units(values: set[Type | Operator]) -> list[str]:
            res: list[str] = []
            for item in values:
                if isinstance(item, Type):
                    res.extend(self.attr_type(self.workflow, item, TA.member))
                elif isinstance(item, Operator):
                    res.extend(self.attr_operator(self.workflow, item,
                        TA.member))
            return res

        yield from units(self.bags[0])
        yield from union([
            u for bag in self.bags[1:] if (u := units(bag))
        ])

    def chronology(self, target: Variable,
            entrance: Variable | None = None) -> Iterator[str]:

        # Connection from the entrance to this node
        if entrance:
            if self.type[entrance] and not self.type[target]:
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
            assert target == self.output
            yield self.triple(self.workflow, TA.output, self.output)

        # Node's own attributes
        yield from self.attributes(target)

        # Connecting to rest of the tree
        yield from union([
            chain.from_iterable(self.chronology(conj, target) for conj in disj)
            for disj in self.conns[target]
        ])

    def triple(self, *items: Node | Path) -> str:
        result = []
        for item in items:
            try:
                result.append(item.n3())
            except TypeError:
                result.append(str(item))
        return " ".join(result) + "."

    def attributes(self, target: Variable) -> Iterator[str]:
        yield from union([self.attr_type(target, t)
            for t in self.type[target]
        ])
        yield from union([self.attr_operator(target, o)
            for o in self.via[target]])

    def attr_operator(self, variable: rdflib.Variable, op: Operator,
            predicate: Node | Path = TA.via) -> Iterator[str]:

        if op.definition:
            from warnings import warn
            warn(f"query used a non-primitive operation {op.name}")
        yield self.triple(variable, predicate, self.language.namespace[op])

    def attr_type(self, variable: rdflib.Variable, type: Type,
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

            if not self.allow_noncanonical and not t.variables():
                raise RuntimeError(f"Encountered non-canonical type {t}")

            assert isinstance(t, TypeOperation) and t.operator != Function
            if t.params:  # and t not in self.cache:
                self.cache[t] = bnode = next(self.generator)
                yield self.triple(variable, predicate, bnode)
                yield self.triple(bnode, RDFS.subClassOf,
                    self.language.namespace[t._operator])
                for i, param in enumerate(t.params, start=1):
                    yield from self.attr_type(bnode, param, predicate=RDF[f"_{i}"])
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
            self.type[current].append(item)
            return [[current]], [current]

        elif isinstance(item, Operator):
            current = next(self.generator)
            self.via[current].append(item)
            return [[current]], [current]

        elif isinstance(item, OR) and all(
                isinstance(i, (Type, Operator)) for i in item.items):
            current = next(self.generator)
            for i in item.items:
                if isinstance(i, Operator):
                    self.via[current].append(i)
                elif isinstance(i, Type):
                    self.type[current].append(i)
            return [[current]], [current]

        # Sequential flows
        elif isinstance(item, (STEPS, JUMPS)):
            skip = isinstance(item, JUMPS)
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
