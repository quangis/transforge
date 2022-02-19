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
from collections import defaultdict, deque

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
        combination = "\n} UNION {\n".join(
            s for x in xs if (s := "\n".join(x))
        )
        if combination:
            yield f"{{{combination}}}"
    elif xs:
        yield from xs[0]


def sparql(*elems: str | Iterable[str]) -> str:
    return "\n".join(chain.from_iterable(
        (elem,) if isinstance(elem, str) else elem for elem in elems
    ))


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

        # Every step occurring in this query
        self.steps: list[Variable] = []

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
        Obtain a SPARQL query.
        """
        return sparql(
            "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>",
            "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>",
            "PREFIX ta: <https://github.com/quangis/transformation-algebra#>",
            "SELECT ?workflow WHERE {",
            self.stmts_output_type() if by_output else (),
            self.stmts_bag(operators=True) if by_operators else (),
            self.stmts_bag(types=True) if by_types else (),
            self.stmts_chronology(self.output) if by_chronology else (),
            "} GROUP BY ?workflow"
        )

    def sparql_chronology_steps(self, step: int | None = None) -> str:
        """
        Obtain a SPARQL query that reveals the exact solutions available at a
        step (or the last step if none is given).
        """

        return sparql(
            "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>",
            "SELECT ?workflow ", (f"?{node}L" for node in self.steps),
            "WHERE {",
            self.stmts_chronology(self.output, stop_at=step, with_labels=True),
            "}"
        )

    def sparql_chronology_diagnostics(self) -> Iterator[tuple[int, str]]:
        """
        Obtain multiple SPARQL queries that reveal how many solutions there are
        at each step in the process.
        """

        for d, _ in enumerate(self.steps):
            yield d, sparql(
                "SELECT (COUNT(*) AS ?number_of_results) WHERE {",
                self.stmts_chronology(self.output, stop_at=d),
                "}"
            )

    def stmts_output_type(self) -> Iterator[str]:
        for t in self.type[self.output]:
            yield from self.stmts_type(self.workflow, t,
               TA.output / RDF.type)

    def stmts_bag(self, types: bool = False, operators: bool = False) \
            -> Iterator[str]:

        def units(values: set[Type | Operator]) -> list[str]:
            res: list[str] = []
            for item in values:
                if isinstance(item, Type):
                    res.extend(self.stmts_type(self.workflow, item, TA.member))
                elif isinstance(item, Operator):
                    res.extend(self.stmts_operator(self.workflow, item,
                        TA.member))
            return res

        yield from units(self.bags[0])
        yield from union([
            u for bag in self.bags[1:] if (u := units(bag))
        ])

    def stmts_chronology(self,
            start: Variable,
            entrance: Variable | None = None,
            stop_at: int | None = None,
            with_labels: bool = False,
            _visited: list[Variable] | None = None,
            ) -> Iterator[str]:

        visited = _visited or []

        if not stop_at or len(visited) < stop_at:

            visited.append(start)

            # Connection from the entrance to this node
            if entrance:
                yield self.stmt_connection(entrance, start)
            else:
                assert start == self.output
                yield self.triple(self.workflow, TA.output, self.output)

            # Node's own attributes
            yield from self.stmts_attributes(start)

            # Connecting to rest of the tree
            yield from union([
                chain.from_iterable(
                    self.stmts_chronology(conj, start, stop_at, with_labels,
                        visited)
                    for conj in disj)
                for disj in self.conns[start]
            ])

            if with_labels and _visited is None:
                for v in visited:
                    yield self.triple(v, RDFS.label, Variable(v + "L"))

    def triple(self, *items: Node | Path) -> str:
        result = []
        for item in items:
            try:
                result.append(item.n3())
            except TypeError:
                result.append(str(item))
        return " ".join(result) + "."

    def stmt_connection(self, entrance: Variable, target: Variable) -> str:
        # assert any(entrance in a for a in self.conns[target])
        if self.type[entrance] and not self.type[target]:
            if self.skips[entrance]:
                return self.triple(entrance, ~TA.feeds * ZeroOrMore, target)
            else:
                return f"BIND({entrance.n3()} AS {target.n3()})"
        else:
            if self.skips[entrance]:
                return self.triple(entrance, ~TA.feeds * OneOrMore, target)
            else:
                return self.triple(entrance, ~TA.feeds, target)

    def stmts_attributes(self, target: Variable) -> Iterator[str]:
        yield from union([self.stmts_type(target, t)
            for t in self.type[target]
        ])
        yield from union([self.stmts_operator(target, o)
            for o in self.via[target]])

    def stmts_operator(self, variable: rdflib.Variable, op: Operator,
            predicate: Node | Path = TA.via) -> Iterator[str]:

        if op.definition:
            from warnings import warn
            warn(f"query used a non-primitive operation {op.name}")
        yield self.triple(variable, predicate, self.language.namespace[op])

    def stmts_type(self, variable: rdflib.Variable, type: Type,
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
                    yield from self.stmts_type(bnode, param,
                        predicate=RDF[f"_{i}"])
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
            self.steps.append(current)
            self.type[current].append(item)
            return [[current]], [current]

        elif isinstance(item, Operator):
            current = next(self.generator)
            self.steps.append(current)
            self.via[current].append(item)
            return [[current]], [current]

        elif isinstance(item, OR) and all(
                isinstance(i, (Type, Operator)) for i in item.items):
            current = next(self.generator)
            self.steps.append(current)
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
