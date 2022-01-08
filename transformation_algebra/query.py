"""
This module intends to provide a way to express transformation trees (i.e.
chains of types and operations) in such a way that a query might use it to
check if a given transformation expression contains types and operations in the
specified order.
"""

from __future__ import annotations

import rdflib
from rdflib.paths import Path, ZeroOrMore, OneOrMore
from rdflib.term import Node
from rdflib.namespace import RDFS, RDF
from itertools import count
from typing import Iterator, Union, Literal

from transformation_algebra.flow import Flow, Flow1, SERIES, LINKED, AND, OR, \
    FlowShorthand
from transformation_algebra.type import Type, TypeOperation, \
    Function, TypeVariable
from transformation_algebra.expr import Operator
from transformation_algebra.lang import Language
from transformation_algebra.graph import TA

Operators = Flow[Union[Type, Operator]]  # really: OR[Operator]
Triple = tuple[Node, Union[Path, Node], Node]


class Query(object):  # TODO subclass rdflib.Query?
    """
    A flow captures some relevant aspects of a conceptual process, in terms of
    the sequence of elements that must occur in it. For example, the following
    flow holds that there must be datatypes `A` and `B` that are fed to an
    operation f that eventually results in a datatype `C`:

    [C, f, AND(A, B)]

    Note that the flow is 'reversed' (from output to input). This allows for a
    convenient tree-like notation, but it may trip you up.

    Furthermore, for succinct notation, lists are interpreted as `SEQ`
    transformation flows and the ellipsis indicates we may skip any number of
    steps.
    """

    def __init__(self,
            lang: Language,
            flow: FlowShorthand[Type | Operator] | None = None,
            generator: Iterator[rdflib.Variable] | None = None,
            by_output: bool = False,
            by_types: bool = False,
            # by_operators: bool = True,
            by_order: bool = True):

        self.language = lang
        self.taxonomy = lang.taxonomy()
        self.namespace = lang.namespace
        self.generator = generator or \
            iter(rdflib.Variable(f"n{i}") for i in count(start=1))
        self.statements: list[str] = []
        self.cache: dict[Type, rdflib.Variable] = dict()

        # Filter by...
        self.by_output = by_output
        self.by_types = by_types
        # self.by_operators = by_operators
        self.by_order = by_order

        self.root = rdflib.Variable("workflow")
        self.output = rdflib.Variable("output")

        self.flow: Flow1[Type | Operator] | None = None
        if flow:
            self.flow = Flow.shorthand(flow)

        if self.flow:

            self.triple(self.root, TA.output, self.output)

            if self.by_output:
                for item in Flow.leaves(self.flow, targets=True):
                    if isinstance(item, Type):
                        self.set_type(self.output, item)

            for item in Flow.leaves(self.flow):
                # if self.by_operators and isinstance(item, Operator):
                #     self.set_operator(self.root, item, TA.member)

                if self.by_types and isinstance(item, Type):
                    self.set_type(self.root, item, TA.member)

            if self.by_order:
                self.connect(self.output, "start", False, self.flow)

    def spawn(self) -> Query:
        return Query(self.language, None, self.generator)

    def sparql(self) -> str:
        """
        Convert the flow to a full SPARQL query.
        """

        query = [
            "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>",
            "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>",
            "PREFIX ta: <https://github.com/quangis/transformation-algebra#>",
            # f"PREFIX {self.prefix}: <{self.namespace}>",
            "SELECT ?workflow ?description WHERE {",
            "OPTIONAL {?workflow rdfs:comment ?description}",
        ]
        query.extend(self.statements)
        query.append("} GROUP BY ?workflow ?description")
        return "\n".join(query)

    def triple(self, *items: Node | Path) -> None:
        """
        Add a triple pattern.
        """
        result = []
        for item in items:
            try:
                result.append(item.n3())
            except TypeError:
                result.append(str(item))
        self.statements.append(" ".join(result) + ".")

    def set_operator(self, variable: rdflib.Variable, op: Operator,
            predicate: Node | Path = TA.via) -> None:
        if op.definition:
            import warnings
            warnings.warn(f"query used a non-primitive operation {op.name}")
        self.triple(variable, predicate, self.namespace[op.name])

    def set_type(self, variable: rdflib.Variable, type: Type,
            predicate: Node | Path = RDF.type) -> None:
        """
        Produce SPARQL constraints for the given (non-function) type.
        """

        t = type.instance().normalize()

        if isinstance(t, TypeVariable):
            # If a type in a trace query contains variables, it must be a
            # wildcard --- because we don't do anything with it
            assert t.wildcard
        elif t in self.taxonomy:
            self.triple(variable,
                predicate / (RDFS.subClassOf * ZeroOrMore),  # type: ignore
                self.namespace[t.text(spacer=",")])
        else:
            assert isinstance(t, TypeOperation) and t.operator != Function
            if t.params and t not in self.cache:
                self.cache[t] = bnode = next(self.generator)
                self.triple(variable, predicate, bnode)
                self.triple(bnode, RDFS.subClassOf,
                    self.namespace[t._operator.name])
                for i, param in enumerate(t.params, start=1):
                    self.set_type(bnode, param, predicate=RDF[f"_{i}"])
            else:
                if not t.params:
                    node = self.namespace[t._operator.name]
                else:
                    node = self.cache[t]
                self.triple(
                    variable,
                    predicate / (RDFS.subClassOf * ZeroOrMore),  # type: ignore
                    node)

    def __str__(self) -> str:
        return "\n".join(self.statements)

    def union(self, *queries: Query) -> None:
        self.statements.append("{")
        self.statements.append("\n} UNION {\n".join(str(q) for q in queries))
        self.statements.append("}")

    def connect(self,
            lvar: rdflib.Variable,
            lunit: Type | Operator | OR[Operator] | Literal["start"] | None,
            lskip: bool,
            item: Flow1[Type | Operator]) -> rdflib.Var | None:

        # Handle units
        is_operator = isinstance(item, Operator) or (
            isinstance(item, OR) and
            all(isinstance(i, Operator) for i in item.items)
        )

        if is_operator or isinstance(item, Type):

            # Connect this node to left node
            if lunit == "start" or (is_operator and isinstance(lunit, Type)):
                if lskip:
                    var = next(self.generator)
                    self.triple(lvar, ~TA.feeds * ZeroOrMore, var)
                else:
                    var = lvar
            else:
                var = next(self.generator)
                self.triple(
                    lvar,
                    ~TA.feeds * OneOrMore if lskip else ~TA.feeds,
                    var
                )

            # Set properties of node itself
            if isinstance(item, Type):
                self.set_type(var, item)
            elif isinstance(item, Operator):
                self.set_operator(var, item)
            else:
                assert isinstance(item, OR)
                subs = []
                for i in item:
                    assert isinstance(i, Operator)
                    subquery = self.spawn()
                    subquery.set_operator(var, i)
                    subs.append(subquery)
                self.union(*subs)
            return var

        elif isinstance(item, AND):

            for i in item:
                subquery = self.spawn()
                subquery.connect(lvar, lunit, lskip, i)
                self.statements.extend(subquery.statements)

        elif isinstance(item, OR):

            subs = []
            for i in item:
                subquery = self.spawn()
                subquery.connect(lvar, lunit, lskip, i)
                subs.append(subquery)
            self.union(*subs)

        else:
            assert isinstance(item, (SERIES, LINKED))

            skip_between = isinstance(item, SERIES)

            for current in item.items[:-1]:
                assert isinstance(current, (Type, Operator)) or (
                    isinstance(current, OR) and
                    all(isinstance(i, Operator) for i in current.items)
                )
                var = self.connect(lvar, lunit, lskip, current)
                lvar = var
                lunit = current
                lskip = skip_between

            lskip = skip_between
            current = item.items[-1]

            return self.connect(lvar, lunit, lskip, current)

        return None
