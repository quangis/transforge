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
from rdflib.namespace import Namespace, RDFS, RDF
from itertools import count
from typing import Iterator, Union, Optional

from transformation_algebra.flow import Flow, Flow1, SEQ, AND, OR, FlowShorthand
from transformation_algebra.type import Type, TypeOperation, \
    Function, TypeVariable
from transformation_algebra.expr import Operator
from transformation_algebra.graph import TA

Operators = Flow[Union[Type, Operator]]  # really: OR[Operator]
Triple = tuple[Node, Union[Path, Node], Node]


class Query(object):  # TODO subclass rdflib.Query?
    """
    A flow captures some relevant aspects of a conceptual process, in terms of
    the sequence of elements that must occur in it. For example, the following
    flow holds that there must be datatypes `A` and `B` that are fed to an
    operation f that eventually results in a datatype `C`:

    [C, ..., f, AND(A, B)]

    Note that the flow is 'reversed' (from output to input). This allows for a
    convenient tree-like notation, but it may trip you up.

    Furthermore, for succinct notation, lists are interpreted as `SEQ`
    transformation flows and the ellipsis indicates we may skip any number of
    steps.
    """

    def __init__(self,
            ns: Namespace,
            flow: FlowShorthand[Type | Operator] | None = None,
            generator: Iterator[rdflib.Variable] | None = None):

        self.namespace = ns
        self.generator = generator or \
            iter(rdflib.Variable(f"n{i}") for i in count(start=1))
        self.statements: list[str] = []

        if flow:
            flow1: Flow1[Type | Operator] = Flow.shorthand(flow)
            self.connect(next(self.generator), None, False, flow1)

    def spawn(self) -> Query:
        return Query(self.namespace, None, self.generator)

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
            "?workflow rdf:type ta:Transformation.",
            "OPTIONAL {?workflow rdfs:comment ?description}",
        ]
        query.extend(self.statements)
        query.append("} GROUP BY ?workflow ?description")
        return "\n".join(query)

        # TODO return sparql.Query instead, to avoid parsing at all. This does
        # not avoid parsing:
        # return sparql.prepareQuery("\n".join(query),
        #         initNs={'ta': TA, 'rdf': RDF, 'rdfs': RDFS}
        # )

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

    def set_operator(self, variable: rdflib.Variable, op: Operator) -> None:
        assert op.name, "operator must have a name at this point"
        if op.definition:
            import warnings
            warnings.warn(f"query used a non-primitive operation {op.name}")
        self.triple(variable, TA.via, self.namespace[op.name])

    def set_type(self, variable: rdflib.Variable, type: Type,
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
            assert t._operator.name, "type must have a name at this point"

            pred = RDF.type if index is None else RDF[f"_{index}"]
            if t.params:
                bnode = next(self.generator)
                self.triple(variable,
                    pred,
                    bnode)
                self.triple(bnode,
                    RDFS.subClassOf,
                    self.namespace[t._operator.name])
                for i, param in enumerate(t.params, start=1):
                    self.set_type(bnode, param, index=i)
            else:
                self.triple(
                    variable,
                    pred / (RDFS.subClassOf * ZeroOrMore),  # type: ignore
                    self.namespace[t._operator.name])

    def __str__(self) -> str:
        return "\n".join(self.statements)

    def union(self, *queries: Query) -> None:
        self.statements.append("{")
        self.statements.append("\n} UNION {\n".join(str(q) for q in queries))
        self.statements.append("}")

    def connect(self,
            lvar: rdflib.Variable,
            lunit: Type | Operator | OR[Operator] | None,
            lskip: bool,
            item: Flow1[Type | Operator]) -> rdflib.Var | None:

        # Handle units
        is_operator = isinstance(item, Operator) or (
            isinstance(item, OR) and
            all(isinstance(i, Operator) for i in item.items)
        )

        if is_operator or isinstance(item, Type):

            # Connect this node to left node
            if not lunit:
                var = lvar
                self.triple(
                    rdflib.Variable("workflow"),
                    TA.result / (~TA.feeds * ZeroOrMore)
                    if lskip else TA.result,
                    var)
            elif is_operator and isinstance(lunit, Type):
                if lskip:
                    var = next(self.generator)
                    self.triple(
                        lvar,
                        ~TA.feeds * ZeroOrMore,
                        var
                    )
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
                for i in item.items:
                    assert isinstance(i, Operator)
                    subquery = self.spawn()
                    subquery.set_operator(var, i)
                    subs.append(subquery)
                self.union(*subs)
            return var

        elif isinstance(item, AND):

            for i in item.items:
                subquery = self.spawn()
                subquery.connect(lvar, lunit, lskip, i)
                self.statements.extend(subquery.statements)

        elif isinstance(item, OR):

            subs = []
            for i in item.items:
                subquery = self.spawn()
                subquery.connect(lvar, lunit, lskip, i)
                subs.append(subquery)
            self.union(*subs)

        else:
            assert isinstance(item, SEQ)

            for i in range(len(item.items) - 1):
                current = item.items[i]
                assert isinstance(current, (Type, Operator)) or (
                    isinstance(current, OR) and
                    all(isinstance(i, Operator) for i in current.items)
                )
                var = self.connect(lvar, lunit, lskip or item.skips[i], current)
                lvar = var
                lunit = current
                lskip = False
            return self.connect(lvar, lunit, item.skips[-2], item.items[-1])

        return None
