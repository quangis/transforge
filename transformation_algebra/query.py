"""
This module intends to provide a way to express transformation trees (i.e.
chains of types and operations) in such a way that a query might use it to
check if a given transformation expression contains types and operations in the
specified order.
"""

from __future__ import annotations

from rdflib import BNode
from rdflib.term import Node, Variable, Path
from rdflib.namespace import RDF
from rdflib.util import guess_format
from itertools import count, chain
from typing import Iterable
from collections import deque

from transformation_algebra.lang import Language
from transformation_algebra.graph import TA, TransformationGraph

def triple(*items: Node | Path) -> str:
    result = []
    for item in items:
        try:
            result.append(item.n3())
        except TypeError:
            result.append(str(item))
    return " ".join(result) + "."


class TransformationQuery(object):
    """
    A transformation can be used to query other transformations: it captures
    some relevant aspects of a process, in terms of the flow of conceptual
    types and operations that must occur in it. For example, the following flow
    holds that there must be types `A` and `B` that are fed to an operation `f`
    that eventually results in a type `C`:

        [] a :Transformation; :output
            [ type: "C"; from:
                [ via: "f"; from:
                    [ type: "A" ],
                    [ type: "B" ]
                ]
            ].

    Note that the flow is 'reversed' (from output to input).
    """

    def __init__(self, lang: Language, graph: TransformationGraph | str,
            format: str = "turtle", with_noncanonical_types: bool = False):

        self.lang = lang

        self.graph = TransformationGraph(language=lang)
        if isinstance(graph, TransformationGraph):
            self.graph += graph
        else:
            self.graph.parse(graph, format=format or guess_format(graph))
        self.graph.parse_shortcuts()

        self.with_noncanonical_types = with_noncanonical_types
        self.root = self.graph.value(predicate=RDF.type, object=TA.Query,
            any=False)
        self.output = self.graph.value(subject=self.root, predicate=TA.output,
            any=False)

        assert self.root and self.output

    def sparql(self,
            by_output: bool = True,
            by_input: bool = False,
            by_types: bool = True,
            by_operators: bool = True,
            by_chronology: bool = True) -> str:
        """
        Obtain a SPARQL query.
        """

        def sparql(*elems: str | Iterable[str]) -> str:
            return "\n".join(chain.from_iterable(
                (elem,) if isinstance(elem, str) else elem for elem in elems
            ))

        result = sparql(
            "PREFIX : <https://github.com/quangis/transformation-algebra#>",
            "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>",
            "SELECT ?workflow WHERE {",
            # self.stmts_output_type() if by_output else (),
            # self.stmts_input_type() if by_input else (),
            # self.stmts_bag(operators=True) if by_operators else (),
            # self.stmts_bag(types=True) if by_types else (),
            # self.stmts_chronology(self.output) if by_chronology else (),
            self.chronology() if by_chronology else (),
            "} GROUP BY ?workflow"
        )
        # print(result)
        return result

    def chronology(self) -> str:
        result = [
            "\n# Chronology",
            "?workflow :output ?_0."
        ]

        generator = iter(Variable(f"_{i}") for i in count())
        mapping: dict[BNode, Variable] = dict()

        # This dictionary helps us to determine whether a node is 'finished',
        # e.g. whether a subsequent type/operation can refer 
        finished: dict[Variable, bool] = dict()

        queue: deque[BNode] = deque([self.output])
        while len(queue) > 0:
            current = queue.popleft()
            assert current not in mapping

            # Assign a variable to this node
            mapping[current] = var = next(generator)

            # Connect it to its predecessors
            for after in self.graph.subjects(TA["from"], current):
                assert after in mapping  # TODO definitely doesn't hold
                bvar = mapping[after]

                if finished[bvar]:
                    result.append(f"{var.n3()} :feeds+ ?{bvar}.")
                else:
                    result.append(f"{var.n3()} :feeds* ?{bvar}.")

            # Determine correct types and operations
            finished[var] = False
            for op in self.graph.objects(current, TA.via):
                result.append(f"{var.n3()} :via {op.n3()}.")
                finished[var] = True
            for tp in self.graph.objects(current, RDF.type):
                result.append(f"{var.n3()} a/rdfs:subClassOf {tp.n3()}.")

            # Add successors
            for before in self.graph.objects(current, TA["from"]):
                queue.append(before)

        return "\n".join(result)
