"""
This module intends to provide a way to express transformation trees (i.e.
chains of types and operations) in such a way that a query might use it to
check if a given transformation expression contains types and operations in the
specified order.
"""

from __future__ import annotations

from rdflib.term import Node, BNode, URIRef, Variable
from rdflib.namespace import RDF
from rdflib.util import guess_format
from itertools import count, chain
from typing import Iterable, Iterator
from collections import deque, defaultdict

from transformation_algebra.lang import Language
from transformation_algebra.graph import TA, TransformationGraph

class TransformationQuery(object):
    """
    A transformation can be used to query other transformations: it should then
    capture relevant aspects of the process, as a flowchart of some conceptual
    types and operations that must occur in it. For example, the following
    holds that there must be types `A` and `B` that are fed to an operation `f`
    that eventually results in a type `C`:

        [] a :Transformation; :output
            [ type: "C"; from:
                [ via: "f"; from:
                    [ type: "A" ],
                    [ type: "B" ]
                ]
            ].

    This transformation can be transformed to a SPARQL query that matches any
    such transformation. Note that the flow is 'reversed' (from output to
    input).
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

        self.root = self.graph.value(predicate=RDF.type, object=TA.Query,
            any=False)
        self.output = self.graph.value(subject=self.root, predicate=TA.output,
            any=False)

        if not self.root:
            raise ValueError("The transformation graph is not a ta:Query.")
        if not self.output:
            raise ValueError("The transformation graph has no output.")

    def sparql(self,
            by_io: bool = True,
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
            "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>",
            "SELECT ?workflow WHERE {",
            # "?workflow a :Transformation.",
            self.io() if by_io else (),
            self.operators() if by_operators else (),
            self.types() if by_types else (),
            self.chronology() if by_chronology else (),
            "} GROUP BY ?workflow"
        )
        return result

    def traverse(self) -> Iterator[Node]:
        """
        Breadth-first traversal of all conceptual steps, starting at the output
        node.
        """
        # TODO order
        visited: set[Node] = set()
        queue: deque[Node] = deque([self.output])
        while len(queue) > 0:
            current = queue.popleft()
            assert current not in visited
            yield current
            for before in self.graph.objects(current, TA["from"]):
                if before not in visited:
                    queue.append(before)
            visited.add(current)

    def types(self) -> Iterable[str]:
        """
        Conditions for matching on the bag of types used in a query.
        """
        result = set()
        for node in self.traverse():
            for tp in self.graph.objects(node, RDF.type):
                assert isinstance(tp, URIRef)
                result.add(f"?workflow :member/rdfs:subClassOf {tp.n3()}.")
        return result

    def operators(self) -> Iterable[str]:
        """
        Conditions for matching on the bag of operators.
        """
        result = set()
        for node in self.traverse():
            for op in self.graph.objects(node, TA.via):
                assert isinstance(op, URIRef)
                result.add(f"?workflow :member {op.n3()}.")
        return result

    def io(self) -> Iterable[str]:
        """
        Conditions for matching on input and outputs of the query.
        """
        result = []
        for i, output in enumerate(self.graph.objects(self.root, TA.output)):
            result.append(f"?workflow :output ?output{i}.")
            for tp in self.graph.objects(output, RDF.type):
                assert isinstance(tp, URIRef)
                result.append(f"?output{i} a/rdfs:subClassOf {tp.n3()}.")
        for i, input in enumerate(self.graph.objects(self.root, TA.input)):
            result.append(f"?workflow :input ?input{i}.")
            for tp in self.graph.objects(input, RDF.type):
                assert isinstance(tp, URIRef)
                result.append(f"?input{i} rdf:type/rdfs:subClassOf {tp.n3()}.")
        return result

    def chronology(self) -> Iterable[str]:
        """
        Conditions for matching the specific order of a query.
        """
        result = [
            "?workflow :output ?_0."
        ]

        generator = iter(Variable(f"_{i}") for i in count())
        mapping: dict[BNode, Variable] = dict()

        # Remember what type/operator is assigned to each step
        operators: dict[Variable, URIRef] = dict()
        types: dict[Variable, URIRef] = dict()

        queue: deque[BNode] = deque([self.output])
        while len(queue) > 0:
            current = queue.popleft()
            assert current not in mapping

            # Assign a variable to this node
            mapping[current] = var = next(generator)

            # Determine correct types and operations
            for op in self.graph.objects(current, TA.via):
                assert isinstance(op, URIRef) and var not in operators
                operators[var] = op
            for tp in self.graph.objects(current, RDF.type):
                assert isinstance(tp, URIRef) and var not in types
                types[var] = tp

            # Connect to "predecessors" (ie. the steps that come after)
            for after in self.graph.subjects(TA["from"], current):
                assert after in mapping  # TODO definitely doesn't hold
                assert isinstance(after, BNode)
                avar = mapping[after]

                # Current and after nodes may refer to the same step if the
                # graph has a bare operator and then a bare type in succession,
                # or if the after node has no information at all
                if not operators.get(avar) and (not types.get(avar) or (
                        operators.get(var) and not types.get(var))):
                    result.append(f"{var.n3()} :feeds* {avar.n3()}.")
                else:
                    result.append(f"{var.n3()} :feeds+ {avar.n3()}.")

            # Determine correct types and operations
            if operator := operators.get(var):
                result.append(f"{var.n3()} :via {operator.n3()}.")
            if ctype := types.get(var):
                result.append(f"{var.n3()} a/rdfs:subClassOf {ctype.n3()}.")

            # Add successors
            for before in self.graph.objects(current, TA["from"]):
                assert isinstance(before, BNode)
                queue.append(before)

        return result
