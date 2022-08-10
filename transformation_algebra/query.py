"""
This module intends to provide a way to express transformation graphs (i.e.
chains of types and operations) in such a way that a query might use it to
check if a given transformation expression contains types and operations in the
specified order.
"""

from __future__ import annotations

from rdflib.term import Node, URIRef, Variable
from rdflib.namespace import RDF
from rdflib.graph import Graph
from itertools import count, chain
from typing import Iterable
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

        [] a :Task; :output
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

    def __init__(self, lang: Language, graph: TransformationGraph,
            with_noncanonical_types: bool = False, by_io: bool = True,
            by_types: bool = True, by_operators: bool = True,
            by_chronology: bool = True, unfold_tree: bool = False):

        self.lang = lang

        self.graph = TransformationGraph(language=lang)
        self.graph += graph

        self.root = self.graph.value(predicate=RDF.type, object=TA.Task,
            any=False)

        if not self.root:
            raise ValueError(f"No {TA.Task.n3()} found in the graph.")

        self.by_io = by_io
        self.by_types = by_types
        self.by_operators = by_operators
        self.by_chronology = by_chronology
        self.unfold_tree = unfold_tree

        # Keep track of the type and operator of each step
        self.type: dict[Variable, URIRef] = dict()
        self.operator: dict[Variable, URIRef] = dict()

        # A surjective mapping, associating variables with the step nodes of
        # the graph on which the query is based.
        self.steps: dict[Variable, Node] = dict()

        self.before: dict[Variable, list[Variable]] = defaultdict(list)
        self.after: dict[Variable, list[Variable]] = defaultdict(list)

        # Assign variables to the outputs of the transformation
        self.outputs: list[Variable] = []

        # If we don't unfold the graph into a tree, there is a 1:1 mapping
        # between variables and step nodes. Otherwise, we will get a new
        # variable for every time we encounter that step.
        self.variables: dict[Node, Variable] | None = \
            None if self.unfold_tree else dict()

        self.generator = iter(Variable(f"_{i}") for i in count())

        for node in self.graph.objects(self.root, TA.output):
            self.outputs.append(self.assign_variables(node))

    def assign_variables(self, node: Node, path: list[Node] = []) -> Variable:
        """
        Depth-first traversal through the graph, assigning one or more
        variables to each step node. Raises error when encountering a cycle.
        """
        if node in path:
            raise ValueError("cycle")

        if self.unfold_tree:
            var = next(self.generator)
        else:
            assert isinstance(self.variables, dict)
            if node in self.variables:
                var = self.variables[node]
            else:
                var = self.variables[node] = next(self.generator)

        assert var not in self.steps or self.steps[var] == node
        self.steps[var] = node

        if tp := self.graph.value(node, TA.type, any=False):
            self.type[var] = tp
        if op := self.graph.value(node, TA.via, any=False):
            self.operator[var] = op

        for next_node in self.graph.objects(node, TA["from"]):
            next_var = self.assign_variables(next_node, path + [node])
            self.before[var].append(next_var)
            self.after[next_var].append(var)

        return var

    def run(self, graph: Graph) -> set[Node]:
        """
        Find all workflows in a graph that match this query.
        """
        return set(r.workflow for r in graph.query(self.sparql()))

    def sparql(self) -> str:
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
            "GRAPH ?workflow {",
            "?workflow a :Transformation.",
            self.io() if self.by_io else (),
            self.operators() if self.by_operators else (),
            self.types() if self.by_types else (),
            self.chronology() if self.by_chronology else (),
            "}} GROUP BY ?workflow"
        )
        return result

    def types(self) -> Iterable[str]:
        """
        Conditions for matching on the bag of types used in a query.
        """
        return [f"?workflow :contains/rdfs:subClassOf* {tp.n3()}."
            for tp in set(self.type.values())]

    def operators(self) -> Iterable[str]:
        """
        Conditions for matching on the bag of operators.
        """
        return [f"?workflow :contains {op.n3()}."
            for op in set(self.operator.values())]

    def io(self) -> Iterable[str]:
        """
        Conditions for matching on input and outputs of the query.
        """
        result = []
        for i, output in enumerate(self.graph.objects(self.root, TA.output)):
            result.append(f"?workflow :output ?output{i}.")
            for tp in self.graph.objects(output, TA.type):
                assert isinstance(tp, URIRef)
                result.append(f"?output{i} :type/rdfs:subClassOf* {tp.n3()}.")
        for i, input in enumerate(self.graph.objects(self.root, TA.input)):
            result.append(f"?workflow :input ?input{i}.")
            for tp in self.graph.objects(input, TA.type):
                assert isinstance(tp, URIRef)
                result.append(f"?input{i} :type/rdfs:subClassOf* {tp.n3()}.")
        return result

    def chronology(self) -> Iterable[str]:
        """
        Conditions for matching the specific order of a query.
        """
        # We can assume at this point that there will not be any cycles

        result: list[str] = []
        visited: set[Variable] = set()
        waiting: list[Variable] = list(self.outputs)
        processing: deque[Variable] = deque()
        while True:

            # Add only those steps from the waitlist for which all subsequent
            # steps have already been visited
            new_waiting = []
            for w in waiting:
                if all(v in visited for v in self.after[w]):
                    processing.append(w)
                else:
                    new_waiting.append(w)
                waiting = new_waiting

            if not len(processing):
                break

            current = processing.popleft()

            if current in visited:
                continue

            if not self.after[current]:
                assert current in self.outputs
                result.append(f"?workflow :output {current.n3()}.")

            # Write connections to previous nodes (ie ones that come after)
            for c in self.after[current]:
                assert c in visited
                # Current and after nodes may refer to the same step if the
                # graph has a bare operator and then a bare type in
                # succession, or if the after node has no information at
                # all
                if not self.operator.get(c) and (not self.type.get(c) or (
                        self.operator.get(current) and not self.type.get(current))):
                    result.append(f"{current.n3()} :to* {c.n3()}.")
                else:
                    result.append(f"{current.n3()} :to+ {c.n3()}.")

            # Write operator/type properties of this step
            if operator := self.operator.get(current):
                result.append(f"{current.n3()} :via {operator.n3()}.")
            if type := self.type.get(current):
                result.append(f"{current.n3()} :type/rdfs:subClassOf* {type.n3()}.")
            visited.add(current)

            # Add successors to queue
            for b in self.before[current]:
                if b not in visited:
                    waiting.append(b)

        return result
