"""
This module intends to provide a way to express transformation graphs (i.e.
chains of types and operations) in such a way that a query might use it to
check if a given transformation expression contains types and operations in the
specified order.
"""

from __future__ import annotations

from rdflib.term import Node, URIRef, Variable, BNode
from rdflib.graph import Graph
from itertools import count, chain
from typing import Iterable, Iterator
from collections import deque, defaultdict

from transforge.namespace import RDF, TF
from transforge.type import Type
from transforge.expr import Operator
from transforge.lang import Language
from transforge.graph import (
    TransformationGraph, CyclicTransformationGraphError)


def union(prefix: str, subjects: Iterable[Node]) -> Iterator[str]:
    """
    Convenience: produce a union of objects in a SPARQL constraint.
    """
    statements = []
    for s in subjects:
        assert isinstance(s, URIRef)
        statements.append(f"{prefix} {s.n3()}.")
    if len(statements) > 1:
        yield f"{{{' } UNION { '.join(statements)}}}"
    elif len(statements) == 1:
        yield statements[0]


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

    def __init__(self, lang: Language, graph: Graph,
            with_noncanonical_types: bool = False, by_io: bool = True,
            by_types: bool = True, by_operators: bool = True,
            by_chronology: bool = True, unfold_tree: bool = False):

        self.lang = lang

        self.root: Node = graph.value(predicate=RDF.type, object=TF.Task, any=False)

        if not self.root:
            raise ValueError(f"No {TF.Task.n3()} found in the graph.")

        self.graph = TransformationGraph(language=lang)
        self.graph += graph
        self.graph.parse_shortcuts()

        self.by_io = by_io
        self.by_types = by_types
        self.by_operators = by_operators
        self.by_chronology = by_chronology
        self.unfold_tree = unfold_tree

        # Keep track of the type and operator of each step
        self.type: dict[Variable, list[Node]] = dict()
        self.operator: dict[Variable, list[Node]] = dict()

        # A surjective mapping, associating variables with the step nodes of
        # the graph on which the query is based.
        self.steps: dict[Variable, Node] = dict()

        # In the other direction, if we don't unfold the graph into a tree,
        # there is a 1:1 mapping between variables and step nodes. Otherwise,
        # we will get a new variable for every time we encounter that step.
        self.generator = iter(Variable(f"_{i}") for i in count())
        self.variables: dict[Node, Variable] | None = \
            None if self.unfold_tree else dict()

        # Assigned variables to the outputs and inputs of the transformation
        self.inputs: list[Variable] = []
        self.outputs: list[Variable] = []

        # Connections between the variables
        self.before: dict[Variable, list[Variable]] = defaultdict(list)
        self.after: dict[Variable, list[Variable]] = defaultdict(list)

        # Traverse the graph
        for node in self.graph.objects(self.root, TF.output):
            self.outputs.append(self.assign_variables(node))

    @staticmethod
    def from_list(lang: Language, aspects: list[Operator | Type | list],
            *nargs, **kwargs) -> TransformationQuery:
        """
        Alternative constructor for making a query. By default, a query is
        created based on an `rdflib.Graph`, but that can be verbose. This is a
        quicker convenience method that constructs the graph from a nested list
        where each nesting level refers to a preceding step. For example, an
        object like the following:

            [D, cb2d_option1, cb2d_option2, [B, a2b, [A]], [C]]

        … gets turned into a `TransformationQuery` based on a graph like this:

            @base <https://language.namespace/>
            @prefix : <https://github.com/quangis/transforge#>
            [] a :Task; :output [
                :type <D>;
                :via <cb2d_option1>, <cb2d_option2>;
                :from [
                    :type <B>;
                    :via <a2b>;
                    :from [
                        :type <A>
                    ]
                ], [
                    :type <C>
                ]
            ].

        Should the query not be based on a tree, only the standard constructor
        will be useful.
        """
        g = TransformationGraph(lang)
        root = BNode()
        g.add((root, RDF.type, TF.Task))

        def f(node, aspect) -> Node:
            if isinstance(aspect, Type):
                g.add((node, TF.type, lang.uri(aspect)))
            elif isinstance(aspect, Operator):
                g.add((node, TF.via, lang.uri(aspect)))
            else:
                assert isinstance(aspect, list)
                predecessor = BNode()
                g.add((node, TF["from"], predecessor))
                for a in aspect:
                    f(predecessor, a)
            return node

        output = BNode()
        for a in aspects:
            f(output, a)

        g.add((root, TF.output, output))

        return TransformationQuery(lang, g, *nargs, **kwargs)

    def assign_variables(self, node: Node, path: list[Node] = []) -> Variable:
        """
        Depth-first traversal through the graph, assigning one or more
        variables to each step node. Raises error when encountering a cycle.
        """
        if node in path:
            raise CyclicTransformationGraphError

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
        self.type[var] = list(self.graph.objects(node, TF.type))
        self.operator[var] = list(self.graph.objects(node, TF.via))

        for next_node in self.graph.objects(node, TF["from"]):
            next_var = self.assign_variables(next_node, path + [node])
            self.before[var].append(next_var)
            self.after[next_var].append(var)

        return var

    def sparql(self) -> str:
        """
        Obtain a SPARQL query.
        """

        def sparql(*elems: str | Iterable[str]) -> str:
            return "\n".join(chain.from_iterable(
                (elem,) if isinstance(elem, str) else elem for elem in elems
            ))

        result = sparql(
            f"BASE <{self.lang.namespace}>",
            "PREFIX : <https://github.com/quangis/transforge#>",
            "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>",
            "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>",
            "SELECT ?workflow WHERE {",
            "GRAPH ?workflow {",
            "{SELECT ?workflow WHERE {",
            "?workflow a :Transformation.",
            self.operators() if self.by_operators else (),
            self.types() if self.by_types else (),
            "} GROUP BY ?workflow}",
            self.io() if self.by_io else (),
            self.chronology() if self.by_chronology else (),
            "}",
            "} GROUP BY ?workflow"
        )
        return result

    def types(self) -> Iterator[str]:
        """
        Conditions for matching on the bag of types used in a query.
        """

        # Only the types that *definitely* occur
        types: set[URIRef] = set()
        for type_choice in self.type.values():
            if len(type_choice) == 1:
                assert isinstance(type_choice[0], URIRef)
                types.add(type_choice[0])

        for type in types:
            yield f"?workflow :contains/rdfs:subClassOf* {type.n3()}."
            # yield f"{next(self.generator).n3()} {next(self.generator).n3()} {type.n3()}."

        # Also include union types. TODO this is temporary until #79 is
        # resolved; see also:
        # https://github.com/quangis/transformation-algebra/issues/77#issuecomment-1215064807
        for type_choice in self.type.values():
            if len([t for t in type_choice if t not in types]) >= 2:
                yield from union(
                    f"{next(self.generator).n3()} {next(self.generator).n3()}",
                    type_choice)

    def operators(self) -> Iterator[str]:
        """
        Conditions for matching on the bag of operators.
        """

        # Only the operators that *definitely* occur
        operators: set[URIRef] = set()
        for operator_choice in self.operator.values():
            if len(operator_choice) == 1:
                assert isinstance(operator_choice[0], URIRef)
                operators.add(operator_choice[0])

        for operator in operators:
            yield f"?workflow :contains {operator.n3()}."

    def io(self) -> Iterator[str]:
        """
        Conditions for matching on input and outputs of the query.
        """
        for output in self.graph.objects(self.root, TF.output):
            yield from union("?workflow :output/:type/rdfs:subClassOf*",
                self.graph.objects(output, TF.type))

        for input in self.graph.objects(self.root, TF.input):
            yield from union("?workflow :input/:type/rdfs:subClassOf*",
                self.graph.objects(input, TF.type))

    def chronology(self) -> Iterator[str]:
        """
        Conditions for matching the specific order of a query.
        """
        # We can assume at this point that there will not be any cycles

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

            # afters = " ".join(a.n3() for a in self.after[current])
            # yield f"{{SELECT DISTINCT {current.n3()} {afters} WHERE {{"

            # Connect the initial nodes (ie outputs)
            if not self.after[current]:
                assert current in self.outputs
                yield f"?workflow :output {current.n3()}."

            # Write connections to previous nodes (ie ones that come after)
            for c in self.after[current]:
                assert c in visited
                # Current and after nodes may refer to the same step if the
                # graph has a bare operator and then a bare type in
                # succession, or if the after node has no information at
                # all
                if not self.operator.get(c) and (not self.type.get(c) or (
                        self.operator.get(current)
                        and not self.type.get(current))):
                    yield f"{c.n3()} :from* {current.n3()}."
                else:
                    yield f"{c.n3()} :from+ {current.n3()}."

            # Write operator/type properties of this step
            yield from union(f"{current.n3()} :via",
                self.operator.get(current, ()))
            yield from union(f"{current.n3()} :type/rdfs:subClassOf*",
                self.type.get(current, ()))

            # Make sure as early as possible that there is no earlier on the
            # same branch
            # for c in self.after[current]:
            #     types = self.type.get(current) or []
            #     if not len(types) == 1:
            #         continue
            #     between = next(self.generator)
            #     yield "FILTER NOT EXISTS {"
            #     yield f"{current.n3()} ^:to+ {between.n3()}."
            #     yield f"{c.n3()} :to* {between.n3()}."
            #     for t in types:
            #         yield f"{between.n3()} :type/rdfs:subClassOf* {t.n3()}."
            #     yield "}"

            # Make sure we don't check the same node
            # yield "}}"

            visited.add(current)

            # Add successors to queue
            for b in self.before[current]:
                if b not in visited:
                    waiting.append(b)
