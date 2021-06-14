"""
This module extends transformation algebras such that its expressions can be
parsed as RDF graphs.
"""

from __future__ import annotations

from transformation_algebra.type import Type, TypeOperation, TypeVar, \
    Function, TypeOperator
from transformation_algebra.expr import \
    TransformationAlgebra, Expr, Base, Application, Abstraction, Data, \
    Operation, Variable, Definition

import io
from itertools import count
from rdflib import URIRef, Graph, Namespace, BNode, Literal
from rdflib.term import Node
from rdflib.namespace import RDF, RDFS
from rdflib.tools.rdf2dot import rdf2dot
from rdflib.plugins import sparql

from typing import Dict, Union, List, Iterator, Optional, Tuple

TA = Namespace(
    "https://github.com/quangis/transformation-algebra/"
    "TransformationAlgebra.rdf#"
)


class TransformationAlgebraRDF(TransformationAlgebra):
    def __init__(self, prefix: str, namespace: Union[Namespace, str]):
        self.prefix = prefix
        self.namespace = Namespace(namespace) \
            if isinstance(namespace, str) else namespace
        super().__init__()

    def uri(self, value: Union[TypeOperator, Definition]) -> URIRef:
        """
        Obtain the URI node for an operation or type operator.
        """
        assert value in self
        if isinstance(value, TypeOperator):
            return getattr(self.namespace, value.name)
        else:
            assert isinstance(value, Definition) and value.name
            return getattr(self.namespace, value.name)

    def vocabulary(self) -> Graph:
        """
        Produce an RDF vocabulary for describing expressions in terms of the
        types and operations defined for this transformation algebra.
        """
        vocab = Graph()

        # Add type operators to the vocabulary
        for t in self.types:
            previous_uri = None
            current: Optional[TypeOperator] = t
            while current:
                current_uri = self.uri(current)
                vocab.add((current_uri, RDF.type, TA.Type))
                if previous_uri:
                    vocab.add((previous_uri, RDF.type, current_uri))
                previous_uri = current_uri
                current = current.supertype

        # Add operations to the vocabulary
        for d in self.definitions.values():
            node = self.uri(d)
            typenode = TA.Data if isinstance(d, Data) else TA.Operation
            vocab.add((node, RDF.type, typenode))
            if d.description:
                vocab.add((node, RDFS.label, d.description))

        return vocab

    def rdf_type(self, graph: Graph, value: Type) -> Node:
        """
        Add an RDF representation of a type to the given graph.
        """
        t = value.instance()
        if isinstance(t, TypeOperation):
            if t.operator == Function:
                operator_node = TA.Function
            else:
                operator_node = self.uri(t.operator)

            node = BNode()
            graph.add((node, RDF.type, operator_node))

            if t.params:
                graph.add((node, RDF.type, RDF.Seq))

            for i, param in enumerate(t.params, start=1):
                param_node = self.rdf_type(graph, param)
                graph.add((node, getattr(RDF, f"_{i}"), param_node))

            return node
        else:
            # TODO don't make new node if we already encountered this variable
            assert isinstance(t, TypeVar)
            node = BNode()
            graph.add((node, RDF.type, TA.TypeVariable))
            graph.add((node, TA.variable, Literal(str(t))))
            return node

    def parse_rdf(self, graph: Graph, string: str) -> BNode:
        """
        Convenience function to parse an expression and add it to an RDF graph
        in one go.
        """
        expr = self.parse(string)
        if expr:
            return self.expr2rdf(graph, expr.primitive())

    def expr2rdf(self, g: Graph, expr: Expr,
            inputs: Dict[str, Node] = {}) -> BNode:
        """
        Translate the given expression to RDF and add it to the given graph.
        Return the root node (of type `ta:Transformation`) to which all
        transformation steps connect via `ta:step`. Input nodes that match the
        labels in the expression are appropriately attached via `ta:source`.
        """

        assert not expr.type.is_function()

        g.bind("ta", TA)
        g.bind(self.prefix, self.namespace)

        root = BNode()
        g.add((root, TA.type, TA.Transformation))
        self._expr2rdf(g, expr, root=root, step=BNode(), inputs=inputs)
        return root

    def _expr2rdf(self, g: Graph, expr: Expr,
            root: Node, step: Node, inputs: Dict[str, Node]) -> Node:
        """
        Translate the given expression to RDF. Return the final node of type
        `ta:Operation` or `ta:Data`.
        """
        assert expr.type

        # Determine the type of this node
        if expr.type.is_function():
            type_node = TA.Operation
        elif isinstance(expr, Variable):
            type_node = TA.Variable
        else:
            type_node = TA.Data

        # Add connections to input or output nodes
        if isinstance(expr, Base):
            assert expr.definition.name
            definition_node = self.uri(expr.definition)
            g.add((step, RDF.type, definition_node))
            if isinstance(expr.definition, Operation):
                g.add((root, TA.step, step))
            elif isinstance(expr.definition, Data) and expr.label:
                try:
                    g.add((step, TA.source, inputs[expr.label]))
                except KeyError as e:
                    msg = f"no input node named '{expr.label}'"
                    raise RuntimeError(msg) from e
        elif isinstance(expr, Abstraction):
            g.add((step, RDF.type, type_node))
            assert isinstance(expr.body, Expr)
            f = self._expr2rdf(g, expr.body, root, step=BNode(), inputs=inputs)
            g.add((step, TA.input, f))
        elif isinstance(expr, Application):
            f = self._expr2rdf(g, expr.f, root, step=step, inputs=inputs)
            x = self._expr2rdf(g, expr.x, root, step=BNode(), inputs=inputs)
            g.add((f, TA.input, x))
            if type_node == TA.Data:
                step = BNode()
                g.add((f, TA.output, step))

        # Add semantic information on the type of node
        if not isinstance(expr, Application) or type_node == TA.Data:
            type2_node = self.rdf_type(g, expr.type)
            g.add((step, RDF.type, type_node))
            g.add((step, TA.type, type2_node))

        return step

    def sparql_type(self, type: Type, name: str,
            name_generator: Iterator[str]) -> Iterator[str]:
        """
        Produce SPARQL constraints for the given (non-function) type.
        """

        t = type.instance()

        if isinstance(t, TypeVar):
            # If a type in a trace query contains variables, it must be a
            # wildcard --- because we don't do anything with it
            assert t.wildcard
        else:
            assert isinstance(t, TypeOperation)
            assert t.operator != Function
            yield (
                f"?{name} "
                f"rdf:type "
                f"<{self.uri(t.operator)}>."
            )
            for i, param in enumerate(t.params, start=1):
                next_name = next(name_generator)
                yield (
                    f"?{name} rdf:_{i} ?{next_name}."
                )
                yield from self.sparql_type(param, next_name, name_generator)


def dot(g: Graph) -> str:
    """
    Return a string of GraphViz dot syntax. You can pass the output to such
    programs as `xdot` to visualize the RDF graph.
    """
    # rdf2dot uses deprecated cgi.escape function; this hack solves that
    import cgi
    import html
    stream = io.StringIO()
    cgi.escape = html.escape  # type: ignore
    rdf2dot(g, stream)
    return stream.getvalue()


class Chain(object):
    """
    A chain captures some relevant aspects of the conceptual process described
    by an algebra expression, in terms of the sequence of types and operations
    that must occur in it.

    The same can also be described in terms of semantic linear time logic
    formulae (SLTL), but since we will be using SPARQL to search through
    workflows, the approach chosen here makes for a straightforward translation.
    """

    def __init__(self, *chain: Union[None, Type, Operation, List[Chain]]):
        """
        Starting from the output node, describe which data types and operations
        must occur, in what order. A `None` value indicates that we may skip
        zero or more steps, and a list indicates that every one of its
        subchains must at some point occur in one of the paths.
        """
        self.chain = list(chain)
        # Lists may only occur in the final place of a chain
        assert not any(isinstance(p, list) for p in self.chain[:-1])

    def to_sparql(self, algebra: TransformationAlgebraRDF) -> sparql.Query:
        """
        Convert this chain to a SPARQL query.
        """

        query = [
            "SELECT ?workflow WHERE {",
            "?workflow ta:transformation ?transformation.",
            "?tranformation rdf:type ta:Transformation.",
            "?transformation ta:step ?step.",
            "?step ta:output ?node0.",
            "FILTER NOT EXISTS {?next_step ta:input ?node0}."
        ]
        query.extend(self.trace(algebra=algebra))
        query.append("}")

        print()
        print("\n".join(query))
        print()

        return sparql.prepareQuery("\n".join(query),
            initNs={'ta': TA, 'rdf': RDF}
        )

    def trace(self,
            algebra: TransformationAlgebraRDF,
            name_generator: Optional[Iterator[str]] = None,
            previous: Optional[Tuple[str, Union[Type, Operation]]] = None,
            skip: bool = False) -> Iterator[str]:
        """
        Trace the paths between each node in this chain to produce a set of
        SPARQL constraints.
        """
        name_generator = name_generator or iter(f"node{i}" for i in count())

        for current in self.chain:
            if current is None:
                skip = True
            elif isinstance(current, list):
                for subchain in current:
                    yield from subchain.trace(
                        algebra, name_generator, previous, skip)
            else:
                assert isinstance(current, (Operation, Type))
                name = next(name_generator)
                if previous:
                    yield (
                        f"?{previous[0]} "
                        f"{self.path(previous[1], current, skip)} "
                        f"?{name}.")

                if isinstance(current, Operation):
                    yield f"?{name} rdf:type {algebra.uri(current)}."
                else:
                    assert isinstance(current, Type)
                    next_name = next(name_generator)
                    yield f"?{name} ta:type ?{next_name}."
                    yield from algebra.sparql_type(current, next_name,
                            name_generator)

                skip = False
                previous = name, current

    def path(self,
            before: Union[Type, Operation],
            after: Union[Type, Operation],
            skip: bool) -> str:
        """
        Produce a SPARQL property path describing the connection between two
        nodes that represent either a data type or an operation. `skip`
        indicates that multiple intermediaries may lie between.
        """
        # See also rdflib.paths

        repeat = "+" if skip else ""
        if isinstance(before, Type):
            if isinstance(after, Type):
                return f"(^ta:output/ta:input){repeat}"
            else:
                assert isinstance(after, Operation)
                return f"(^ta:output){repeat}"
        else:
            assert isinstance(before, Operation)
            if isinstance(after, Type):
                return f"(ta:input){repeat}"
            else:
                assert isinstance(after, Type)
                return f"(ta.input/^ta:output){repeat}"
