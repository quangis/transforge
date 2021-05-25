"""
This module extends transformation algebras such that its expressions can be
parsed as RDF graphs.
"""

from __future__ import annotations

from transformation_algebra.expr import \
    TransformationAlgebra, Expr, Base, Application, Abstraction, Data, \
    Operation, Variable

import io
from rdflib import Graph, Namespace, BNode, Literal
from rdflib.term import Node
from rdflib.namespace import RDF
from rdflib.tools.rdf2dot import rdf2dot

from typing import Dict

TA = Namespace(
    "https://github.com/quangis/transformation-algebra/"
    "TransformationAlgebra.rdf#"
)


class TransformationRDF(TransformationAlgebra):
    def __init__(self, prefix: str, namespace: Namespace, *nargs, **kwargs):
        self.prefix = prefix
        self.namespace = namespace
        super().__init__(*nargs, **kwargs)

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
            definition_node = getattr(self.namespace, expr.definition.name)
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
            g.add((step, RDF.type, type_node))
            g.add((step, TA.type, Literal(expr.type)))

        return step


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
