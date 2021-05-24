"""
This module extends transformation algebras such that its expressions can be
parsed as RDF graphs.
"""

from __future__ import annotations

from transformation_algebra.expr import \
    TransformationAlgebra, Expr, Base, Application, Abstraction, Data, \
    Operation

import io
from rdflib import Graph, Namespace, BNode, Literal
from rdflib.term import Node
from rdflib.namespace import RDF
from rdflib.tools.rdf2dot import rdf2dot

TA = Namespace(
    "https://github.com/quangis/transformation-algebra/"
    "TransformationAlgebra.rdf#"
)


class TransformationRDF(TransformationAlgebra):
    def __init__(self, prefix: Namespace, *nargs, **kwargs):
        self.namespace = prefix
        super().__init__(*nargs, **kwargs)

    def parse_rdf(self, graph: Graph, string: str) -> None:

        graph.bind("cct", self.namespace)
        expr = self.parse(string)
        if expr:
            self.rdf(graph, expr.primitive())

    def rdf(self, g: Graph, expr: Expr, **inputs: Node) -> BNode:
        """
        Translate the given expression to RDF and add it to the given graph.
        Return the output node (with an `rdf:type` of either `ta:Data` or
        `ta:Operation`). Input nodes that match the labels in the expression
        are appropriately attached.
        """
        τ = expr.type
        assert τ

        g.bind("ta", TA)

        output = BNode()
        if isinstance(expr, Base):
            assert expr.definition.name

            alg_type = getattr(self.namespace, expr.definition.name)
            g.add((output, RDF.type, alg_type))

            if isinstance(expr.definition, Data) and expr.label:
                try:
                    # TODO or directly connect to input?
                    g.add((output, TA.input, inputs[expr.label]))
                except KeyError as e:
                    msg = f"no input node named '{expr.label}'"
                    raise RuntimeError(msg) from e

        elif isinstance(expr, Application):
            f = self.rdf(g, expr.f)
            x = self.rdf(g, expr.x)
            g.add((f, TA.input, x))
            g.add((f, TA.output, output))

        elif isinstance(expr, Abstraction):
            assert isinstance(expr.body, Expr)
            f = self.rdf(g, expr.body)
            g.add((output, TA.input, f))

        g.add((output, TA.type, Literal(τ)))
        g.add((output, RDF.type, TA.Operation if τ.is_function() else TA.Data))

        return output


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
