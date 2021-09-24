"""
Tests for the graph conversion process of algebra expressions. To check the
tests, it's recommended to draw out the graphs with a pen.
"""

import unittest

from rdflib import Namespace, Graph, BNode, RDF, URIRef
from rdflib.term import Node
from rdflib.compare import to_isomorphic, graph_diff

from typing import Iterator, Dict, Optional, Any

from transformation_algebra import error
from transformation_algebra.type import Type
from transformation_algebra.expr import \
    Expr, TransformationAlgebra, Data, Operation
from transformation_algebra.rdf import TransformationAlgebraRDF, TA

ALG = Namespace('ALG#')


class Step(object):
    def __init__(self,
            transformer: Optional[URIRef],
            *inputs: Any,
            type: Optional[Type] = None,
            internals: list[Any] = []):
        self.inputs = inputs
        self.type = type
        self.transformer = transformer
        self.internals = internals


class TestAlgebraRDF(unittest.TestCase):

    def assertIsomorphic(self, actual: Graph, expected: Graph) -> None:
        """
        Compare two graphs to test if they are equal
        """
        actual = to_isomorphic(actual)
        expected = to_isomorphic(expected)

        nodes_expected = set(node for triple in expected for node in triple
            if not isinstance(node, BNode))
        nodes_actual = set(node for triple in actual for node in triple
            if not isinstance(node, BNode))

        # Test that there are no missing nodes
        for n in nodes_expected:
            self.assertIn(n, nodes_actual)

        # Test that there are no extraneous nodes
        self.assertFalse(nodes_actual - nodes_expected)

        # If all nodes are there, there must be some wrong connection
        self.assertEqual(actual, expected)

    def compare(self,
            alg: TransformationAlgebraRDF,
            expr: Expr,
            manual: Dict[Any, Step]) -> None:
        """
        Compare the automatically constructed graph with a manual counterpart.
        """
        root = BNode()

        # Calculate the graph from the expression
        actual = Graph()
        alg.rdf_expr(actual, expr, root,
            include_labels=False, include_types=False)

        # Manually construct the expected graph
        expected = Graph()
        nodes = {i: BNode() for i in manual}
        expected.add((root, RDF.type, TA.Transformation))

        for i, step in manual.items():
            if step.transformer:
                assert step.inputs
                kind = TA.TransformedData
                expected.add((nodes[i], TA.transformer, step.transformer))
            elif not step.inputs:
                kind = TA.SourceData
            else:
                kind = TA.InternalData

            expected.add((root, TA.step, nodes[i]))
            expected.add((nodes[i], RDF.type, kind))
            for internal in step.internals:
                expected.add((nodes[i], TA.internal, nodes[internal]))

            for j in step.inputs:
                expected.add((nodes[j], TA.feeds, nodes[i]))

        self.assertIsomorphic(actual, expected)

    def test_basic(self):
        A = Type.declare("A")
        x = Data(A, name="x")
        f = Operation(A ** A, name="f")
        alg = TransformationAlgebraRDF('alg', ALG)
        alg.add(f, x)

        self.compare(alg,
            f(x),
            {1: Step(None), 2: Step(ALG.f, 1)}
        )

    def test_operation_as_parameter_with_argument(self):
        A = Type.declare("A")
        x = Data(A, name="x")
        f = Operation((A ** A) ** A ** A, name="f")
        g = Operation(A ** A, name="g")
        alg = TransformationAlgebraRDF('alg', ALG)
        alg.add(f, g, x)

        self.compare(alg,
            f(g, x), {
                "x": Step(None),
                "λ": Step(None, "x"),
                "f": Step(ALG.f, "g", "x", internals=["λ"]),
                "g": Step(ALG.g, "λ")
            }
        )
