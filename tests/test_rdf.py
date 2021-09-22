import unittest

from rdflib import Namespace, Graph, BNode, RDF
from rdflib.term import Node
from rdflib.compare import to_isomorphic, graph_diff
from rdflib.tools.rdf2dot import rdf2dot

from typing import Iterator

from transformation_algebra import error
from transformation_algebra.type import Type
from transformation_algebra.expr import \
    TransformationAlgebra, Data, Operation
from transformation_algebra.rdf import TransformationAlgebraRDF, TA


def nodes(graph: Graph) -> set[Node]:
    return set(node for triple in graph for node in triple
        if not isinstance(node, BNode))


def visualize(graph: Graph, filename: str) -> None:
    """
    Visualize a graph. For debugging.
    """

    with open(filename, 'w') as f:
        rdf2dot(graph, f)


class TestAlgebraRDF(unittest.TestCase):

    def test_basic(self):

        A = Type.declare("A")
        x = Data(A, name="x")
        f = Operation(A ** A, name="f")

        ALG = Namespace('ALG#')
        alg = TransformationAlgebraRDF('alg', ALG)
        alg.add(f, x)

        actual = Graph()
        alg.rdf_expr(actual, f(x), root=BNode(),
            include_labels=False, include_types=False)

        expected = Graph()
        root = BNode()
        step_types = [TA.SourceData, TA.TransformedData]
        steps = [BNode() for _ in step_types]

        expected.add((root, RDF.type, TA.Transformation))
        for t, n in zip(step_types, steps):
            expected.add((root, TA.step, n))
            expected.add((n, RDF.type, t))

        expected.add((steps[1], TA.transformer, ALG.f))
        expected.add((steps[0], TA.feeds, steps[1]))

        actual = to_isomorphic(actual)
        expected = to_isomorphic(expected)

        visualize(actual, 'out_actual.dot')
        visualize(expected, 'out_expected.dot')

        with open('out_expected.dot', 'w') as f:
            rdf2dot(expected, f)

        # Check that every non-blank node that should be there is there
        missing = nodes(expected) - nodes(actual)
        extraneous = nodes(actual) - nodes(expected)
        self.assertFalse(missing)
        self.assertFalse(extraneous)

        # If all nodes are there, there must be some wrong connection
        self.assertEqual(actual, expected)
