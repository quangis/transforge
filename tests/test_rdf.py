"""
Tests for the graph conversion process of algebra expressions. To check the
tests, it's recommended to draw out the graphs with a pen.
"""

# Untested:
# - source labels
# - types

import unittest

from rdflib import Namespace, Graph, BNode, RDF, URIRef
from rdflib.term import Node
from rdflib.compare import to_isomorphic, graph_diff
from rdflib.tools.rdf2dot import rdf2dot

from typing import Iterator, Dict, Optional, Union

from transformation_algebra import error
from transformation_algebra.type import Type
from transformation_algebra.expr import \
    Expr, TransformationAlgebra, Data, Operation
from transformation_algebra.rdf import TransformationAlgebraRDF, TA

ALG = Namespace('ALG#')


class Step(object):
    """
    A step is a data structure only used for testing. It represents a data node
    in a transformation graph, in a format that makes it easy to convert a
    dictionary of them to a real `rdflib` `Graph`.
    """

    def __init__(self,
            *inputs: str,
            op: Optional[URIRef] = None,
            type: Optional[Type] = None,
            internal_to: Union[str, list[str]] = []):
        self.inputs = inputs
        self.type = type
        self.transformer = op
        self.internal_to = [internal_to] if isinstance(internal_to, str) \
            else internal_to


def graph_auto(alg: TransformationAlgebraRDF, expr: Expr) -> Graph:
    """
    Transform an expression to a transformation graph.
    """
    root = BNode()
    g = Graph()
    alg.rdf_expr(g, expr, root,
        include_labels=False, include_types=False)
    return g


def graph_manual(**steps: Step) -> Graph:
    """
    Manually construct a transformation graph.
    """
    root = BNode()
    g = Graph()
    nodes = {i: BNode() for i in steps}
    g.add((root, RDF.type, TA.Transformation))

    for i, step in steps.items():
        if step.transformer:
            assert step.inputs and not step.internal_to
            kind = TA.TransformedData
            g.add((nodes[i], TA.transformer, step.transformer))
        elif step.internal_to:
            kind = TA.InternalData
            for internal in step.internal_to:
                g.add((nodes[internal], TA.internal, nodes[i]))
        else:
            kind = TA.SourceData

        g.add((root, TA.step, nodes[i]))
        g.add((nodes[i], RDF.type, kind))

        for j in step.inputs:
            g.add((nodes[j], TA.feeds, nodes[i]))
    return g


class TestAlgebraRDF(unittest.TestCase):

    def assertIsomorphic(self, actual: Graph, expected: Graph,
            visualize: bool = False) -> None:
        """
        Compare two graphs to test if they are isomorphic.
        """

        if visualize:
            with open('actual.dot', 'w') as handle:
                rdf2dot(actual, handle)
            with open('expected.dot', 'w') as handle:
                rdf2dot(expected, handle)

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

    def test_basic(self):
        """
        Test the basic case of converting an expression consisting of an
        operation and a data source.
        """
        A = Type.declare("A")
        a = Data(A, name="a")
        f = Operation(A ** A, name="f")
        alg = TransformationAlgebraRDF('alg', ALG)
        alg.add(f, a)

        self.assertIsomorphic(
            graph_auto(alg, f(a)),
            graph_manual(
                a=Step(),
                f=Step("a", op=ALG.f)
            )
        )

    def test_operation_as_sole_parameter(self):
        """
        Operations passed as a parameter must have an internal operation.
        """
        A = Type.declare("A")
        f = Operation((A ** A) ** A, name="f")
        g = Operation(A ** A, name="g")
        alg = TransformationAlgebraRDF('alg', ALG)
        alg.add(f, g)

        self.assertIsomorphic(
            graph_auto(alg, f(g)),
            graph_manual(
                λ=Step(internal_to="f"),
                f=Step("g", op=ALG.f),
                g=Step("λ", op=ALG.g)
            )
        )

    def test_operation_as_parameter(self):
        """
        Operations passed as a parameter must have an internal operation that
        gets the same inputs as the operation that got the parameter operation.
        """
        A = Type.declare("A")
        a = Data(A, name="a")
        f = Operation((A ** A) ** A ** A, name="f")
        g = Operation(A ** A, name="g")
        alg = TransformationAlgebraRDF('alg', ALG)
        alg.add(f, g, a)

        self.assertIsomorphic(
            graph_auto(alg, f(g, a)),
            graph_manual(
                a=Step(),
                λ=Step("a", internal_to="f"),
                f=Step("g", "a", op=ALG.f),
                g=Step("λ", op=ALG.g),
            )
        )

    def test_abstraction_as_parameter(self):
        """
        Test that abstractions have a sensible representation, with an internal
        operation that synthesizes the value of variables.
        """

        A = Type.declare("A")
        a = Data(A, name="a")
        f = Operation(A ** A, name="f")
        g = Operation(A ** A, name="g", derived=lambda x: f(f(x)))
        h = Operation((A ** A) ** A ** A, name="h")
        alg = TransformationAlgebraRDF('alg', ALG)
        alg.add(f, g, h, a)

        self.assertIsomorphic(
            graph_auto(alg, h(g, a).primitive()),
            graph_manual(
                h=Step("f2", "a", op=ALG.h),
                f2=Step("f1", op=ALG.f),
                f1=Step("λ", op=ALG.f),
                λ=Step("a", internal_to="h"),
                a=Step(),
            )
        )

    def test_complex_abstraction_as_parameter(self):
        """
        Same as before, but with a more complex abstraction that uses multiple
        variables.
        """

        A = Type.declare("A")
        a = Data(A, name="a")
        b = Data(A, name="b")
        f = Operation(A ** A ** A, name="f")
        g = Operation(A ** A ** A, name="g", derived=lambda x, y: f(y, f(x, y)))
        h = Operation((A ** A ** A) ** A ** A ** A, name="h")
        alg = TransformationAlgebraRDF('alg', ALG)
        alg.add(f, g, h, a, b)

        self.assertIsomorphic(
            graph_auto(alg, h(g, a, b).primitive()),
            graph_manual(
                h=Step("f2", "a", "b", op=ALG.h),
                f2=Step("λ", "f1", op=ALG.f),
                f1=Step("λ", op=ALG.f),
                λ=Step("a", "b", internal_to="h"),
                a=Step(),
                b=Step(),
            )
        )

    def test_empty_abstraction_as_parameter(self):
        """
        Test that an abstraction that does nothing, as in `f (λx.x)`, indeed
        has only nodes `λ → f`.
        """

        A = Type.declare("A")
        f = Operation((A ** A) ** A, name="f")
        id = Operation(A ** A, name="id", derived=lambda x: x)
        alg = TransformationAlgebraRDF('alg', ALG)
        alg.add(f, id)

        self.assertIsomorphic(
            graph_auto(alg, f(id).primitive()),
            graph_manual(
                f=Step("λ", op=ALG.f),
                λ=Step(internal_to="f"),
            )
        )

    def test_abstraction_same_as_primitive(self):
        """
        Test that an abstraction of the form `f (λx.g x)` is not only
        conceptually the same as `f g`, but actually leads to the same graph.
        """

        A = Type.declare("A")
        f = Operation((A ** A) ** A, name="f")
        g = Operation(A ** A, name="g")
        h = Operation(A ** A, name="h", derived=lambda x: g(x))
        alg = TransformationAlgebraRDF('alg', ALG)
        alg.add(f, g, h)

        self.assertIsomorphic(
            graph_auto(alg, f(h).primitive()),
            graph_auto(alg, f(g))
        )

    def test_cycle(self):
        """
        It is possible to get a cyclic transformation graph when passing
        multiple operations as parameter.
        """
        A = Type.declare("A")
        a = Data(A, name="a")
        f = Operation((A ** A) ** (A ** A) ** (A ** A) ** A ** A, name="f")
        g = Operation(A ** A, name="g")
        h = Operation(A ** A, name="h")
        e = Operation(A ** A, name="e")
        alg = TransformationAlgebraRDF('alg', ALG)
        alg.add(a, f, g, h, e)

        self.assertIsomorphic(
            graph_auto(alg, f(g, h, e, a)),
            graph_manual(
                a=Step(),
                f=Step("a", "g", "h", "e", op=ALG.f),
                gλ=Step("a", "h", "e", internal_to="f"),
                hλ=Step("a", "g", "e", internal_to="f"),
                eλ=Step("a", "g", "h", internal_to="f"),
                g=Step("gλ", op=ALG.g),
                h=Step("hλ", op=ALG.h),
                e=Step("eλ", op=ALG.e),
            )
        )

    def test_nested_operation_as_parameter(self):
        """
        When an operator is used as a parameter that in turn uses an operator
        as parameter, all the values passed to the outer operation should be
        available to all internal operations. We do that by making sure to
        share the internal operation.
        """
        A = Type.declare("A")
        a = Data(A, name="a")
        b = Data(A, name="b")
        f = Operation((A ** A) ** A ** A, name="f")
        g = Operation((A ** A) ** A ** (A ** A), name="g")
        h = Operation(A ** A, name="h")
        alg = TransformationAlgebraRDF('alg', ALG)
        alg.add(a, b, f, g, h)

        self.assertIsomorphic(
            graph_auto(alg, f(g(h, a), b)),
            graph_manual(
                a=Step(),
                b=Step(),
                f=Step("g", "b", op=ALG.f),
                g=Step("h", "a", op=ALG.g),
                h=Step("λ", op=ALG.h),
                λ=Step("a", "b", internal_to=["f", "g"]),
            )
        )
