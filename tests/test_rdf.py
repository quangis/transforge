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
            op: Optional[URIRef] = None,
            input: Union[str, list[str]] = [],
            type: Optional[Type] = None,
            internal: Optional[str] = None):
        self.inputs = [input] if isinstance(input, str) else input
        self.type = type
        self.transformer = op
        self.internal = internal
        assert not (internal and op) and not (op and not input)


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
            kind = TA.TransformedData
            g.add((nodes[i], TA.transformer, step.transformer))
        elif step.internal:
            kind = TA.InternalData
            g.add((nodes[step.internal], TA.internal, nodes[i]))
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
                f=Step(ALG.f, input="a")
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
                λ=Step(internal="f"),
                f=Step(ALG.f, input="g"),
                g=Step(ALG.g, input="λ")
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
                λ=Step(internal="f", input="a"),
                f=Step(ALG.f, input=["g", "a"]),
                g=Step(ALG.g, input="λ"),
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
                h=Step(ALG.h, input=["f2", "a"]),
                f2=Step(ALG.f, input="f1"),
                f1=Step(ALG.f, input="λ"),
                λ=Step(internal="h", input="a"),
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
                h=Step(ALG.h, input=["f2", "a", "b"]),
                f2=Step(ALG.f, input=["λ", "f1"]),
                f1=Step(ALG.f, input="λ"),
                λ=Step(internal="h", input=["a", "b"]),
                a=Step(),
                b=Step(),
            )
        )

    def test_empty_abstraction_as_parameter(self):
        """
        Test that an abstraction that does nothing, as in `f (λx.x)`, indeed
        has only an internal node `λ` that directly feeds `f`.
        """

        A = Type.declare("A")
        f = Operation((A ** A) ** A, name="f")
        id = Operation(A ** A, name="id", derived=lambda x: x)
        alg = TransformationAlgebraRDF('alg', ALG)
        alg.add(f, id)

        self.assertIsomorphic(
            graph_auto(alg, f(id).primitive()),
            graph_manual(
                f=Step(ALG.f, input="λ"),
                λ=Step(internal="f"),
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
                f=Step(ALG.f, input=["a", "g", "h", "e"]),
                gλ=Step(internal="f", input=["a", "h", "e"]),
                hλ=Step(internal="f", input=["a", "g", "e"]),
                eλ=Step(internal="f", input=["a", "g", "h"]),
                g=Step(ALG.g, input="gλ"),
                h=Step(ALG.h, input="hλ"),
                e=Step(ALG.e, input="eλ"),
            )
        )

    def test_nested_operation_as_parameter(self):
        """
        When an operator is used as a parameter that in turn uses an operator
        as parameter, all the values passed to the outer operation should be
        available at the beginning of the pipeline of the inner operations. We
        do that by feeding the outer internal operation to the inner one. See
        issues #37 and #41.
        """
        A = Type.declare("A")
        a = Data(A, name="a")
        b = Data(A, name="b")
        outer = Operation((A ** A) ** A ** A, name="outer")
        inner = Operation((A ** A) ** A ** (A ** A), name="inner")
        f = Operation(A ** A, name="f")
        alg = TransformationAlgebraRDF('alg', ALG)
        alg.add(a, b, outer, inner, f)

        self.assertIsomorphic(
            graph_auto(alg, outer(inner(f, a), b)),
            graph_manual(
                a=Step(),
                b=Step(),
                outer=Step(ALG.outer, input=["inner", "b"]),
                inner=Step(ALG.inner, input=["f", "a", "outerλ"]),
                f=Step(ALG.f, input="innerλ"),
                innerλ=Step(internal="inner", input=["a", "outerλ"]),
                outerλ=Step(internal="outer", input="b"),
            )
        )
