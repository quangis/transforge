"""
Tests for the graph conversion process of algebra expressions. To check the
tests, it's recommended to draw out the graphs with a pen.
"""

# Untested:
# - source labels
# - types

import unittest

from rdflib import Namespace, Graph, BNode, Literal, RDF, RDFS, URIRef
from rdflib.term import Node
from rdflib.compare import to_isomorphic, graph_diff
from rdflib.tools.rdf2dot import rdf2dot

from typing import Iterator, Dict, Optional, Union

from transformation_algebra import error
from transformation_algebra.type import Type, TypeVar
from transformation_algebra.expr import \
    Expr, TransformationAlgebra, Data, Operation
from transformation_algebra.rdf import TA, TransformationGraph, TANamespace


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


def graph_auto(alg: TransformationAlgebra, namespace: Namespace,
        value: Union[Expr, Type]) -> Graph:
    """
    Transform an expression to a transformation graph.
    """
    g = TransformationGraph(alg, namespace,
        include_labels=False, include_types=False, include_kinds=False)
    if isinstance(value, Expr):
        root = BNode()
        g.add_expr(value, root)
    else:
        assert isinstance(value, Type)
        g.add_type(value)
    return g


def graph_manual(include_steps: bool = False, include_kinds: bool = False,
        **steps: Step) -> Graph:
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
            g.add((nodes[i], TA.via, step.transformer))
        elif step.internal:
            kind = TA.InternalData
            g.add((nodes[step.internal], TA.internal, nodes[i]))
        else:
            kind = TA.SourceData

        if include_steps:
            g.add((root, TA.step, nodes[i]))

        if include_kinds:
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
        alg = TransformationAlgebra()
        alg.add(f, a)
        ALG = TANamespace('ALG#', alg)

        self.assertIsomorphic(
            graph_auto(alg, ALG, f(a)),
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
        alg = TransformationAlgebra()
        alg.add(f, g)
        ALG = TANamespace('ALG#', alg)

        self.assertIsomorphic(
            graph_auto(alg, ALG, f(g)),
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
        alg = TransformationAlgebra()
        alg.add(f, g, a)
        ALG = TANamespace('ALG#', alg)

        self.assertIsomorphic(
            graph_auto(alg, ALG, f(g, a)),
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
        alg = TransformationAlgebra()
        alg.add(f, g, h, a)
        ALG = TANamespace('ALG#', alg)

        self.assertIsomorphic(
            graph_auto(alg, ALG, h(g, a).primitive()),
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
        alg = TransformationAlgebra()
        alg.add(f, g, h, a, b)
        ALG = TANamespace('ALG#', alg)

        self.assertIsomorphic(
            graph_auto(alg, ALG, h(g, a, b).primitive()),
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
        alg = TransformationAlgebra()
        alg.add(f, id)
        ALG = TANamespace('ALG#', alg)

        self.assertIsomorphic(
            graph_auto(alg, ALG, f(id).primitive()),
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
        alg = TransformationAlgebra()
        alg.add(f, g, h)
        ALG = TANamespace('ALG#', alg)

        self.assertIsomorphic(
            graph_auto(alg, ALG, f(h).primitive()),
            graph_auto(alg, ALG, f(g))
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
        alg = TransformationAlgebra()
        alg.add(a, f, g, h, e)
        ALG = TANamespace('ALG#', alg)

        self.assertIsomorphic(
            graph_auto(alg, ALG, f(g, h, e, a)),
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
        alg = TransformationAlgebra()
        alg.add(a, b, outer, inner, f)
        ALG = TANamespace('ALG#', alg)

        self.assertIsomorphic(
            graph_auto(alg, ALG, outer(inner(f, a), b)),
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

    def test_function_abstraction_body(self):
        """
        The body of an abstraction may be a function.
        """
        A = Type.declare("A")
        a = Data(A, name="a")
        f = Operation((A ** A ** A) ** A ** A, name="f")
        g = Operation(A ** A ** A, name="g")
        h = Operation(A ** A ** A, name="h", derived=lambda x: g(x))
        alg = TransformationAlgebra()
        alg.add(f, g, a)
        ALG = TANamespace('ALG#', alg)

        self.assertIsomorphic(
            graph_auto(alg, ALG, f(h, a).primitive()),
            graph_manual(
                a=Step(),
                g=Step(ALG.g, input="λ"),
                f=Step(ALG.f, input=["g", "a"]),
                λ=Step(internal="f", input="a"),
            )
        )

    def test_type_reuse(self):
        """
        Test that type nodes are reused when transformed to RDF.
        """
        A = Type.declare("A")
        F = Type.declare("F", params=1)
        G = Type.declare("G", params=2)
        alg = TransformationAlgebra()
        alg.add_types(A, F, G)
        ALG = TANamespace('ALG#', alg)

        g = Graph()
        n1 = BNode()
        n2 = BNode()
        g.add((n1, RDFS.subClassOf, ALG.F))
        g.add((n1, RDF._1, ALG.A))
        g.add((n2, RDFS.subClassOf, ALG.G))
        g.add((n2, RDF._1, n1))
        g.add((n2, RDF._2, n1))
        self.assertIsomorphic(
            graph_auto(alg, ALG, G(F(A), F(A))),
            g
        )

    def test_type_variable_reuse(self):
        """
        Test that bound type variables are properly reused.
        """
        A = Type.declare("A")
        F = Type.declare("F", params=2)
        alg = TransformationAlgebra()
        alg.add_types(A, F)
        ALG = TANamespace('ALG#', alg)
        g = Graph()
        x, y = TypeVar(), TypeVar()
        x.bind(A)
        y.bind(A)
        n1 = BNode()
        g.add((n1, RDFS.subClassOf, ALG.F))
        g.add((n1, RDF._1, ALG.A))
        g.add((n1, RDF._2, ALG.A))
        self.assertIsomorphic(
            graph_auto(alg, ALG, F(x, y)),
            g
        )
