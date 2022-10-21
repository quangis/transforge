"""
Tests for the graph conversion process of algebra expressions. To check the
tests, it's recommended to draw out the graphs with a pen.
"""

from __future__ import annotations
import unittest

from rdflib import Graph, BNode, RDF, RDFS, URIRef
from rdflib.term import Node, Literal
from rdflib.compare import to_isomorphic
from rdflib.tools.rdf2dot import rdf2dot

from transformation_algebra.type import Type, TypeOperator, TypeVariable, \
    SubtypeMismatch, TypeAlias, TypeOperation, Top, Bottom
from transformation_algebra.expr import Expr, Operator, Source, \
    ApplicationError
from transformation_algebra.lang import Language
from transformation_algebra.graph import TA, TEST, TransformationGraph
from transformation_algebra.workflow import WorkflowDict

from typing import Union

class Step(object):
    """
    A step is a data structure only used for testing. It represents a data node
    in a transformation graph, in a format that makes it easy to convert a
    dictionary of them to a real `rdflib` `Graph`.
    """

    def __init__(self,
            via: URIRef | None = None,
            type: URIRef | None = None,
            input: str | list[str] = [],
            internal: str | None = None,
            is_output: bool = False,
            origin: Node | None = None):
        self.via = via
        self.type = type
        self.inputs = [input] if isinstance(input, str) else input
        self.internal = internal
        self.is_output = is_output
        self.origin = origin
        assert not (internal and via) and not (via and not input)


def graph_auto(alg: Language, value: Expr | Type) -> Graph:
    """
    Transform an expression to a transformation graph.
    """
    g = TransformationGraph(alg, minimal=True, with_operators=True,
        with_type_parameters=True, with_noncanonical_types=True)
    if isinstance(value, Expr):
        root = BNode()
        g.add_expr(value, root)
    else:
        assert isinstance(value, Type)
        g.add_type(value)
    return g


def graph_manual(with_classes: bool = False, **steps: Step) -> Graph:
    """
    Manually construct a transformation graph.
    """
    root = BNode()
    g = Graph()
    nodes = {i: BNode() for i in steps}
    if with_classes:
        g.add((root, RDF.type, TA.Transformation))

    for i, step in steps.items():
        if step.via:
            g.add((nodes[i], TA.via, step.via))
        elif step.internal:
            g.add((nodes[step.internal], TA.internal, nodes[i]))

        if step.origin:
            g.add((nodes[i], TA.origin, step.origin))

        if step.is_output:
            g.add((root, TA.output, nodes[i]))

        if step.type:
            g.add((nodes[i], TA.type, step.type))

        for j in step.inputs:
            g.add((nodes[i], TA["from"], nodes[j]))
    return g


TypeOp = Union[TypeOperator, TypeOperation]
Types = Union[TypeOp, tuple[TypeOp]]

def make_taxonomy(lang: Language, d: dict[Types, Types]) -> Graph:
    """
    Manually construct a taxonomy graph.
    """
    g = Graph()

    for sups, subs in d.items():
        for sup in (sups if isinstance(sups, tuple) else (sups,)):
            for sub in (subs if isinstance(subs, tuple) else (subs,)):
                g.add((lang.uri(sub), RDFS.subClassOf, lang.uri(sup)))

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
        A = TypeOperator()
        f = Operator(type=A ** A)
        alg = Language(locals(), namespace=TEST)

        a = Source(type=A)
        self.assertIsomorphic(
            graph_auto(alg, f(a)),
            graph_manual(
                a=Step(),
                f=Step(TEST.f, input="a")
            )
        )

    def test_operation_as_sole_parameter(self):
        """
        Operations passed as a parameter must have an internal operation.
        """
        A = TypeOperator()
        f = Operator(type=(A ** A) ** A)
        g = Operator(type=A ** A)
        alg = Language(locals(), namespace=TEST)

        self.assertIsomorphic(
            graph_auto(alg, f(g)),
            graph_manual(
                λ=Step(internal="f"),
                f=Step(TEST.f, input="g"),
                g=Step(TEST.g, input="λ")
            )
        )

    def test_operation_as_parameter(self):
        """
        Operations passed as a parameter must have an internal operation that
        gets the same inputs as the operation that got the parameter operation.
        """
        A = TypeOperator("A")
        f = Operator(type=(A ** A) ** A ** A)
        g = Operator(type=A ** A)
        alg = Language(locals(), namespace=TEST)

        a = Source(type=A)
        self.assertIsomorphic(
            graph_auto(alg, f(g, a)),
            graph_manual(
                a=Step(),
                λ=Step(internal="f", input="a"),
                f=Step(TEST.f, input=["g", "a"]),
                g=Step(TEST.g, input="λ"),
            )
        )

    def test_abstraction_as_parameter(self):
        """
        Test that abstractions have a sensible representation, with an internal
        operation that synthesizes the value of variables.
        """

        A = TypeOperator()
        f = Operator(type=A ** A)
        g = Operator(type=A ** A, body=lambda x: f(f(x)))
        h = Operator(type=(A ** A) ** A ** A)
        alg = Language(locals(), namespace=TEST)

        a = Source(type=A)
        self.assertIsomorphic(
            graph_auto(alg, h(g, a).primitive()),
            graph_manual(
                h=Step(TEST.h, input=["f2", "a"]),
                f2=Step(TEST.f, input="f1"),
                f1=Step(TEST.f, input="λ"),
                λ=Step(internal="h", input="a"),
                a=Step(),
            )
        )

    def test_complex_abstraction_as_parameter(self):
        """
        Same as before, but with a more complex abstraction that uses multiple
        variables.
        """

        A = TypeOperator()
        f = Operator(type=A ** A ** A)
        g = Operator(type=A ** A ** A, body=lambda x, y: f(y, f(x, y)))
        h = Operator(type=(A ** A ** A) ** A ** A ** A)
        alg = Language(locals(), namespace=TEST)

        a = Source(type=A)
        b = Source(type=A)
        self.assertIsomorphic(
            graph_auto(alg, h(g, a, b).primitive()),
            graph_manual(
                h=Step(TEST.h, input=["f2", "a", "b"]),
                f2=Step(TEST.f, input=["λ", "f1"]),
                f1=Step(TEST.f, input="λ"),
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

        A = TypeOperator()
        f = Operator(type=(A ** A) ** A)
        id = Operator(type=A ** A, body=lambda x: x)
        alg = Language(locals(), namespace=TEST)

        self.assertIsomorphic(
            graph_auto(alg, f(id).primitive()),
            graph_manual(
                f=Step(TEST.f, input="λ"),
                λ=Step(internal="f"),
            )
        )

    def test_abstraction_same_as_primitive(self):
        """
        Test that an abstraction of the form `f (λx.g x)` is not only
        conceptually the same as `f g`, but actually leads to the same graph.
        """

        A = TypeOperator()
        f = Operator(type=(A ** A) ** A)
        g = Operator(type=A ** A)
        h = Operator(type=A ** A, body=lambda x: g(x))
        alg = Language(locals(), namespace=TEST)

        self.assertIsomorphic(
            graph_auto(alg, f(h).primitive()),
            graph_auto(alg, f(g))
        )

    def test_cycle(self):
        """
        It is possible to get a cyclic transformation graph when passing
        multiple operations as parameter.
        """
        A = TypeOperator()
        f = Operator(type=(A ** A) ** (A ** A) ** (A ** A) ** A ** A)
        g = Operator(type=A ** A)
        h = Operator(type=A ** A)
        e = Operator(type=A ** A)
        alg = Language(locals(), namespace=TEST)

        a = Source(type=A)
        self.assertIsomorphic(
            graph_auto(alg, f(g, h, e, a)),
            graph_manual(
                a=Step(),
                f=Step(TEST.f, input=["a", "g", "h", "e"]),
                gλ=Step(internal="f", input=["a", "h", "e"]),
                hλ=Step(internal="f", input=["a", "g", "e"]),
                eλ=Step(internal="f", input=["a", "g", "h"]),
                g=Step(TEST.g, input="gλ"),
                h=Step(TEST.h, input="hλ"),
                e=Step(TEST.e, input="eλ"),
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
        A = TypeOperator()
        outer = Operator(type=(A ** A) ** A ** A)
        inner = Operator(type=(A ** A) ** A ** (A ** A))
        f = Operator(type=A ** A)
        alg = Language(locals(), namespace=TEST)

        a = Source(type=A)
        b = Source(type=A)
        self.assertIsomorphic(
            graph_auto(alg, outer(inner(f, a), b)),
            graph_manual(
                a=Step(),
                b=Step(),
                outer=Step(TEST.outer, input=["inner", "b"]),
                inner=Step(TEST.inner, input=["f", "a", "outerλ"]),
                f=Step(TEST.f, input="innerλ"),
                innerλ=Step(internal="inner", input=["a", "outerλ"]),
                outerλ=Step(internal="outer", input="b"),
            )
        )

    def test_function_abstraction_body(self):
        """
        The body of an abstraction may be a function.
        """
        A = TypeOperator()
        f = Operator(type=(A ** A ** A) ** A ** A)
        g = Operator(type=A ** A ** A)
        h = Operator(type=A ** A ** A, body=lambda x: g(x))
        alg = Language(locals(), namespace=TEST)

        a = Source(type=A)
        self.assertIsomorphic(
            graph_auto(alg, f(h, a).primitive()),
            graph_manual(
                a=Step(),
                g=Step(TEST.g, input="λ"),
                f=Step(TEST.f, input=["g", "a"]),
                λ=Step(internal="f", input="a"),
            )
        )

    def test_type_reuse(self):
        """
        Test that type nodes are reused when transformed to RDF.
        """
        A = TypeOperator("A")
        F = TypeOperator("F", params=1)
        G = TypeOperator("G", params=2)
        alg = Language(locals(), namespace=TEST)

        g = Graph()
        n1 = BNode()
        n2 = BNode()
        g.add((n1, RDFS.subClassOf, TEST.F))
        g.add((n1, RDF._1, TEST.A))
        g.add((n2, RDFS.subClassOf, TEST.G))
        g.add((n2, RDF._1, n1))
        g.add((n2, RDF._2, n1))
        self.assertIsomorphic(
            graph_auto(alg, G(F(A), F(A))),
            g
        )

    def test_type_variable_reuse(self):
        """
        Test that bound type variables are properly reused.
        """
        A = TypeOperator()
        F = TypeOperator(params=2)
        alg = Language(locals(), namespace=TEST)

        g = Graph()
        x, y = TypeVariable(), TypeVariable()
        x.bind(A())
        y.bind(A())
        n1 = BNode()
        g.add((n1, RDFS.subClassOf, TEST.F))
        g.add((n1, RDF._1, TEST.A))
        g.add((n1, RDF._2, TEST.A))
        self.assertIsomorphic(
            graph_auto(alg, F(x, y)),
            g
        )

    def test_taxonomy(self):
        # Test that subtype relations are encoded in the vocabulary
        A = TypeOperator()
        B = TypeOperator(supertype=A)
        C = TypeOperator(supertype=A)
        D = TypeOperator(supertype=C)
        lang = Language(locals(), namespace=TEST)

        actual = TransformationGraph(lang, minimal=True)
        actual.add_taxonomy()

        expected = Graph()
        expected.add((TEST.B, RDFS.subClassOf, lang.uri(A)))
        expected.add((TEST.C, RDFS.subClassOf, lang.uri(A)))
        expected.add((TEST.D, RDFS.subClassOf, lang.uri(C)))

        self.assertIsomorphic(expected, actual)

    def test_sources_typed(self):
        # Test that sources are properly typed. See issue #66.
        A, B = TypeOperator(), TypeOperator()
        f = Operator(type=A ** B)
        ℒ = Language(locals(), namespace=TEST)

        root = BNode()
        actual = TransformationGraph(ℒ,
            minimal=True, with_types=True, with_operators=True)
        actual.add_workflow(WorkflowDict(root,
            {BNode(): ("f (1: A)", [TEST.source])},
            {TEST.source}))

        expected = graph_manual(
            x=Step(type=TEST.A),
            f=Step(via=TEST.f, input="x", type=TEST.B),
        )

        self.assertIsomorphic(actual, expected)

    def test_order_of_subtype_application(self):
        # A repeat of the test in `test_type.py` to make sure that the order of
        # subtype application does not matter, but now on the expression level.
        A, B = TypeOperator(), TypeOperator()
        A1 = TypeOperator(supertype=A)
        f = Operator(type=lambda α: α ** α ** B)
        ℒ = Language(locals(), namespace=TEST)

        root = BNode()

        # 1
        actual1 = TransformationGraph(ℒ,
            minimal=True, with_types=True, with_operators=True)
        actual1.add_workflow(WorkflowDict(root, {
            TEST.sourceA: ("-: A", []),
            TEST.sourceA1: ("-: A1", []),
            TEST.app: ("f 1 2", [TEST.sourceA, TEST.sourceA1])
        }))

        # 2
        actual2 = TransformationGraph(ℒ,
            minimal=True, with_types=True, with_operators=True)
        actual2.add_workflow(WorkflowDict(root, {
            TEST.sourceA: ("-: A", []),
            TEST.sourceA1: ("-: A1", []),
            TEST.app: ("f 1 2", [TEST.sourceA1, TEST.sourceA])
        }))

        # Mismatch
        actual3 = TransformationGraph(ℒ,
            minimal=True, with_types=True, with_operators=True)
        self.assertRaises(ApplicationError, actual3.add_workflow, WorkflowDict(root, {
            TEST.sourceA: ("-: A", []),
            TEST.sourceB: ("-: B", []),
            TEST.app: ("f 1 2", [TEST.sourceA, TEST.sourceB])
        }))

    def test_reuse_sources(self):
        # Test that the same source can be reused.
        A, B = TypeOperator(), TypeOperator()
        A1 = TypeOperator(supertype=A)
        f = Operator(type=A1 ** A ** B)
        ℒ = Language(locals(), namespace=TEST)

        root = BNode()
        actual = TransformationGraph(ℒ,
            minimal=True, with_types=True, with_operators=True)
        actual.add_workflow(WorkflowDict(root, {
            TEST.step1: ("-: A1", []),
            TEST.step2: ("f 1 2", [TEST.step1, TEST.step1])
        }))

    def test_unify_type_when_attaching_source(self):
        # Expressions that have a variable type, but are incorporated in a
        # workflow such that their type gets fixed, should be unified; and this
        # should happen in time for the type to be reflected in the graph
        # representation. See issue #31.

        A = TypeOperator()
        f = Operator(type=lambda x: x ** x)
        ℒ = Language(locals(), namespace=TEST)

        actual = TransformationGraph(ℒ,
            minimal=True, with_types=True, with_operators=True)

        root = BNode()
        actual.add_workflow(WorkflowDict(root, {
            TEST.step1: ("-: A", []),
            TEST.step2: ("f 1", [TEST.step1])
        }))

        expected = graph_manual(
            source=Step(type=TEST.A),
            app=Step(TEST.f, input="source", type=TEST.A),
        )

        self.assertIsomorphic(actual, expected)

    def test_inter_tool_types(self):
        # Types should work between tools, not just inside tools. Say that we
        # have a function f : x ** x | x << Ratio and a tool T that implements
        # that function. We then write the algebra expression for T as
        # f(-: Ratio), but that should mean that `f` still produces a subtype of
        # Ratio if presented with a subtype of Ratio. See issue #72.

        A = TypeOperator()
        B = TypeOperator(supertype=A)
        f = Operator(type=lambda x: x ** x)
        ℒ = Language(locals(), namespace=TEST)

        actual = TransformationGraph(ℒ,
            minimal=True, with_types=True, with_operators=True)

        root = BNode()
        actual.add_workflow(WorkflowDict(root, {
            TEST.step1: ("-: B", []),
            TEST.step2: ("f(1: A)", [TEST.step1])
        }))

        expected = graph_manual(
            source=Step(type=TEST.B),
            app=Step(TEST.f, input="source", type=TEST.B),
        )

        self.assertIsomorphic(actual, expected)

    def test_within_tool_types(self):
        # Optionally, only the types at the entrance and exit of a tool will be
        # recorded. See issue #81

        A = TypeOperator()
        F = TypeOperator(params=1)
        f = Operator(type=lambda x: x ** F(x))
        lang = Language(locals(), namespace=TEST,
            canon={A, F(A), F(F(A)), F(F(F(A)))})

        expected = graph_manual(
            s1=Step(type=TEST.A),
            s2=Step(TEST.f, input="s1", type=None),
            s3=Step(TEST.f, input="s2", type=lang.uri(F(F(A)))),
            s4=Step(TEST.f, input="s3", type=lang.uri(F(F(F(A))))),
        )

        actual = TransformationGraph(lang,
            minimal=True, with_operators=True,
            with_types=True, with_intermediate_types=False)
        actual.add_taxonomy()

        expected += actual

        root = BNode()
        actual.add_workflow(WorkflowDict(root, {
            TEST.step1: ("f (f (-: A))", []),
            TEST.step2: ("f (1: F(F(_)))", [TEST.step1])
        }))

        self.assertIsomorphic(actual, expected)

    def test_disabling_of_output_passthrough(self):
        # Optionally, instead of passing the output of one tool directly to the
        # next tool (with a type that is potentially more specific than the
        # most general type the tool can work with), it is possible to disable
        # this mechanism and instead have every tool work with the most general
        # type. In the graph, to connect two tool applications, there is then a
        # Source (of the general type) that takes an expression output (of the
        # specific type) as input. See issue #81.

        A = TypeOperator('A')
        B = TypeOperator('B', supertype=A)
        TypeOperator('C', supertype=B)
        f = Operator(type=lambda x: x ** x [x <= A])
        lang = Language(locals(), namespace=TEST)

        expected = graph_manual(
            s1=Step(type=TEST.B),
            s2=Step(TEST.f, input="s1", type=TEST.B),
            s3=Step(type=TEST.A, input="s2"),
            s4=Step(TEST.f, input="s3", type=TEST.A),
        )

        actual = TransformationGraph(lang, passthrough=False,
            minimal=True, with_operators=True, with_types=True)
        root = BNode()
        actual.add_workflow(WorkflowDict(root, {
            TEST.tool1: ("f (1: B)", [TEST.src1]),
            TEST.tool2: ("f (1: A)", [TEST.tool1])
        }, {TEST.src1}))

        self.assertIsomorphic(actual, expected)

    def test_workflow_origins(self):
        A = TypeOperator('A')
        f = Operator(type=lambda x: x ** x)
        lang = Language(locals(), namespace=TEST)

        expected = graph_manual(
            s1=Step(type=TEST.A, origin=TEST.src),
            s2=Step(TEST.f, input="s1", type=TEST.A, origin=TEST.tool),
        )

        actual = TransformationGraph(lang, passthrough=False,
            minimal=True, with_operators=True, with_types=True,
            with_workflow_origin=True)
        root = BNode()
        actual.add_workflow(WorkflowDict(root, {
            TEST.tool: ("f (1: A)", [TEST.src]),
        }, {TEST.src}))

        self.assertIsomorphic(actual, expected)

    def test_shortcut_parsing(self):
        A = TypeOperator()
        F = TypeOperator(params=1)
        f = Operator(type=lambda x: x ** x)
        lang = Language(locals(), namespace=TEST, canon={A, F(A)})

        root = BNode()

        actual = TransformationGraph(lang)
        actual.add((root, TEST.type, Literal("F(A)")))
        actual.add((root, TEST.via, Literal("f")))
        actual.parse_shortcuts()

        expected = TransformationGraph(lang)
        expected.add((root, TA.type, lang.uri(F(A))))
        expected.add((root, TA.via, lang.uri(f)))

        self.assertIsomorphic(actual, expected)

    def test_taxonomy2(self):
        A = TypeOperator()
        B = TypeOperator(supertype=A)
        F = TypeOperator(params=2)
        lang = Language(scope=locals(), namespace=TEST, canon={A, F(A, A)})

        actual = TransformationGraph(lang, minimal=True)
        actual.add_taxonomy()
        expected = make_taxonomy(lang, {
            A: B,
            F(A, A): (F(A, B), F(B, A)),
            (F(B, A), F(A, B)): F(B, B)
        })
        self.assertIsomorphic(expected, actual)

    def test_complex_taxonomy(self):
        A = TypeOperator()
        B = TypeOperator(supertype=A)
        C = TypeOperator(supertype=A)
        F = TypeOperator(params=2)
        lang = Language(scope=locals(), namespace=TEST, canon={A, F(A, A)})

        actual = TransformationGraph(lang, minimal=True)
        actual.add_taxonomy()

        expected = make_taxonomy(lang, {
            A: (B, C),
            F(A, A): (F(A, B), F(A, C), F(B, A), F(C, A)),
            F(A, B): (F(B, B), F(C, B)),
            F(A, C): (F(B, C), F(C, C)),
            F(B, A): (F(B, B), F(B, C)),
            F(C, A): (F(C, B), F(C, C)),
        })

        self.assertIsomorphic(expected, actual)

    def test_taxonomy_with_subtypes_connected(self):
        # Make sure that, when expanding canonical types, only the subtypes are
        # included, and that they are connected to the Top type properly
        A = TypeOperator()
        B = TypeOperator(supertype=A)
        F = TypeOperator(params=2)
        lang = Language(scope=locals(), namespace=TEST, include_top=True,
            canon={A, F(A, B)})

        actual = TransformationGraph(lang, minimal=True)
        actual.add_taxonomy()

        expected = make_taxonomy(lang, {
            Top: (A, F(Top, Top)),
            A: B,
            F(Top, Top): (F(Top, B), F(A, Top)),
            F(Top, B): F(A, B),
            F(A, Top): (F(A, B), F(B, Top)),
            F(B, Top): F(B, B),
            F(A, B): F(B, B),
        })

        self.assertIsomorphic(expected, actual)

    def test_taxonomy_including_top_bottom(self):
        A = TypeOperator()
        B = TypeOperator(supertype=A)
        F = TypeOperator(params=2)
        lang = Language(scope=locals(), namespace=TEST,
            include_top=True, include_bottom=True, canon={A, F(A, A)})

        actual = TransformationGraph(lang, minimal=True)
        actual.add_taxonomy()
        expected = make_taxonomy(lang, {
            Top: (A, F(Top, Top)),
            A: B,
            B: Bottom,
            F(Top, Top):
                (F(A, Top), F(Top, A)),
            F(Top, A): (F(A, A), F(Top, B)),
            F(A, Top): (F(A, A), F(B, Top)),
            F(B, Top): (F(Bottom, Top), F(B, A)),
            F(Top, B): (F(Top, Bottom), F(A, B)),
            F(A, A): (F(A, B), F(B, A)),
            F(A, B): (F(B, B), F(A, Bottom)),
            F(B, A): (F(B, B), F(Bottom, A)),
            F(A, Bottom): F(B, Bottom),
            F(Bottom, A): F(Bottom, B),
            F(B, B): (F(B, Bottom), F(Bottom, B)),
            F(Top, Bottom): (F(A, Bottom)),
            F(Bottom, Top): (F(Bottom, A)),
            F(B, Bottom): F(Bottom, Bottom),
            F(Bottom, B): F(Bottom, Bottom),
            F(Bottom, Bottom): Bottom
        })
        self.assertIsomorphic(expected, actual)

    def test_taxonomy_including_only_top(self):
        A = TypeOperator()
        B = TypeOperator(supertype=A)
        F = TypeOperator(params=2)
        lang = Language(scope=locals(), namespace=TEST,
            include_top=True, canon={A, F(A, A)})

        actual = TransformationGraph(lang, minimal=True)
        actual.add_taxonomy()
        expected = make_taxonomy(lang, {
            Top: (A, F(Top, Top)),
            A: B,
            F(Top, Top): (F(A, Top), F(Top, A)),
            F(Top, A): (F(A, A), F(Top, B)),
            F(A, Top): (F(A, A), F(B, Top)),
            F(B, Top): F(B, A),
            F(Top, B): F(A, B),
            F(A, A): (F(A, B), F(B, A)),
            F(A, B): F(B, B),
            F(B, A): F(B, B),
        })
        self.assertIsomorphic(expected, actual)
