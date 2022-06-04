from __future__ import annotations

import unittest
from rdflib.term import BNode, Node, URIRef, Literal
from rdflib.namespace import RDF

from transformation_algebra.type import TypeOperator, Type, TypeAlias
from transformation_algebra.expr import Operator, Expr
from transformation_algebra.lang import Language
from transformation_algebra.graph import TransformationGraph, TA, TEST
from transformation_algebra.query import TransformationQuery


def make_query(lang: Language, obj: tuple[Operator | Type | list]
        ) -> TransformationQuery:
    """
    Convenience method for constructing a query.

    An object like this like this:

        (C, b2c, [(B, a2b, [A])])

    … gets turned into the following RDF graph:

        @prefix : <https://github.com/quangis/transformation-algebra#>.
        @prefix from: <https://github.com/quangis/transformation-algebra#from>.
        @prefix type: <https://example.com/#type>.
        @prefix via: <https://example.com/#operator>.
        [] a :Task; :output
            [ type: "C"; via: "b2c"; from:
                [ type: "B"; via: "a2b"; from:
                    [ type: "A" ]]].

    … which serves as the basis for a TransformationQuery.
    """
    ns = lang.namespace
    g = TransformationGraph(lang)
    root = BNode()
    g.add((root, RDF.type, TA.Task))

    def f(current: tuple[Operator | Type | list] | Type | Operator) -> Node:
        if isinstance(current, (Type, Operator)):
            return f((current,))
        else:
            node = BNode()
            for aspect in current:
                if isinstance(aspect, Type):
                    g.add((node, ns.type, Literal(str(aspect))))
                elif isinstance(aspect, Operator):
                    g.add((node, ns.via, Literal(str(aspect))))
                else:
                    assert isinstance(aspect, list)
                    for pre in aspect:
                        pnode = f(pre)
                        g.add((node, TA["from"], pnode))
            return node

    g.add((root, TA.output, f(obj)))
    g.parse_shortcuts()

    return TransformationQuery(lang, g)


def make_graph(lang: Language, workflows: dict[URIRef, Expr]) -> TransformationGraph:
    """
    Convenience method for constructing a graph containing workflows.
    """
    graph = TransformationGraph(lang)
    graph.add_vocabulary()
    for wfnode, content in workflows.items():
        e = graph.add_expr(content, wfnode)
        graph.add((wfnode, RDF.type, TA.Transformation))
        graph.add((wfnode, TA.output, e))
    return graph


class TestAlgebra(unittest.TestCase):

    def assertQuery(self, lang: Language,
            graph: TransformationGraph, q_obj: tuple[Type | Operator | list],
            results: set[Node] | None, **kwargs) -> None:

        query = make_query(lang, q_obj)
        self.assertEqual(
            results or set(),
            set(r.workflow for r in graph.query(query.sparql(**kwargs)))
        )

    def test_serial(self):
        # Ensure that we can query for types/operations that occur in sequence
        A, B, C, D = (TypeOperator() for _ in range(4))
        a2b = Operator(type=A ** B)
        b2c = Operator(type=B ** C)
        lang = Language(locals(), namespace=TEST)

        graph = make_graph(lang, {TEST.wf1: b2c(a2b(~A))})

        self.assertQuery(lang, graph, (b2c, C, [a2b, B, [A]]),
            results={TEST.wf1})
        self.assertQuery(lang, graph, (C, [(B, [A])]),
            results={TEST.wf1})
        self.assertQuery(lang, graph, (b2c, [a2b]),
            results={TEST.wf1})
        self.assertQuery(lang, graph, (a2b, C, [(b2c, B, [A])]),
            results=None)
        self.assertQuery(lang, graph, (b2c, B, [(a2b, C, [A])]),
            results=None)

    def test_serial_skip(self):
        # Ensure that we can query for types/operations that not only occur in
        # sequence, but where a subsequent type may directly result from the
        # preceding operation
        A, B, C, D = (TypeOperator() for _ in range(4))
        a2b = Operator(type=A ** B)
        b2c = Operator(type=B ** C)
        c2d = Operator(type=C ** D)
        alg = Language(locals(), namespace=TEST)

        graph = make_graph(alg, {TEST.wf1: c2d(b2c(a2b(~A)))})

        self.assertQuery(alg, graph, (D, [A]),
            results={TEST.wf1})
        self.assertQuery(alg, graph, (c2d, [a2b]),
            results={TEST.wf1})
        self.assertQuery(alg, graph, (c2d, [(b2c, [a2b])]),
            results={TEST.wf1})
        self.assertQuery(alg, graph, (D, [(b2c, [A])]),
            results={TEST.wf1})
        self.assertQuery(alg, graph, (D, [(b2c, A)]),
            results=None)
        self.assertQuery(alg, graph, (D, [(B, [(a2b, [A])])]),
            results={TEST.wf1})

    def test_parallel(self):
        A, B, C, D = (TypeOperator() for _ in range(4))
        a2b = Operator(type=A ** B)
        b2c = Operator(type=B ** C)
        c2d = Operator(type=C ** D)
        bc2d = Operator(type=B ** C ** D)
        alg = Language(locals(), namespace=TEST)

        graph = make_graph(alg, {
            TEST.wf1: bc2d(~B, ~C),
            TEST.wf2: bc2d(a2b(~A), b2c(~B))
        })

        self.assertQuery(alg, graph, (D, bc2d, [B, C]),
            results={TEST.wf1, TEST.wf2})
        self.assertQuery(alg, graph, (D, bc2d, [B, C, a2b, b2c]),
            results={TEST.wf2})
        self.assertQuery(alg, graph, (D, bc2d, [(B, [a2b]), (C, [b2c])]),
            results={TEST.wf2})
        self.assertQuery(alg, graph, (D, bc2d, [A, B]),
            results={TEST.wf2})

    # Commented out because choice support will be reimplemented later
    # def test_choice(self):

    #     A, B, C, D = (TypeOperator() for _ in range(4))
    #     a2b = Operator(type=A ** B)
    #     b2c = Operator(type=B ** C)
    #     b2c2 = Operator(type=B ** C)
    #     bc2d = Operator(type=B ** C ** D)
    #     lang = Language(locals(), namespace=TEST)

    #     graph = make_graph(lang, {
    #         TEST.wf1: bc2d(a2b(~A), b2c(~B)),
    #         TEST.wf2: bc2d(~B, b2c(a2b(~A)))
    #     })

    #     self.assertQuery(lang, graph, STEPS(D, bc2d, OR(A, D)),
    #         results=None)
    #     self.assertQuery(lang, graph, [D, bc2d, OR(A, B)],
    #         results={TEST.wf1, TEST.wf2})
    #     self.assertQuery(lang, graph, [D, bc2d, OR(B, C)],
    #         results={TEST.wf1, TEST.wf2})

    #     self.assertQuery(lang, graph, STEPS(D, bc2d, OR(STEPS(B, a2b), STEPS(C, a2b))),
    #         results={TEST.wf1})

    #     # Choice between operations in non-last place
    #     self.assertQuery(lang, graph, STEPS(D, bc2d, OR(b2c, b2c2), B),
    #         results={TEST.wf1, TEST.wf2})

    def test_sequenced_skips(self):

        A, B, C = (TypeOperator(x) for x in "ABC")
        a2b, b2c = Operator(type=A ** B), Operator(type=B ** C)
        lang = Language(locals(), namespace=TEST)

        graph = make_graph(lang, {
            TEST.wf1: b2c(a2b(~A)),
            TEST.wf2: ~A
        })

        # Test that a query for direct output really only captures that
        self.assertQuery(lang, graph, (C, a2b), results=set())

        # Test that a query for indirect output also captures direct output
        self.assertQuery(lang, graph, (C, [a2b]), results={TEST.wf1})
        self.assertQuery(lang, graph, (C, [b2c]), results={TEST.wf1})

        # Test that a query that skips the result type may still capture it
        self.assertQuery(lang, graph, ([C],), results={TEST.wf1})
        self.assertQuery(lang, graph, ([A],), results={TEST.wf1, TEST.wf2})

    # def test_multiple_usage_of_units(self):
    #     # The same unit may be used multiple times, so simply assigning a
    #     # variable to a unit will lead to problems: one unit may have multiple
    #     # variables, and multiple variables may refer to the same unit. Adding
    #     # skips as a property of units poses a problem for the same reason: if
    #     # a unit is used once in a series WITH a skip, all subsequent uses of
    #     # that unit would also have that skip.

    #     A, B, C, D = (TypeOperator() for _ in range(4))
    #     a2b1 = Operator(type=A ** B)
    #     a2b2 = Operator(type=A ** B)
    #     a2b = OR(a2b1, a2b2)
    #     b2c = Operator(type=B ** C)
    #     b2d = Operator(type=B ** D)
    #     cd2a = Operator(type=C ** D ** A)
    #     lang = Language(locals(), namespace=TEST)

    #     graph = make_graph(lang, {
    #         TEST.wf1: cd2a(b2c(a2b1(~A)), b2d(a2b2(~A)))
    #     })

    #     self.assertQuery(lang, graph,
    #         STEPS(A, AND(
    #             [C, a2b],
    #             STEPS(C, a2b)
    #         )),
    #         results={}
    #     )
    #     self.assertQuery(lang, graph,
    #         [A, AND(
    #             [D, a2b],
    #             [C, a2b]
    #         )],
    #         results={TEST.wf1}
    #     )

    def test_query_specific_aspects(self):
        # See issue #46
        A, B, C = (TypeOperator() for _ in range(3))
        a2b = Operator(type=A ** B)
        b2c = Operator(type=B ** C)
        lang = Language(locals(), namespace=TEST)

        graph = make_graph(lang, {
            TEST.wf1: a2b(~A),
            TEST.wf2: b2c(a2b(~A))
        })

        self.assertQuery(lang, graph,
            (B,), by_io=False, by_chronology=False,
            results={TEST.wf1, TEST.wf2}
        )
        self.assertQuery(lang, graph,
            (B,), by_io=True, by_chronology=False,
            results={TEST.wf1}
        )
        self.assertQuery(lang, graph,
            (C, [(A, [B])]), by_io=True, by_chronology=False,
            results={TEST.wf2}
        )
        self.assertQuery(lang, graph,
            (C, [(A, [B])]), by_io=True, by_chronology=True,
            results={}
        )

    def test_that_supertypes_are_captured(self):
        # Test that using a supertype in a query would still return a workflow
        # that uses a subtype
        X = TypeOperator()
        Y = TypeOperator(supertype=X)
        F = TypeOperator(params=1)
        FX, FY = TypeAlias(F(X)), TypeAlias(F(Y))
        lang = Language(locals(), namespace=TEST)

        graph = make_graph(lang, {
            TEST.x: ~X,
            TEST.y: ~Y,
            TEST.fx: ~F(X),
            TEST.fy: ~F(Y)
        })

        # graph.serialize("test.ttl", format="ttl")
        # print(Query(lang.namespace, X).sparql())

        self.assertQuery(lang, graph, X, results={TEST.x, TEST.y})
        self.assertQuery(lang, graph, Y, results={TEST.y})
        self.assertQuery(lang, graph, F(X), results={TEST.fx, TEST.fy})
        self.assertQuery(lang, graph, F(Y), results={TEST.fy})

    def test_that_repeated_nodes_are_handled_correctly(self):
        A = TypeOperator('A')
        f = Operator('f', type=A ** A)
        lang = Language(locals(), namespace=TEST)

        graph = make_graph(lang, {
            TEST.wf1: ~A,
            TEST.wf2: f(~A),
        })
        self.assertQuery(lang, graph, (A,), results={TEST.wf1, TEST.wf2})
        self.assertQuery(lang, graph, (A, [A]), results={TEST.wf2})

    # # def test_that_sources_with_nonnormalized_type_get_type_in_graph(self):
    # #     # There was an issue where the type of a source would not be saved in
    # #     # the graphs if it was not yet normalized, causing a subtle bug.
    # #     # I commented the test out for now since figuring out a MWE was
    # #     # taking too much time; the issue was with not normalizing the source
    # #     T = TypeOperator()
    # #     S = TypeOperator()
    # #     F = Operator(type=T ** S)
    # #     lang = Language(locals(), namespace=TEST)
    # #     graph = TransformationGraph(lang)
    # #     graph.add_vocabulary()
    # #     graph.add_workflow(TEST.wf1, {TEST.x: ("F(1)", [TEST.y])}, {TEST.y})
    # #     graph.serialize("test.ttl")
    # #     self.assertQuery(lang, graph, [S, T], results={TEST.wf1})

    # # @unittest.skip("Obsoleted since it is no longer possible to skip at "
    # #         "the beginning of a flow.")
    # # def test_distribution_of_skips(self):
    # #     # See issue #61:
    # #     # Test that [A, ..., AND(f, g)] is not interpreted the same as [A,
    # #     # AND([..., f], [..., g])]: the first says that, before A, there
    # #     # should be a particular node that takes the outputs of both f and g
    # #     # as input; the second simply says that f and g occur sometime before
    # #     # A.

    # #     A, B, C, D = (TypeOperator() for _ in range(4))
    # #     a2a = Operator(type=A ** A)
    # #     a2b = Operator(type=A ** B)
    # #     ab2c = Operator(type=A ** B ** C)
    # #     c2d = Operator(type=C ** D)
    # #     lang = Language(locals())

    # #     graph = make_graph(lang,
    # #             direct=c2d(ab2c(a2a(~A), a2b(~A))),
    # #             indirect=c2d(ab2c(~A, a2b(a2a(~A)))))

    # #     self.assertQuery(lang, graph, [D, ..., AND(a2a, a2b)],
    # #         results={TEST.direct})
    # #     self.assertQuery(lang, graph, [D, AND([..., a2a], [..., a2b])],
    # #         results={TEST.direct, TEST.indirect})

    # # @unittest.skip("unsupported flows")
    # # def test_unsupported(self):

    # #     graph = make_graph(lang,
    # #         wf1=f2(f(~A), g(~B)),
    # #         wf2=f2(~B, g(f(~A)))
    # #     )

    # #     # Choice between sequences outside of last place
    # #     self.assertQuery(lang, graph, [D, f2, OR([g, f], [m, n]), A],
    # #         results={TEST.wf1})

    def test_sensible_order(self):
        # If you have a transformation graph query that goes:
        #
        #   input  →  output
        #      ↘ inter ↗
        #
        # Then, depending on Python's whims, either the intermediate or input
        # node may be first to be visited after the output node. In case we
        # visit the input node first, we have a problem, since we must declare
        # that it is preceded by a node that has not yet been assigned any
        # properties (variable name, types etcetera). And if we choose the
        # input node before the inter node, then the inter node might be
        # considered twice. We must deal with both these issues.
        lang = Language(locals(), TEST)

        # Repeat a couple of times to make it more likely that the bug, if
        # present, is caught
        for _ in range(10):
            graph = TransformationGraph(lang)
            root = BNode()
            A, B, C = BNode(), BNode(), BNode()
            graph.add((root, RDF.type, TA.Task))
            graph.add((root, TA.output, A))
            graph.add((A, TA["from"], B))
            graph.add((A, TA["from"], C))
            graph.add((B, TA["from"], C))
            result = list(TransformationQuery(lang, graph).chronology())
            self.assertTrue(
                (result == ['?workflow :output ?_0.', '?_1 :to* ?_0.',
                 '?_2 :to* ?_0.', '?_2 :to* ?_1.']) or
                (result == ['?workflow :output ?_0.', '?_1 :to* ?_0.',
                 '?_2 :to* ?_1.', '?_2 :to* ?_0.'])
            )

    def test_cycles(self):
        # The code that makes sure we pass `test_sensible_order` must not
        # introduce infinite loops when we encounter a cycle:
        #
        #   input  ←  output
        #      ↘ inter ↗
        lang = Language(locals(), TEST)
        graph = TransformationGraph(lang)

        graph = TransformationGraph(lang)
        root = BNode()
        A = BNode()
        graph.add((root, RDF.type, TA.Task))
        graph.add((root, TA.output, A))
        graph.add((A, TA["from"], A))
        query = TransformationQuery(lang, graph)
        self.assertRaises(Exception, TransformationQuery.chronology, query)

        graph = TransformationGraph(lang)
        root = BNode()
        A, B = BNode(), BNode()
        graph.add((root, RDF.type, TA.Task))
        graph.add((root, TA.output, A))
        graph.add((A, TA["from"], B))
        graph.add((B, TA["from"], A))
        query = TransformationQuery(lang, graph)
        self.assertRaises(Exception, TransformationQuery.chronology, query)

        graph = TransformationGraph(lang)
        root = BNode()
        A, B, C = BNode(), BNode(), BNode()
        graph.add((root, RDF.type, TA.Task))
        graph.add((root, TA.output, A))
        graph.add((A, TA["from"], B))
        graph.add((B, TA["from"], C))
        graph.add((C, TA["from"], A))
        query = TransformationQuery(lang, graph)
        self.assertRaises(Exception, TransformationQuery.chronology, query)

        graph = TransformationGraph(lang)
        root = BNode()
        A, B, C, D = BNode(), BNode(), BNode(), BNode()
        graph.add((root, RDF.type, TA.Task))
        graph.add((root, TA.output, D))
        graph.add((D, TA["from"], A))
        graph.add((A, TA["from"], B))
        graph.add((B, TA["from"], C))
        graph.add((C, TA["from"], A))
        query = TransformationQuery(lang, graph)
        self.assertRaises(Exception, TransformationQuery.chronology, query)


if __name__ == '__main__':
    unittest.main()
