from __future__ import annotations

import unittest
from rdflib.term import Node, URIRef
from rdflib.namespace import RDF

from transformation_algebra.flow import FlowShorthand
from transformation_algebra.type import TypeOperator, Type
from transformation_algebra.expr import Operator, Expr
from transformation_algebra.lang import Language
from transformation_algebra.graph import TransformationGraph, TA, TEST
from transformation_algebra.query import Query, OR, AND


def make_graph(lang: Language,
        workflows: dict[URIRef, Expr | dict[Expr, list[Expr | Node]]]
        ) -> TransformationGraph:
    """
    Convenience method for constructing a graph containing workflows.
    """
    graph = TransformationGraph(lang)
    graph.add_vocabulary()
    for wfnode, content in workflows.items():
        if isinstance(content, Expr):
            e = graph.add_expr(content, wfnode)
            graph.add((wfnode, RDF.type, TA.Transformation))
            graph.add((wfnode, TA.output, e))
        else:
            graph.add_workflow(wfnode, content)
    return graph


class TestAlgebra(unittest.TestCase):

    def assertQuery(self, lang: Language,
            graph: TransformationGraph,
            query: Query | FlowShorthand[Type | Operator],
            results: set[Node] | None) -> None:

        if isinstance(query, Query):
            query1 = query
        else:
            query1 = Query(lang, query)

        self.assertEqual(
            results or set(),
            set(r.workflow for r in graph.query(query1.sparql()))
        )

    def test_serial(self):

        A, B, C, D = (TypeOperator() for _ in range(4))
        a2b = Operator(type=A ** B)
        b2c = Operator(type=B ** C)
        alg = Language(locals(), namespace=TEST)

        graph = make_graph(alg, {TEST.wf1: b2c(a2b(~A))})

        self.assertQuery(alg, graph, (C, b2c, B, a2b, A),
            results={TEST.wf1})
        self.assertQuery(alg, graph, (C, B, A),
            results={TEST.wf1})
        self.assertQuery(alg, graph, (b2c, a2b),
            results={TEST.wf1})
        self.assertQuery(alg, graph, (C, a2b, B, b2c, A),
            results=None)
        self.assertQuery(alg, graph, (B, b2c, C, a2b, A),
            results=None)

    def test_serial_skip(self):
        A, B, C, D = (TypeOperator() for _ in range(4))
        a2b = Operator(type=A ** B)
        b2c = Operator(type=B ** C)
        c2d = Operator(type=C ** D)
        alg = Language(locals(), namespace=TEST)

        graph = make_graph(alg, {TEST.wf1: c2d(b2c(a2b(~A)))})

        self.assertQuery(alg, graph, [D, A],
            results={TEST.wf1})
        self.assertQuery(alg, graph, [c2d, a2b],
            results={TEST.wf1})
        self.assertQuery(alg, graph, (c2d, b2c, a2b),
            results={TEST.wf1})
        self.assertQuery(alg, graph, [D, b2c, A],
            results={TEST.wf1})
        self.assertQuery(alg, graph, (D, b2c, A),
            results=None)
        self.assertQuery(alg, graph, [D, B, a2b, A],
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

        self.assertQuery(alg, graph, [D, bc2d, AND(B, C)],
            results={TEST.wf1, TEST.wf2})
        self.assertQuery(alg, graph, [D, bc2d, AND(B, C, a2b, b2c)],
            results={TEST.wf2})
        self.assertQuery(alg, graph, [D, bc2d, AND((B, a2b), (C, b2c))],
            results={TEST.wf2})
        self.assertQuery(alg, graph, [D, bc2d, AND(A, B)],
            results={TEST.wf2})
        self.assertQuery(alg, graph, (D, bc2d, AND(A, B)),
            results=None)

    def test_choice(self):

        A, B, C, D = (TypeOperator() for _ in range(4))
        a2b = Operator(type=A ** B)
        b2c = Operator(type=B ** C)
        b2c2 = Operator(type=B ** C)
        bc2d = Operator(type=B ** C ** D)
        lang = Language(locals(), namespace=TEST)

        graph = make_graph(lang, {
            TEST.wf1: bc2d(a2b(~A), b2c(~B)),
            TEST.wf2: bc2d(~B, b2c(a2b(~A)))
        })

        self.assertQuery(lang, graph, (D, bc2d, OR(A, D)),
            results=None)
        self.assertQuery(lang, graph, [D, bc2d, OR(A, B)],
            results={TEST.wf1, TEST.wf2})
        self.assertQuery(lang, graph, [D, bc2d, OR(B, C)],
            results={TEST.wf1, TEST.wf2})

        self.assertQuery(lang, graph, (D, bc2d, OR((B, a2b), (C, a2b))),
            results={TEST.wf1})

        # Choice between operations in non-last place
        self.assertQuery(lang, graph, (D, bc2d, OR(b2c, b2c2), B),
            results={TEST.wf1, TEST.wf2})

    def test_sequenced_skips(self):

        A, B, C = (TypeOperator() for _ in range(3))
        a2b, b2c = Operator(type=A ** B), Operator(type=B ** C)
        lang = Language(locals(), namespace=TEST)

        graph = make_graph(lang, {
            TEST.e: b2c(a2b(~A)),
            TEST.e2: ~A
        })

        # Test that a query for direct output really only captures that
        self.assertQuery(lang, graph, (C, a2b), results=set())

        # Test that a query for indirect output also captures direct output
        self.assertQuery(lang, graph, [C, a2b], results={TEST.e})
        self.assertQuery(lang, graph, [C, b2c], results={TEST.e})

        # Test that a query that skips the result type may still capture it
        # self.assertQuery(lang, graph, [..., C], results={TEST.e})
        # self.assertQuery(lang, graph, [..., A], results={TEST.e, TEST.e2})

    def test_multiple_usage_of_units(self):
        # The same unit may be used multiple times, so simply assigning a
        # variable to a unit will lead to problems: one unit may have multiple
        # variables, and multiple variables may refer to the same unit. Adding
        # skips as a property of units poses a problem for the same reason: if
        # a unit is used once in a series WITH a skip, all subsequent uses of
        # that unit would also have that skip.

        A, B, C, D = (TypeOperator() for _ in range(4))
        a2b1 = Operator(type=A ** B)
        a2b2 = Operator(type=A ** B)
        a2b = OR(a2b1, a2b2)
        b2c = Operator(type=B ** C)
        b2d = Operator(type=B ** D)
        cd2a = Operator(type=C ** D ** A)
        lang = Language(locals(), namespace=TEST)

        graph = make_graph(lang, {
            TEST.wf1: cd2a(b2c(a2b1(~A)), b2d(a2b2(~A)))
        })

        self.assertQuery(lang, graph,
            (A, AND(
                [C, a2b],
                (C, a2b)
            )),
            results={}
        )
        self.assertQuery(lang, graph,
            [A, AND(
                [D, a2b],
                [C, a2b]
            )],
            results={TEST.wf1}
        )

    def test_that_supertypes_are_captured(self):
        # Test that using a supertype in a query would still return a workflow
        # that uses a subtype
        X = TypeOperator()
        Y = TypeOperator(supertype=X)
        F = TypeOperator(params=1)
        FX, FY = F(X), F(Y)
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

    # @unittest.skip("Obsoleted since it is no longer possible to skip at "
    #         "the beginning of a flow.")
    # def test_distribution_of_skips(self):
    #     # See issue #61:
    #     # Test that [A, ..., AND(f, g)] is not interpreted the same as [A,
    #     # AND([..., f], [..., g])]: the first says that, before A, there
    #     # should be a particular node that takes the outputs of both f and g
    #     # as input; the second simply says that f and g occur sometime before
    #     # A.

    #     A, B, C, D = (TypeOperator() for _ in range(4))
    #     a2a = Operator(type=A ** A)
    #     a2b = Operator(type=A ** B)
    #     ab2c = Operator(type=A ** B ** C)
    #     c2d = Operator(type=C ** D)
    #     lang = Language(locals())

    #     graph = make_graph(lang,
    #             direct=c2d(ab2c(a2a(~A), a2b(~A))),
    #             indirect=c2d(ab2c(~A, a2b(a2a(~A)))))

    #     self.assertQuery(lang, graph, [D, ..., AND(a2a, a2b)],
    #         results={TEST.direct})
    #     self.assertQuery(lang, graph, [D, AND([..., a2a], [..., a2b])],
    #         results={TEST.direct, TEST.indirect})

    # @unittest.skip("unsupported flows")
    # def test_unsupported(self):

    #     graph = make_graph(lang,
    #         wf1=f2(f(~A), g(~B)),
    #         wf2=f2(~B, g(f(~A)))
    #     )

    #     # Choice between sequences outside of last place
    #     self.assertQuery(lang, graph, [D, f2, OR([g, f], [m, n]), A],
    #         results={TEST.wf1})


if __name__ == '__main__':
    unittest.main()
