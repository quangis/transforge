from __future__ import annotations

import unittest
from rdflib.term import Node
from rdflib.namespace import Namespace, RDF

from transformation_algebra.type import TypeOperator
from transformation_algebra.expr import Operator, Expr
from transformation_algebra.lang import Language
from transformation_algebra.graph import TransformationGraph, \
    LanguageNamespace, TA
from transformation_algebra.query import TransformationQuery, \
    NestedFlow, ANY, ALL

TEST = Namespace("https://example.com/#")

A = TypeOperator()
B = TypeOperator()
C = TypeOperator()
D = TypeOperator()
f = Operator(type=A ** B)
g = Operator(type=B ** C)
h = Operator(type=C ** D)
m = Operator(type=B ** C)
n = Operator(type=A ** B)
f2 = Operator(type=B ** C ** D)
alg = Language(locals())


def make_graph(**workflows: Expr | dict[Expr, list[Expr | Node]]
        ) -> TransformationGraph:
    """
    Convenience method for constructing a graph containing workflows.
    """
    graph = TransformationGraph(alg, TEST)
    for name, content in workflows.items():
        wf = TEST[name]
        if isinstance(content, Expr):
            e = graph.add_expr(content, wf)
            graph.add((wf, RDF.type, TA.Transformation))
            graph.add((wf, TA.result, e))
        else:
            graph.add_workflow(wf, content)
    return graph


class TestAlgebra(unittest.TestCase):

    def assertQuery(self, graph: TransformationGraph,
            query: TransformationQuery | NestedFlow,
            results: set[Node] | None) -> None:

        if isinstance(query, TransformationQuery):
            query1 = query
        else:
            query1 = TransformationQuery(query, TEST)

        self.assertEqual(
            results or set(),
            set(r.workflow for r in graph.query(query1.sparql()))
        )

    def test_serial(self):
        graph = make_graph(wf1=g(f(~A)))

        self.assertQuery(graph, [C, g, B, f, A],
            results={TEST.wf1})
        self.assertQuery(graph, [C, B, A],
            results={TEST.wf1})
        self.assertQuery(graph, [g, f],
            results={TEST.wf1})
        self.assertQuery(graph, [C, f, B, g, A],
            results=None)
        self.assertQuery(graph, [B, g, C, f, A],
            results=None)

    def test_serial_skip(self):
        graph = make_graph(wf1=h(g(f(~A))))

        self.assertQuery(graph, [D, ..., A],
            results={TEST.wf1})
        self.assertQuery(graph, [h, ..., f],
            results={TEST.wf1})
        self.assertQuery(graph, [h, g, f],
            results={TEST.wf1})
        self.assertQuery(graph, [D, ..., g, ..., A],
            results={TEST.wf1})
        self.assertQuery(graph, [D, g, A],
            results=None)
        self.assertQuery(graph, [D, ..., B, ..., f, ..., A],
            results={TEST.wf1})

    def test_parallel(self):
        graph = make_graph(wf1=f2(~B, ~C), wf2=f2(f(~A), g(~B)))

        self.assertQuery(graph, [D, f2, ALL(B, C)],
            results={TEST.wf1, TEST.wf2})
        self.assertQuery(graph, [D, f2, ALL([..., B], [..., C])],
            results={TEST.wf1, TEST.wf2})
        self.assertQuery(graph, [D, f2, ALL(B, C, f, g)],
            results={TEST.wf2})
        self.assertQuery(graph, [D, f2, ALL([B, f], [C, g])],
            results={TEST.wf2})
        self.assertQuery(graph, [D, f2, ALL([..., A], [..., B])],
            results={TEST.wf2})
        self.assertQuery(graph, [D, f2, ALL(A, B)],
            results=None)

    def test_choice(self):
        graph = make_graph(
            wf1=f2(f(~A), g(~B)),
            wf2=f2(~B, g(f(~A)))
        )

        self.assertQuery(graph, [D, f2, ANY(A, D)],
            results=None)
        self.assertQuery(graph, [D, f2, ANY(A, B)],
            results={TEST.wf1, TEST.wf2})
        self.assertQuery(graph, [D, f2, ANY(B, C)],
            results={TEST.wf1, TEST.wf2})
        self.assertQuery(graph, [D, f2, ANY([B, f], [C, f])],
            results={TEST.wf1})

        # Choice between operations in non-last place
        self.assertQuery(graph, [D, f2, ANY(g, m), B],
            results={TEST.wf1, TEST.wf2})

        # Choice between sequences in non-last place
        self.assertQuery(graph, [D, f2, ANY([g, f], [m, n]), A],
            results={TEST.wf1})


if __name__ == '__main__':
    unittest.main()
