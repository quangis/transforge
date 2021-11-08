from __future__ import annotations

import unittest
from rdflib.term import Node
from rdflib.namespace import Namespace, RDF

from transformation_algebra.type import TypeOperator
from transformation_algebra.expr import Operator, Expr
from transformation_algebra.lang import Language
from transformation_algebra.graph import TransformationGraph, \
    LanguageNamespace, TA
from transformation_algebra.query import TransformationQuery, NestedNotation

TEST = Namespace("https://example.com/#")

A = TypeOperator()
B = TypeOperator()
C = TypeOperator()
D = TypeOperator()
f = Operator(type=A ** B)
g = Operator(type=B ** C)
h = Operator(type=C ** D)
alg = Language(locals())
ALG = LanguageNamespace("ALG#", alg)


def make_graph(**workflows: Expr | dict[Expr, list[Expr | Node]]
        ) -> TransformationGraph:
    """
    Convenience method for constructing a graph containing workflows.
    """
    graph = TransformationGraph(alg, ALG)
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
            query: TransformationQuery | NestedNotation,
            results: set[Node] | None) -> None:

        if isinstance(query, TransformationQuery):
            query1 = query
        else:
            query1 = TransformationQuery(query, ALG)

        self.assertEqual(
            results or set(),
            set(r.workflow for r in graph.query(query1.sparql()))
        )

    def test_basic(self):
        graph = make_graph(wf1=g(f(~A)))

        self.assertQuery(graph, (C, g, B, f, A),
            results={TEST.wf1})
        self.assertQuery(graph, (C, B, A),
            results={TEST.wf1})
        self.assertQuery(graph, (g, f),
            results={TEST.wf1})
        self.assertQuery(graph, (C, f, B, g, A),
            results=None)
        self.assertQuery(graph, (B, g, C, f, A),
            results=None)

    def test_skip(self):
        graph = make_graph(wf1=h(g(f(~A))))

        self.assertQuery(graph, (D, ..., A),
            results={TEST.wf1})
        self.assertQuery(graph, (h, ..., f),
            results={TEST.wf1})
        self.assertQuery(graph, (h, g, f),
            results={TEST.wf1})
        self.assertQuery(graph, (D, ..., g, ..., A),
            results={TEST.wf1})
        self.assertQuery(graph, (D, g, A),
            results=None)
        self.assertQuery(graph, (D, ..., B, ..., f, ..., A),
            results={TEST.wf1})


if __name__ == '__main__':
    unittest.main()
