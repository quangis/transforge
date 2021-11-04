import unittest

from rdflib.namespace import Namespace, RDFS, RDF

from transformation_algebra.type import TypeOperator
from transformation_algebra.expr import Operator
from transformation_algebra.lang import Language
from transformation_algebra.graph import TransformationGraph, LanguageNamespace, TA
from transformation_algebra.query import TransformationQuery

from rdflib.tools.rdf2dot import rdf2dot

TEST = Namespace("TEST#")


class TestAlgebra(unittest.TestCase):

    def test_basic(self):
        A = TypeOperator()
        B = TypeOperator()
        C = TypeOperator()
        f = Operator(type=A ** B)
        g = Operator(type=B ** C)
        alg = Language(locals())
        ALG = LanguageNamespace("ALG#", alg)

        wf = TEST.workflow
        graph = TransformationGraph(alg, ALG)
        e = graph.add_expr(g(f(~A)), wf)

        graph.add((wf, RDF.type, TA.Transformation))
        graph.add((wf, TA.result, e))
        graph.add((wf, RDFS.comment, TEST.comment))

        query = TransformationQuery(C, g, B, f, A, namespace=ALG)
        result = list(graph.query(query.sparql()))

        with open('graph.dot', 'w') as handle:
            rdf2dot(graph, handle)

        with open('graph.rq', 'w') as handle:
            handle.write(query.sparql())

        self.assertEqual(len(result), 1)


if __name__ == '__main__':
    unittest.main()
