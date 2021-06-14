from transformation_algebra.type import Type
from transformation_algebra.expr import Data, Operation
from transformation_algebra.rdf import TransformationAlgebraRDF, dot, Chain

from rdflib import Namespace, Graph

A, B, C = Type.declare('A'), Type.declare('B'), Type.declare('C')
data = Data(A)
ab = Operation(A ** B)
bc = Operation(B ** C)
compose = Operation(
    lambda α, β, γ: (β ** γ) ** (α ** β) ** (α ** γ),
    derived=lambda f, g, x: f(g(x))
)

algebra = TransformationAlgebraRDF(
    "cct", Namespace("https://github.com/quangis/cct/CCT.rdf#")
)
algebra.add(**globals())

g = Graph()
output_node = algebra.parse_rdf(g, "compose bc ab data")
q = Chain(A, None, [Chain(A), Chain(C)]).to_sparql(algebra)
print(q)
# result = g.query(q)

# print(dot(g))
