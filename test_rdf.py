from transformation_algebra.type import Type
from transformation_algebra.expr import Data, Operation
from transformation_algebra.rdf import TransformationRDF, dot

from rdflib import Namespace, Graph

Int = Type.declare('Int')
add = Operation(Int ** Int ** Int, name='add')
one = Data(Int)
algebra = TransformationRDF(
    "cct", Namespace("https://github.com/quangis/cct/CCT.rdf#"),
    one=one,
    add1=Operation(
        Int ** Int,
        derived=lambda x: add(x, one)
    ),
    compose=Operation(
        lambda α, β, γ: (β ** γ) ** (α ** β) ** (α ** γ),
        derived=lambda f, g, x: f(g(x))
    )
)
g = Graph()
output_node = algebra.parse_rdf(g, "compose add1 add1 one")

print(dot(g))
