from transformation_algebra.type import Type
from transformation_algebra.expr import Data, Operation
from transformation_algebra.rdf import TransformationRDF, dot

from rdflib import Namespace, Graph

A, B, C = Type.declare('A'), Type.declare('B'), Type.declare('C')
algebra = TransformationRDF(
    "cct", Namespace("https://github.com/quangis/cct/CCT.rdf#"),
    a=Data(A),
    ab=Operation(A ** B),
    bc=Operation(B ** C)
)
g = Graph()
output_node = algebra.parse_rdf(g, "bc (ab a)")
result = g.query(
    """
    SELECT ?type1 ?type2
    WHERE {
      ?data1 (^ta:input/ta:output)+ ?data2.
      ?data1 ta:type ?type1.
      ?data2 ta:type ?type2.
    }
    """)

for data1, data2 in result:
    print(f"{data1} -> {data2}")

#print(dot(g))
