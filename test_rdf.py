from transformation_algebra.type import Type
from transformation_algebra.expr import Data, Operation
from transformation_algebra.rdf import TransformationRDF

from rdflib import Namespace, Graph
from rdflib.tools.rdf2dot import rdf2dot
import pydot

CCT = Namespace("https://github.com/quangis/cct/CCT.rdf#")

Int = Type.declare('Int')
add = Operation(Int ** Int ** Int, name='add')
one = Data(Int)
algebra = TransformationRDF(
    CCT,
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

import io
def visualize(g):
    """
    Return dot syntax.
    python3 test_rdf.py | xdot -
    """
    stream = io.StringIO()
    import cgi
    import html
    cgi.escape = html.escape
    rdf2dot(g, stream)
    return stream.getvalue()

print(visualize(g))
