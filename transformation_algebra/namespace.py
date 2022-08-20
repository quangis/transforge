import rdflib
from rdflib import Namespace

TA = Namespace("https://github.com/quangis/transformation-algebra#")
WF = Namespace('http://geographicknowledge.de/vocab/Workflow.rdf#')
TOOLS = Namespace(
    'https://github.com/quangis/cct/blob/master/tools/tools.ttl#')
# TOOLS = Namespace('http://geographicknowledge.de/vocab/GISTools.rdf#')
EX = Namespace('https://example.com/')
RDF = rdflib.RDF
RDFS = rdflib.RDFS
