from __future__ import annotations

import rdflib
from rdflib import Namespace
from rdflib.term import URIRef

TF = Namespace('https://github.com/quangis/transforge#')
WF = Namespace('http://geographicknowledge.de/vocab/Workflow.rdf#')
TOOLS = Namespace(
    'https://github.com/quangis/cct/blob/master/tools/tools.ttl#')
# TOOLS = Namespace('http://geographicknowledge.de/vocab/GISTools.rdf#')
EX = Namespace('https://example.com/#')
RDF = rdflib.RDF
RDFS = rdflib.RDFS


def shorten(uri: URIRef | str) -> str:
    """
    Return the part after the last #, or if that is empty, after the last /.
    """
    s = str(uri)
    last = "#".join(s.split("#")[-1:])
    if last == s:
        last = "/".join(s.split("/")[-1:])
    return last or s
