"""
Utility functions for common tasks.
"""

from __future__ import annotations

import csv
import importlib.machinery
import importlib.util
from rdflib import Graph  # type: ignore
from rdflib.term import Node, Literal  # type: ignore
from rdflib.namespace import Namespace, RDF, RDFS  # type: ignore
from rdflib.util import guess_format
from rdflib.tools.rdf2dot import rdf2dot  # type: ignore
from rdflib.plugins.stores.sparqlstore import SPARQLStore
from pathlib import Path
from plumbum import cli  # type: ignore
from itertools import chain
from transformation_algebra import TransformationQuery, TransformationGraph, \
    TA, Language

# Namespaces
WF = Namespace('http://geographicknowledge.de/vocab/Workflow.rdf#')
TOOLS = Namespace('https://github.com/quangis/cct/blob/master/tools/tools.ttl#')
# TOOLS = Namespace('http://geographicknowledge.de/vocab/GISTools.rdf#')
REPO = Namespace('https://example.com/#')


def graph(url: str) -> Graph:
    g = Graph()
    g.parse(url, format=guess_format(url))
    return g


def lang(path: str) -> Language:
    """
    Import a transformation language from a Python module containing one.
    """
    name = Path(path).stem
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    assert spec
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    languages = [lang for obj in dir(module)
        if isinstance(lang := getattr(module, obj), Language)]
    assert len(languages) == 1
    return languages[0]
