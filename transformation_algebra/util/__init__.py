"""
Utility functions for common tasks.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
from rdflib import Graph
from rdflib.namespace import Namespace
from rdflib.util import guess_format
from pathlib import Path
from transformation_algebra import Language

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
