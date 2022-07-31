"""
Utility functions for common tasks.
"""

from __future__ import annotations

import urllib.request
import importlib.machinery
import importlib.util
from rdflib import Graph
from rdflib.namespace import Namespace
from rdflib.util import guess_format
from rdflib.plugins.stores.sparqlstore import SPARQLStore
from pathlib import Path
from transformation_algebra import Language
from typing import Literal

# Namespaces
WF = Namespace('http://geographicknowledge.de/vocab/Workflow.rdf#')
TOOLS = Namespace('https://github.com/quangis/cct/blob/master/tools/tools.ttl#')
# TOOLS = Namespace('http://geographicknowledge.de/vocab/GISTools.rdf#')
REPO = Namespace('https://example.com/#')


def graph(url: str) -> Graph:
    g = Graph()
    g.parse(url, format=guess_format(url))
    return g


def endpoint(url: str, auth: Literal["NONE", "BASIC", "DIGEST"] = "NONE",
        username: str | None = None, password: str | None = None) -> Graph:
    # While you can pass authorization to the `SPARQLStore` object, this
    # authorization is basic; DIGEST authorization is only handled for
    # `SPARQLUpdateStore`. To work around this, we add a custom opener to the
    # `urllib.request` module that SPARQLConnector is using under the hood. It
    # seems we could also pull the external dependency of SPARQLWrapper, as
    # mentioned in <https://github.com/RDFLib/rdflib/issues/343>, but RDFLib
    # dropped this dependency in <https://github.com/RDFLib/rdflib/pull/744>,
    # so I'd rather not re-import it (but see also
    # <https://github.com/RDFLib/sparqlwrapper/pull/126>).
    if auth != "NONE":
        assert username and password
        passmgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        passmgr.add_password(None, url, username, password)
        auth_handler: urllib.request.BaseHandler
        if auth == "BASIC":
            auth_handler = urllib.request.HTTPBasicAuthHandler(passmgr)
        else:
            assert auth == "DIGEST"
            auth_handler = urllib.request.HTTPDigestAuthHandler(passmgr)
        opener = urllib.request.build_opener(auth_handler)
        urllib.request.install_opener(opener)
    store = SPARQLStore(url)
    return Graph(store)


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
