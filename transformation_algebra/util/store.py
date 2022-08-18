"""
Utility class for the workflow store.
"""
# TODO: Is this the best way? Not better to use SPARQLWrapper?

from __future__ import annotations

import urllib.request  # using urllib.request because rdflib uses it
from http.client import HTTPResponse
from urllib.parse import quote, urlparse
from urllib.request import urlopen, Request
from rdflib import Dataset, Graph, URIRef
from rdflib.plugins.stores.sparqlstore import SPARQLStore
from typing import Literal


class TransformationStore(Dataset):
    def __init__(self,
            url_gsp: str,
            url_sparql: str,
            cred: tuple[str, str] | None = None,
            auth: Literal["NONE", "BASIC", "DIGEST"] | None = None):

        url = urlparse(url_gsp)
        self.host = f"{url.scheme}://{url.netloc}"
        self.url_gsp = url_gsp
        self.url_sparql = url_sparql

        # While authorization can be passed to the `SPARQLStore` object, DIGEST
        # authorization is only handled for `SPARQLStore`. To work around
        # this, we add a custom opener to the `urllib.request` module that
        # SPARQLConnector is using under the hood. It seems we could also pull
        # the external dependency of SPARQLWrapper, mentioned in
        # <https://github.com/RDFLib/rdflib/issues/343>, but RDFLib dropped
        # this dependency in <https://github.com/RDFLib/rdflib/pull/744>, so
        # I'd rather not re-import it (but see also
        # <https://github.com/RDFLib/sparqlwrapper/pull/126>).
        if auth is None:
            auth = "DIGEST" if cred else "NONE"

        if auth != "NONE":
            username, password = cred or (None, None)
            assert username and password
            passmgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
            passmgr.add_password(None, self.host, username, password)
            auth_handler: urllib.request.BaseHandler
            if auth == "DIGEST":
                auth_handler = urllib.request.HTTPDigestAuthHandler(passmgr)
            else:
                assert auth == "BASIC"
                auth_handler = urllib.request.HTTPBasicAuthHandler(passmgr)
            opener = urllib.request.build_opener(auth_handler)
            urllib.request.install_opener(opener)

        store = SPARQLStore(url_sparql)
        super().__init__(store)

    @staticmethod
    def backend(backend: str, *nargs, **kwargs) -> TransformationStore:
        backend = backend.lower()
        if backend == "fuseki":
            return Fuseki(*nargs, **kwargs)
        elif backend == "marklogic":
            return MarkLogic(*nargs, **kwargs)
        else:
            raise RuntimeError

    def put(self, g: Graph) -> HTTPResponse:
        """
        Remove old graph (if any) and insert given one in its place.
        """
        # cf. <https://www.w3.org/TR/sparql11-http-rdf-update/#http-put>
        # TODO may throw errors etc

        if g.base:
            url = f"{self.url_gsp}?graph={quote(str(g.base))}"
        else:
            url = f"{self.url_gsp}?default"

        return urlopen(Request(
            url, method='PUT',
            headers={"Content-Type": "application/rdf+xml"},
            data=g.serialize(format="xml", encoding="utf-8")
        ))

    def get(self, uri: URIRef | str) -> Graph:
        url = f"{self.url_gsp}?graph={quote(str(uri))}"
        resp = urlopen(Request(
            url, method='GET',
            headers={"Accept": "text/application/rdf+xml; charset=utf-8"},
        ))
        data = resp.read()
        g = Graph()
        g.parse(data=data, format="xml")
        return g


class MarkLogic(TransformationStore):
    # cf. <https://docs.marklogic.com/guide/semantics/REST>
    def __init__(self, url: str, **kwargs):
        super().__init__(
            url_gsp=url + "/v1/graphs",
            url_sparql=url + "/v1/graphs/sparql",
            **kwargs)


class Fuseki(TransformationStore):
    def __init__(self, url: str, **kwargs):
        super().__init__(
            url_gsp=url.rstrip("/") + "/data",
            url_sparql=url.rstrip("/") + "/query"
        )
