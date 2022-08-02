"""
Utility class for the workflow store.
"""
# TODO: Is this the best way? Not better to use SPARQLWrapper?

from __future__ import annotations

import urllib.request  # using urllib.request because rdflib uses it
from http.client import HTTPResponse
from urllib.parse import quote
from urllib.request import urlopen, Request
from rdflib import Dataset, Graph
from rdflib.plugins.stores.sparqlstore import SPARQLUpdateStore
from typing import Literal


class TransformationStore(Dataset):
    def __init__(self, url: str,
            cred: tuple[str, str] | None = None,
            auth: Literal["NONE", "BASIC", "DIGEST"] | None = None,
            graphstore_protocol: str = "/data",
            sparql_query: str = "/query",
            sparql_update: str = "/update"):

        self.url = url
        self.graphstore_protocol = graphstore_protocol

        # While authorization can be passed to the `SPARQLStore` object, DIGEST
        # authorization is only handled for `SPARQLUpdateStore`. To work around
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
            passmgr.add_password(None, url, username, password)
            auth_handler: urllib.request.BaseHandler
            if auth == "DIGEST":
                auth_handler = urllib.request.HTTPDigestAuthHandler(passmgr)
            else:
                assert auth == "BASIC"
                auth_handler = urllib.request.HTTPBasicAuthHandler(passmgr)
            opener = urllib.request.build_opener(auth_handler)
            urllib.request.install_opener(opener)

        store = SPARQLUpdateStore(url + sparql_query, url + sparql_update)
        super().__init__(store)

    def put(self, g: Graph) -> HTTPResponse:
        """
        Remove old graph (if any) and insert given one in its place.
        """
        # cf. <https://www.w3.org/TR/sparql11-http-rdf-update/#http-put>
        # TODO may throw errors etc

        assert g.base
        r = Request(
            f"{self.url}{self.graphstore_protocol}?graph={quote(str(g.base))}",
            method='PUT',
            headers={"Content-Type": "text/turtle"},
            data=g.serialize(format="ttl", encoding="utf-8")
        )
        return urlopen(r)
