"""
Utility class for the workflow store.
"""

from __future__ import annotations

import urllib.request
from rdflib import Graph
from rdflib.plugins.stores.sparqlstore import SPARQLStore
from typing import Literal


class WorkflowStore(Graph):
    def __init__(self, store: SPARQLStore):
        super().__init__(store)

    @staticmethod
    def endpoint(url: str, auth: Literal["NONE", "BASIC", "DIGEST"] = "NONE",
            username: str | None = None,
            password: str | None = None) -> WorkflowRepository:
        # While authorization can be passed to the `SPARQLStore` object, DIGEST
        # authorization is only handled for `SPARQLUpdateStore`. To work around
        # this, we add a custom opener to the `urllib.request` module that
        # SPARQLConnector is using under the hood. It seems we could also pull
        # the external dependency of SPARQLWrapper, mentioned in
        # <https://github.com/RDFLib/rdflib/issues/343>, but RDFLib dropped
        # this dependency in <https://github.com/RDFLib/rdflib/pull/744>, so
        # I'd rather not re-import it (but see also
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
        return WorkflowRepository(store)
