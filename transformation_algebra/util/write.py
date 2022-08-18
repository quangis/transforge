"""
Utility functions for common tasks.
"""

from __future__ import annotations
from rdflib import Graph, Dataset
from rdflib.util import guess_format
from rdflib.tools.rdf2dot import rdf2dot
from transformation_algebra.namespace import TA, WF, TOOLS, REPO
from transformation_algebra.graph import TransformationGraph
from transformation_algebra.util.store import TransformationStore


def to_store(*graphs: Graph, **kwargs):
    """
    Convenience method to send one or more graphs to the given store,
    overwriting old ones if they exist.
    """
    ds = TransformationStore.backend(**kwargs)
    for g in graphs:
        ds.put(g)


def to_file(*graphs: Graph, path: str, format: str | None = None):
    """
    Convenience method to write one or more graphs to the given file.
    """
    result: Graph
    if len(graphs) == 1:
        result = graphs[0]
    elif format == "trig":
        result = Dataset()
        for g in graphs:
            subgraph = result.graph(g.base, g.base)
            subgraph += g
    else:
        g = Graph()
        for g in graphs:
            result += g

    result.bind("ta", TA)
    result.bind("wf", WF)
    result.bind("tools", TOOLS)
    result.bind("repo", REPO)

    for g in graphs:
        if isinstance(g, TransformationGraph):
            if g.language.prefix:
                result.bind(g.language.prefix, g.language.namespace)

    # Produce output file
    if path:
        if format == "dot":
            with open(path, 'w') as f:
                rdf2dot(result, f)
        else:
            result.serialize(path,
                format=format or guess_format(path))
