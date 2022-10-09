"""
Utility functions for common tasks.
"""

from __future__ import annotations
from pathlib import Path
from rdflib import Graph, Dataset
from rdflib.util import guess_format
from rdflib.tools.rdf2dot import rdf2dot
from transformation_algebra.namespace import TA, WF, TOOLS, EX
from transformation_algebra.graph import TransformationGraph
from transformation_algebra.util.store import TransformationStore


def graph(*urls: str | Path, format: str | None = None) -> Graph:
    g = Graph()
    for url in urls:
        g.parse(str(url), format=format or guess_format(url))
    return g


def to_store(*graphs: Graph, **kwargs):
    """
    Convenience method to send one or more graphs to the given store,
    overwriting old ones if they exist.
    """
    ds = TransformationStore.backend(**kwargs)
    for g in graphs:
        ds.put(g)


def to_file(*graphs: TransformationGraph, path: str, format: str | None = None):
    """
    Convenience method to write one or more graphs to the given file.
    """
    if format == "dot":
        for g1 in graphs:
            g1.visualize()
    else:
        result: Graph
        if len(graphs) == 1:
            result = graphs[0]
        elif format == "trig":
            result = Dataset()
            for g in graphs:
                subgraph = result.graph(g.base, g.base)
                subgraph += g
        else:
            for g in graphs:
                result += g

        result.bind("ta", TA)
        result.bind("wf", WF)
        result.bind("tools", TOOLS)
        result.bind("ex", EX)

        for g in graphs:
            if isinstance(g, TransformationGraph):
                if g.language.prefix:
                    result.bind(g.language.prefix, g.language.namespace)

        # Produce output file
        if path:
            result.serialize(path,
                format=format or guess_format(path))
