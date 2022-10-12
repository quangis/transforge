"""
Utility functions for common tasks.
"""

from __future__ import annotations
from sys import stdout
from pathlib import Path
from rdflib import Graph, Dataset
from rdflib.util import guess_format
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


def to_file(*graphs: TransformationGraph, path: str | None,
        format: str | None = None):
    """
    Convenience method to write one or more graphs to the given file.
    """
    result: str | None
    if format == "dot":
        if len(graphs) == 1:
            result = graphs[0].visualize(path)
        else:
            raise RuntimeError("there must be exactly one graph")
    else:
        all_g: Graph
        if len(graphs) == 1:
            all_g = graphs[0]
        elif format == "trig":
            all_g = Dataset()
            for g in graphs:
                subgraph = all_g.graph(g.base, g.base)
                subgraph += g
        else:
            for g in graphs:
                all_g += g

        all_g.bind("ta", TA)
        all_g.bind("wf", WF)
        all_g.bind("tools", TOOLS)
        all_g.bind("ex", EX)

        for g in graphs:
            if isinstance(g, TransformationGraph):
                if g.language.prefix:
                    all_g.bind(g.language.prefix, g.language.namespace)

        if path:
            all_g.serialize(path, format=format or guess_format(path))
        else:
            result = all_g.serialize(format=format or guess_format(path))

    if not path:
        print(result)
