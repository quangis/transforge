"""
Utility functions for common tasks.
"""

from __future__ import annotations
from pathlib import Path
from rdflib import Graph
from rdflib.util import guess_format
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
