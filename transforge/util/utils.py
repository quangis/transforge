"""
This module contains some utility functions for uploading graphs, reading 
graphs, etcetera.
"""

import sys
from transforge.graph import TransformationGraph
from transforge.namespace import namespaces
from pathlib import Path
from rdflib import Dataset, Graph
from typing import Literal, TextIO


def bind_all(g: Graph) -> None:
    for prefix, ns in namespaces.items():
        g.bind(prefix, ns)
    if isinstance(g, TransformationGraph):
        if g.language.prefix:
            g.bind(g.language.prefix, g.language.namespace)


def write_graphs(*graphs: TransformationGraph,
        file: Path | TextIO = sys.stdout,
        format: Literal["dot", "trig", "ttl", "rdf", "json-ld"] = "ttl"):
    """
    Convenience method to write one or more transformation graphs to the given 
    file.
    """

    assert len(graphs) >= 1

    g: Dataset | Graph
    if format == "trig":
        g = Dataset()
        for graph in graphs:
            subgraph = g.graph(graph.base, graph.base)
            subgraph += graph
    elif len(graphs) > 1:
        g = Graph()
        for graph in graphs:
            g += graph
    else:
        assert len(graphs) == 1
        g = graphs[0]
    bind_all(g)

    result: str
    if format == "dot":
        if len(graphs) == 1:
            assert isinstance(g, TransformationGraph)
            r = g.visualize(None)
            assert isinstance(r, str)
            result = r
        else:
            raise RuntimeError("currently there must be exactly one graph "
                "when using the dot format")
    else:
        result = g.serialize(format=format)

    if isinstance(file, TextIO):
        print(result, file=file)
    else:
        assert isinstance(file, Path)
        with open(file, 'w') as f:
            f.write(result)
