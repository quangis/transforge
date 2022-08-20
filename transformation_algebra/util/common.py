"""
Utility functions for common tasks.
"""

from __future__ import annotations
from pathlib import Path
from rdflib import Graph, Dataset
from rdflib.term import Node, URIRef, Literal
from rdflib.util import guess_format
from rdflib.tools.rdf2dot import rdf2dot
from transformation_algebra.namespace import TA, WF, TOOLS, REPO, RDF, RDFS
from transformation_algebra.lang import Language
from transformation_algebra.graph import TransformationGraph
from transformation_algebra.util.store import TransformationStore


def graph(url: str | Path, format: str | None = None) -> Graph:
    url = str(url)
    g = Graph()
    g.parse(url, format=format or guess_format(url))
    return g


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


def build_transformation(language: Language, tools: Graph, workflow: Graph,
        **kwargs) -> TransformationGraph:

    g = Graph()
    root = workflow.value(None, RDF.type, WF.Workflow, any=False)
    assert isinstance(root, URIRef)
    sources = set(workflow.objects(root, WF.source))
    tool_apps: dict[Node, tuple[str, list[Node]]] = {}
    for step in workflow.objects(root, WF.edge):
        out = workflow.value(step, WF.output, any=False)

        # Find expression for the tool associated with this application
        tool = workflow.value(
            step, WF.applicationOf, any=False)
        assert tool, "workflow has an edge without a tool"
        expr = tools.value(
            tool, language.namespace.expression, any=False)
        assert expr, f"{tool} has no algebra expression"

        tool_apps[out] = expr, [node
            for pred in (WF.input1, WF.input2, WF.input3)
            if (node := workflow.value(step, pred))
        ]

    # Build transformation graph
    g = TransformationGraph(language)
    step2expr = g.add_workflow(root, tool_apps, sources)

    # Annotate the expression nodes that correspond with output nodes
    # of a tool with said tool
    # TODO incorporate this into add_workflow
    if kwargs.get('with_labels'):
        for step in workflow.objects(root, WF.edge):
            out = workflow.value(step, WF.output, any=False)
            tool = workflow.value(step, WF.applicationOf, any=False)
            g.add((step2expr[out], RDFS.comment, Literal(
                "using " + tool[len(TOOLS):])))
        for source in sources:
            for comment in workflow.objects(source, RDFS.comment):
                g.add((source, RDFS.comment, comment))

    # Add original workflow
    g += workflow
    return g
