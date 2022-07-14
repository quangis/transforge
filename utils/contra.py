#!/usr/bin/env python3

from __future__ import annotations

import csv
import importlib.machinery
import importlib.util
from rdflib import Graph  # type: ignore
from rdflib.term import Node, Literal  # type: ignore
from rdflib.namespace import Namespace, RDF, RDFS  # type: ignore
from rdflib.util import guess_format
from rdflib.tools.rdf2dot import rdf2dot  # type: ignore
from pathlib import Path
from plumbum import cli  # type: ignore
from itertools import chain
from transformation_algebra import TransformationQuery, TransformationGraph, \
    TA, Language
from typing import NamedTuple, Iterable

# Namespaces
WF = Namespace('http://geographicknowledge.de/vocab/Workflow.rdf#')
TOOLS = Namespace('https://github.com/quangis/cct/blob/master/tools/tools.ttl#')
# TOOLS = Namespace('http://geographicknowledge.de/vocab/GISTools.rdf#')
REPO = Namespace('https://example.com/#')


def graph(url: str) -> Graph:
    if url.startswith("http://") or url.startswith("https://"):
        g = Graph(store='SPARQLStore')
        g.open(url)
    else:
        g = Graph()
        g.parse(url, format=guess_format(url))
    return g


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


class Tatool(cli.Application):
    """
    A utility to create RDFs, graph visualizations, queries and other files
    relevant to workflows annotated with tool descriptions in terms of a
    transformation language
    """

    def main(self, *args):
        if args:
            print(f"Unknown command {args[0]}")
            return 1
        if not self.nested_command:
            self.help()
            return 1


@Tatool.subcommand("merge")
class Merger(cli.Application):
    """
    Merge RDF graphs
    """

    @cli.positional(cli.NonexistentPath, cli.ExistingFile)
    def main(self, output, *inputs):
        g = Graph()
        for i in inputs:
            g.parse(str(i))
        g.serialize(str(output), format='ttl', encoding='utf-8')


@Tatool.subcommand("vocab")
class VocabBuilder(cli.Application):
    "Build vocabulary file for the transformation language"

    language = cli.SwitchAttr(["-l", "--language"], argtype=lang,
        mandatory=True, help="Transformation language on which to operate")

    format = cli.SwitchAttr(["-f", "--format"],
        cli.Set("rdf", "ttl", "json-ld", "dot"), default="ttl")

    @cli.positional(cli.NonexistentPath)
    def main(self, output):
        Path(output).parent.mkdir(parents=True, exist_ok=True)  # build path should exist

        if self.format == "dot":
            vocab = TransformationGraph(self.language, minimal=True, with_labels=True)
            vocab.add_taxonomy()
            with open(output, 'w') as f:
                rdf2dot(vocab, f)
        else:
            vocab = TransformationGraph(self.language)
            vocab.add_vocabulary()
            vocab.serialize(str(output), format=self.format, encoding='utf-8')


@Tatool.subcommand("graph")
class TransformationGraphBuilder(cli.Application):
    """
    Generate transformation graphs for entire workflows, concatenating the
    algebra expressions for each individual use of a tool
    """

    language = cli.SwitchAttr(["-l", "--language"], argtype=lang,
        mandatory=True, help="Transformation language on which to operate")

    tools = cli.SwitchAttr(["--tools"], argtype=graph, mandatory=True,
        help="RDF graph containing the tool ontology")
    format = cli.SwitchAttr(["--format"],
        cli.Set("rdf", "ttl", "json-ld", "dot"), default="ttl")
    blocked = cli.Flag(["--blocked"], default=False,
        help="Do not pass output type of one tool to the next")
    opaque = cli.Flag(["--opaque"], default=False,
        help="Do not annotate types internal to the tools")

    @cli.positional(cli.ExistingFile, cli.NonexistentPath)
    def main(self, wf_path, output_path):
        visual = self.format == "dot"

        # Read input workflow graph
        wfg = Graph()
        wfg.parse(wf_path, format='ttl')
        root = wfg.value(None, RDF.type, WF.Workflow, any=False)
        sources = set(wfg.objects(root, WF.source))
        tool_apps: dict[Node, tuple[str, list[Node]]] = {}
        for step in wfg.objects(root, WF.edge):
            out = wfg.value(step, WF.output, any=False)

            # Find expression for the tool associated with this application
            tool = wfg.value(
                step, WF.applicationOf, any=False)
            assert tool, "workflow has an edge without a tool"
            expr = self.tools.value(
                tool, self.language.namespace.expression, any=False)
            assert expr, f"{tool} has no algebra expression"

            tool_apps[out] = expr, [node
                for pred in (WF.input1, WF.input2, WF.input3)
                if (node := wfg.value(step, pred))
            ]

        # Build transformation graph
        g = TransformationGraph(self.language, minimal=visual, with_labels=visual,
            with_noncanonical_types=False,
            with_intermediate_types=not self.opaque,
            passthrough=not self.blocked)
        step2expr = g.add_workflow(root, tool_apps, sources)

        # Annotate the expression nodes that correspond with output nodes of a
        # tool with said tool
        # TODO incorporate this into add_workflow
        if visual:
            for step in wfg.objects(root, WF.edge):
                out = wfg.value(step, WF.output, any=False)
                tool = wfg.value(step, WF.applicationOf, any=False)
                g.add((step2expr[out], RDFS.comment, Literal(
                    "using " + tool[len(TOOLS):])))
            for source in sources:
                for comment in wfg.objects(source, RDFS.comment):
                    g.add((source, RDFS.comment, comment))
            # g.add((step2expr[wf.output], RDFS.comment, Literal("output")))

        # Produce output file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        if visual:
            with open(output_path, 'w') as f:
                rdf2dot(g, f)
        else:
            g.serialize(str(output_path), format=self.format, encoding='utf-8')


class Task(NamedTuple):
    name: str
    query: TransformationQuery
    expected: set[Node]
    actual: set[Node]


@Tatool.subcommand("query")
class QueryRunner(cli.Application):
    """
    Run transformation queries against a SPARQL endpoint. If no endpoint is
    given, just output the query instead.
    """

    language = cli.SwitchAttr(["-l", "--language"], argtype=lang,
        mandatory=True, help="Transformation language on which to operate")
    output = cli.SwitchAttr(["-o", "--output"], cli.NonexistentPath,
        mandatory=True, help="Output file")
    format = cli.SwitchAttr(["--format"], cli.Set("sparql", "csv"),
        default="csv", help="Output format")
    chronological = cli.Flag(["--chronological"],
        default=False, help="Take into account order")
    blackbox = cli.Flag(["--blackbox"],
        default=False, help="Only consider input and output of the workflows")
    endpoint = cli.SwitchAttr(["--endpoint"], argtype=graph,
        help="SPARQL endpoint to send queries to")

    def evaluate(self, path, **opts) -> Task:
        """
        Parse and run a single task.
        """
        graph = TransformationGraph.from_rdf(path, self.language)
        query = TransformationQuery(self.language, graph, **opts)
        return Task(name=path.stem, query=query,
            expected=set(graph.objects(query.root, TA.implementation)),
            actual=(query.run(self.endpoint) if self.endpoint else set()))

    def summarize(self, tasks: Iterable[Task]) -> None:
        """
        Write a CSV summary of the tasks to the output.
        """

        workflows = set.union(*chain(
            (t.actual for t in tasks),
            (t.expected for t in tasks)))

        header = ["Task", "Precision", "Recall"] + sorted([
            str(wf)[len(REPO):] for wf in workflows])

        with open(self.output, 'w', newline='') as h:
            n_tpos, n_tneg, n_fpos, n_fneg = 0, 0, 0, 0
            w = csv.DictWriter(h, fieldnames=header)
            w.writeheader()
            for task in tasks:
                result: dict[str, str] = {"Task": task.name}
                expected, actual = task.expected, task.actual
                for wf in workflows:
                    s = "●" if wf in actual else "○"
                    if not expected:
                        s += "?"
                    elif (wf in actual) ^ (wf in expected):
                        s += "⨯"
                    result[str(wf)[len(REPO):]] = s
                n_fpos += len(actual - expected)
                n_fneg += len(expected - actual)
                n_tpos += len(actual.intersection(expected))
                n_tneg += len(workflows - expected - actual)
                w.writerow(result)
            try:
                w.writerow({
                    "Precision": "{0:.3f}".format(n_tpos / (n_tpos + n_fpos)),
                    "Recall": "{0:.3f}".format(n_tpos / (n_tpos + n_fneg))
                })
            except ZeroDivisionError:
                w.writerow({"Precision": "?", "Recall": "?"})

    @cli.positional(cli.ExistingFile)
    def main(self, *QUERY_FILE):
        if not QUERY_FILE:
            self.help()
            return 1
        else:
            # Parse tasks and optionally run associated queries
            tasks = [self.evaluate(task_file, by_io=True,
                by_operators=False, by_types=not self.blackbox,
                by_chronology=self.chronological and not self.blackbox,
            ) for task_file in QUERY_FILE]

            # Summarize query results
            if self.format == "sparql":
                with open(self.output, 'w', newline='') as h:
                    h.write("---\n")
                    for task in tasks:
                        h.write(task.query.sparql())
                        h.write("\n\nActual: ")
                        h.write(", ".join(t.n3() for t in task.actual))
                        h.write("\nExpected: ")
                        h.write(", ".join(t.n3() for t in task.expected))
                        h.write("\n---\n")
            else:
                assert self.format == "csv"
                self.summarize(tasks)


if __name__ == '__main__':
    Tatool.run()
