"""
Command-line interface for common tasks.
"""

from __future__ import annotations

import csv
from rdflib import Graph
from rdflib.term import Node, Literal, URIRef
from rdflib.namespace import RDF, RDFS
from rdflib.tools.rdf2dot import rdf2dot
from plumbum import cli  # type: ignore
from itertools import chain
from transformation_algebra import TransformationQuery, TransformationGraph, \
    TA
from typing import NamedTuple, Iterable

from transformation_algebra.util import graph, lang, WF, TOOLS, REPO
from transformation_algebra.util.store import TransformationStore


def cred(s: str) -> tuple[str, str]:
    username, _, password = s.partition(":")
    return username, password


class Application(cli.Application):
    language = cli.SwitchAttr(["-L", "--language"], argtype=lang,
        mandatory=True, help="Transformation language on which to operate")


class WithTools:
    tools = cli.SwitchAttr(["-T", "--tools"], argtype=graph, mandatory=True,
        help="RDF graph containing the tool ontology")


class WithRDF:
    output_path = cli.SwitchAttr(["-o", "--output"],
        help="file which to write to")
    output_format = cli.SwitchAttr(["-t", "--to"],
        cli.Set("rdf", "ttl", "json-ld", "dot"), default="ttl",
        requires=["-o"])

    def maybe_output(self, g: Graph):
        # Produce output file
        if self.output_path:
            if self.output_format == "dot":
                with open(self.output_path, 'w') as f:
                    rdf2dot(g, f)
            else:
                g.serialize(self.output_path, format=self.output_format)


class WithServer:
    backend = cli.SwitchAttr(["-b", "--backend"],
        cli.Set("fuseki", "marklogic"))
    server = cli.SwitchAttr(["-s", "--server"],
        help="server to which to send the graph to", requires=["-b"])
    cred = cli.SwitchAttr(["-u", "--user"], argtype=cred, requires=["-s"])

    def maybe_upload(self, g: Graph):
        # Insert new graph into server (overwriting old one if it exists)
        if self.server:
            ds = TransformationStore.backend(self.backend,
                self.server, cred=self.cred)
            ds.put(g)


class CLI(cli.Application):
    """
    A utility to create RDFs, graph visualizations, queries and other files
    relevant to workflows annotated with tool descriptions in terms of a
    transformation language
    """

    PROGNAME = "transformation_algebra"

    def main(self, *args):
        if args:
            print(f"Unknown command {args[0]}")
            return 1
        if not self.nested_command:
            self.help()
            return 1


@CLI.subcommand("merge")
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


@CLI.subcommand("vocab")
class VocabBuilder(Application, WithRDF, WithServer):
    "Build vocabulary file for the transformation language"

    def main(self):
        if self.output_format == "dot":
            vocab = TransformationGraph(self.language, minimal=True,
                with_labels=True)
            vocab.add_taxonomy()
        else:
            vocab = TransformationGraph(self.language)
            vocab.add_vocabulary()
        vocab.base = self.language.namespace
        self.maybe_output(vocab)
        self.maybe_upload(vocab)


@CLI.subcommand("graph")
class TransformationGraphBuilder(Application, WithTools, WithRDF, WithServer):
    """
    Generate transformation graphs for entire workflows, concatenating the
    algebra expressions for each individual use of a tool
    """

    blocked = cli.Flag(["--blocked"], default=False,
        help="Do not pass output type of one tool to the next")
    opaque = cli.Flag(["--opaque"], default=False,
        help="Do not annotate types internal to the tools")

    @cli.positional(cli.ExistingFile)
    def main(self, wf_path):
        visual = self.output_format == "dot"
        if visual:
            assert not self.endpoint

        # Read input workflow graph
        wfg = Graph()
        wfg.parse(wf_path, format='ttl')
        root = wfg.value(None, RDF.type, WF.Workflow, any=False)
        assert isinstance(root, URIRef)
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
        g = TransformationGraph(self.language, minimal=visual,
            with_labels=visual, with_noncanonical_types=False,
            with_intermediate_types=not self.opaque,
            passthrough=not self.blocked)
        g.base = root
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

        self.maybe_output(g)
        self.maybe_upload(g)


class Task(NamedTuple):
    name: str
    query: TransformationQuery
    expected: set[Node]
    actual: set[Node]


@CLI.subcommand("query")
class QueryRunner(Application, WithServer):
    """
    Run transformation queries against a SPARQL endpoint. If no endpoint is
    given, just output the query instead.
    """

    output = cli.SwitchAttr(["-o", "--output"], cli.NonexistentPath,
        mandatory=True, help="Output file")
    output_format = cli.SwitchAttr(["-t", "--to"], cli.Set("sparql", "csv"),
        default="csv", help="Output format")

    chronological = cli.Flag(["--chronological"],
        default=False, help="Take into account order")
    blackbox = cli.Flag(["--blackbox"],
        default=False, help="Only consider input and output of the workflows")

    def evaluate(self, path, **opts) -> Task:
        """
        Parse and run a single task.
        """
        graph = TransformationGraph.from_rdf(path, self.language)
        query = TransformationQuery(self.language, graph, **opts)
        return Task(name=path.stem, query=query,
            expected=set(graph.objects(query.root, TA.implementation)),
            actual=(query.run(self.graph) if self.graph else set()))

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

            if self.server:
                username, password = self.cred or (None, None)
                self.graph = TransformationStore(self.backend, self.server,
                    cred=self.cred)
            else:
                self.graph = None

            # Parse tasks and optionally run associated queries
            tasks = [self.evaluate(task_file, by_io=True,
                by_operators=False, by_types=not self.blackbox,
                by_chronology=self.chronological and not self.blackbox)
                for task_file in QUERY_FILE]

            # Summarize query results
            if self.output_format == "sparql":
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
                assert self.output_format == "csv"
                self.summarize(tasks)


if __name__ == '__main__':
    CLI.run()