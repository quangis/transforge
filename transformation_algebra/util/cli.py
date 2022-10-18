"""
Command-line interface for common tasks.
"""

from __future__ import annotations

import csv
import platform
import importlib.machinery
import importlib.util
from sys import stderr
from pathlib import Path
from itertools import chain
from glob import glob

from plumbum import cli  # type: ignore
from rdflib import Graph
from rdflib.term import Node
from rdflib.util import guess_format
from transformation_algebra.lang import Language
from transformation_algebra.graph import TransformationGraph
from transformation_algebra.query import TransformationQuery
from transformation_algebra.namespace import TA, EX
from transformation_algebra.workflow import WorkflowGraph
from transformation_algebra.util.store import TransformationStore
from transformation_algebra.util.common import (to_store, to_file)
from typing import NamedTuple, Iterable


def lang(path_or_module: str) -> Language:
    """
    Import a transformation language from a Python module containing one.
    """
    try:
        module = __import__(path_or_module)
    except ModuleNotFoundError:
        name = Path(path_or_module).stem
        loader = importlib.machinery.SourceFileLoader(name, path_or_module)
        spec = importlib.util.spec_from_loader(name, loader)
        assert spec
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
    finally:
        languages = [lang for obj in dir(module)
            if isinstance(lang := getattr(module, obj), Language)]
        assert len(languages) == 1
        return languages[0]


def cred(s: str) -> tuple[str, str]:
    username, _, password = s.partition(":")
    return username, password


class Application(cli.Application):
    language = cli.SwitchAttr(["-L", "--language"], argtype=lang,
        mandatory=True, help="Transformation language on which to operate")


class WithTools:
    @cli.switch(["-T", "--tools"], list=True, argtype=str,
        help="RDF graph(s) containing the tool ontology")
    def _tools(self, urls: list[str]) -> Graph:
        self.tools = Graph()
        for url in urls:
            self.tools.parse(url, format=guess_format(url))
        return self.tools


class WithRDF:
    output_path = cli.SwitchAttr(["-o", "--output"],
        help="file which to write to, or - for stdout")
    output_format = cli.SwitchAttr(["-t", "--to"],
        cli.Set("rdf", "ttl", "trig", "json-ld", "dot"), default="trig",
        requires=["-o"])


class WithServer:
    backend = cli.SwitchAttr(["-b", "--backend"],
        cli.Set("fuseki", "marklogic"))
    server = cli.SwitchAttr(["-s", "--server"],
        help="server to which to send the graph to", requires=["-b"])
    cred = cli.SwitchAttr(["-u", "--user"], argtype=cred, requires=["-s"])


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

        if self.server:
            to_store(vocab, backend=self.backend, url=self.server,
                cred=self.cred)

        if self.output_path:
            to_file(vocab, path=self.output_path, format=self.output_format)


@CLI.subcommand("graph")
class TransformationGraphBuilder(Application, WithTools, WithRDF, WithServer):
    """
    Generate transformation graphs for entire workflows, concatenating the
    algebra expressions for each individual use of a tool
    """

    expressions = cli.SwitchAttr(["-e", "--expression"], list=True,
        help="Provide an expression to add to the graph.")
    blocked = cli.Flag(["--blocked"], default=False,
        help="Do not pass output type of one tool to the next")
    opaque = cli.Flag(["--opaque"], default=False,
        help="Do not annotate types internal to the tools")

    @cli.positional(cli.ExistingFile)
    def main(self, *wf_paths):
        results: list[Graph] = []

        if not (wf_paths or self.expressions):
            print("Error: missing expression or workflow graph", file=stderr)
            return 1

        # Windows does not interpret asterisks as globs, so we do that manually
        if platform.system() == 'Windows':
            wf_paths = [globbed
                for original in wf_paths.copy()
                for globbed in glob(original)]

        for i, expr in enumerate(self.expressions):
            tg = TransformationGraph(self.language,
                with_noncanonical_types=False,
                with_intermediate_types=not self.opaque,
                passthrough=not self.blocked)
            tg.uri = root = EX[f"expr{i}"]
            e = tg.add_expr(self.language.parse(expr), root)
            tg.add((root, TA.output, e))
            results.append(tg)

        for wf_path in wf_paths:
            wf = WorkflowGraph(self.language, self.tools)
            wf.parse(wf_path, format=guess_format(wf_path))
            wf.refresh()

            tg = TransformationGraph(self.language,
                with_noncanonical_types=False,
                with_intermediate_types=not self.opaque,
                passthrough=not self.blocked)
            tg.add_workflow(wf)
            tg += wf
            results.append(tg)

        if self.server:
            to_store(*results, backend=self.backend, url=self.server,
                cred=self.cred)
        if self.output_path:
            path = self.output_path
            path = None if path == "-" else path
            to_file(*results, path=path, format=self.output_format)


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
            str(wf)[len(EX):] for wf in workflows])

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
                    result[str(wf)[len(EX):]] = s
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
                self.store = TransformationStore(self.backend, self.server,
                    cred=self.cred)
            else:
                self.store = None

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
