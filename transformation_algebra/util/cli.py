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
from rdflib import Graph, Dataset
from rdflib.term import Node, Literal
from rdflib.util import guess_format
from transformation_algebra.base import TransformationError
from transformation_algebra.lang import Language
from transformation_algebra.graph import TransformationGraph
from transformation_algebra.query import TransformationQuery
from transformation_algebra.namespace import TA, EX, WF, TOOLS
from transformation_algebra.workflow import WorkflowGraph
from transformation_algebra.util.store import TransformationStore
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
        cli.Set("rdf", "ttl", "trig", "json-ld", "dot"), default="trig")

    def write(self, *graphs: TransformationGraph):
        """
        Convenience method to write one or more graphs to the given file.
        """
        if not self.output_path or self.output_path == "-":
            path = None
        else:
            path = self.output_path

        result: str | None
        if self.output_format == "dot":
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
                all_g = Graph()
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
                all_g.serialize(path,
                    format=self.output_format or guess_format(path))
            else:
                result = all_g.serialize(
                    format=self.output_format or guess_format(path))

        if not path:
            print(result)


class WithServer:
    backend = cli.SwitchAttr(["-b", "--backend"],
        cli.Set("fuseki", "marklogic"), requires=["-s"])
    server = cli.SwitchAttr(["-s", "--server"],
        help="server to which to send the graph to", requires=["-b"])
    cred = cli.SwitchAttr(["-u", "--user"], argtype=cred, requires=["-s"])

    def upload(self, *graphs: Graph, **kwargs):
        """
        Convenience method to send one or more graphs to the given store,
        overwriting old ones if they exist.
        """
        if self.server:
            ds = TransformationStore.backend(
                self.backend, self.server, self.cred)
            for g in graphs:
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

        self.upload(vocab)
        self.write(vocab)


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
    skip_error = cli.Flag(["--skip-error"], default=False,
        help="Skip failed transformation graphs instead of exiting")

    @cli.positional(cli.ExistingFile)
    def main(self, *WORKFLOW_FILE):
        results: list[Graph] = []

        if not (WORKFLOW_FILE or self.expressions):
            print("Error: missing expression or workflow graph", file=stderr)
            return 1

        # Windows does not interpret asterisks as globs, so we do that manually
        if platform.system() == 'Windows':
            WORKFLOW_FILE = [globbed
                for original in WORKFLOW_FILE.copy()
                for globbed in glob(original)]

        for i, expr in enumerate(self.expressions):
            tg = TransformationGraph(self.language,
                with_noncanonical_types=False,
                with_intermediate_types=not self.opaque,
                passthrough=not self.blocked)
            tg.uri = root = EX[f"expr{i}"]

            try:
                e = tg.add_expr(self.language.parse(expr), root)
            except TransformationError as e:
                if self.skip_error:
                    print(f"Skipping expression:\n\t{e}", file=stderr)
                else:
                    print(e, file=stderr)
                    return 1

            tg.add((root, TA.output, e))
            results.append(tg)

        for wf_path in WORKFLOW_FILE:
            wf = WorkflowGraph(self.language, self.tools)
            wf.parse(wf_path, format=guess_format(wf_path))
            wf.refresh()

            tg = TransformationGraph(self.language,
                with_noncanonical_types=False,
                with_intermediate_types=not self.opaque,
                passthrough=not self.blocked)

            try:
                tg.add_workflow(wf)
            except TransformationError as e:
                if self.skip_error:
                    print(f"Skipping {wf_path}:\n\t{e}", file=stderr)
                else:
                    print(e, file=stderr)
                    return 1
            else:
                tg += wf
                results.append(tg)

        self.upload(*results)
        self.write(*results)


class Task(NamedTuple):
    name: str
    query: TransformationQuery
    expected: set[Node]
    actual: set[Node]


@CLI.subcommand("query")
class QueryRunner(Application, WithServer, WithRDF):
    """
    Run transformation queries against a SPARQL endpoint. If no endpoint is
    given, just output the query instead.
    """

    chronological = cli.Flag(["--chronological"],
        default=False, help="Take into account order")
    blackbox = cli.Flag(["--blackbox"],
        default=False, help="Only consider input and output of the workflows")

    def evaluate(self, path, **opts) -> Graph:
        """
        Parse and run a single task.
        """
        in_graph = Graph()
        in_graph.parse(path)
        query = TransformationQuery(self.language, in_graph, **opts)

        # expected = set(graph.objects(query.root, TA.implementation))
        actual = query.run(self.store) if self.store else set()
        sparql = query.sparql()

        out_graph = query.graph
        out_graph.add((query.root, TA.sparql, Literal(sparql)))
        for match in actual:
            out_graph.add((query.root, TA.match, match))
        return out_graph

        # return Task(name=path.stem, query=query,
        #     expected=set(graph.objects(query.root, TA.implementation)),
        #     actual=(query.run(self.graph) if self.graph else set()))

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

            # Windows does not interpret asterisks as globs
            if platform.system() == 'Windows':
                QUERY_FILE = [globbed
                    for original in QUERY_FILE.copy()
                    for globbed in glob(original)]

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

            self.write(*tasks)

            # if self.output_path:
            #     path = self.output_path
            #     path = None if path == "-" else path
            #     to_file(*tasks, path=path, format="ttl")

            # # Summarize query results
            # if self.output_format == "sparql":
            #     with open(self.output, 'w', newline='') as h:
            #         h.write("---\n")
            #         for task in tasks:
            #             h.write(task.query.sparql())
            #             h.write("\n\nActual: ")
            #             h.write(", ".join(t.n3() for t in task.actual))
            #             h.write("\nExpected: ")
            #             h.write(", ".join(t.n3() for t in task.expected))
            #             h.write("\n---\n")
            # else:
            #     assert self.output_format == "csv"
            #     self.summarize(tasks)


if __name__ == '__main__':
    CLI.run()
