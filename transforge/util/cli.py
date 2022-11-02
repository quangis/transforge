"""
Command-line interface for common tasks.
"""

from __future__ import annotations

import csv
import platform
import importlib.machinery
import importlib.util
from sys import stdout, stderr
from pathlib import Path
from itertools import chain
from collections import defaultdict
from glob import glob

from plumbum import cli  # type: ignore
from rdflib import Graph, Dataset
from rdflib.term import Literal, Node, URIRef
from rdflib.util import guess_format
from transforge.expr import ApplicationError
from transforge.lang import Language, ParseError
from transforge.graph import TransformationGraph, \
    WorkflowCompositionError
from transforge.query import TransformationQuery
from transforge.namespace import TF, EX, WF, TOOLS, RDF, shorten
from transforge.workflow import WorkflowGraph
from transforge.util.store import TransformationStore


caught_errors = (WorkflowCompositionError, ApplicationError, ParseError)


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

        if self.output_path == "-":
            path = None
            to_stdout = True
        else:
            path = self.output_path
            to_stdout = False

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

            all_g.bind("tf", TF)
            all_g.bind("wf", WF)
            all_g.bind("tools", TOOLS)
            all_g.bind("ex", EX)

            for g in graphs:
                if isinstance(g, TransformationGraph):
                    if g.language.prefix:
                        all_g.bind(g.language.prefix, g.language.namespace)

            result = all_g.serialize(path,
                format=self.output_format or guess_format(path))

        if to_stdout:
            assert result
            print(result)


class WithServer:
    server = cli.SwitchAttr(["-s", "--server"],
        help="server to send graphs to. "
        "format: backend@host; "
        "supported backends: fuseki, marklogic")
    cred = cli.SwitchAttr(["-u", "--user"], requires=["-s"])

    def upload(self, *graphs: Graph, **kwargs):
        """
        Convenience method to send one or more graphs to the given store,
        overwriting old ones if they exist.
        """

        if self.server:
            backend, _, host = self.server.partition("@")

            if self.cred:
                username, _, password = self.cred.partition(":")
                cred = username, password
            else:
                cred = None

            ds = TransformationStore.backend(backend, host, cred)
            for g in graphs:
                ds.put(g)


class CLI(cli.Application):
    """
    A utility to create RDFs, graph visualizations, queries and other files
    relevant to workflows annotated with tool descriptions in terms of a
    transformation language
    """

    PROGNAME = "transforge"

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

    def main(self, *WORKFLOW_FILE):
        results: list[Graph] = []

        if not (WORKFLOW_FILE or self.expressions):
            print("Error: missing expression or workflow graph", file=stderr)
            return 1

        # Windows does not interpret asterisks as globs, so we do that manually
        if platform.system() == 'Windows':
            WORKFLOW_FILE = tuple(globbed
                for original in WORKFLOW_FILE
                for globbed in glob(original))

        for i, expr in enumerate(self.expressions):
            tg = TransformationGraph(self.language,
                with_noncanonical_types=True,
                with_intermediate_types=not self.opaque,
                passthrough=not self.blocked)
            tg.uri = root = EX[f"expr{i}"]

            try:
                e = tg.add_expr(self.language.parse(expr), root)
            except caught_errors as e:
                if self.skip_error:
                    print(f"Skipping expression:\n\t{e}", file=stderr)
                else:
                    print(e, file=stderr)
                    return 1

            tg.add((root, TF.output, e))
            results.append(tg)

        for wf_path in WORKFLOW_FILE:
            wf = WorkflowGraph(self.language, self.tools)
            wf.parse(wf_path, format=guess_format(wf_path))
            wf.refresh()

            tg = TransformationGraph(self.language,
                with_noncanonical_types=True,
                with_intermediate_types=not self.opaque,
                passthrough=not self.blocked)

            try:
                tg.add_workflow(wf)
            except caught_errors as e:
                if self.skip_error:
                    print(f"Skipping {wf_path}:\n\t{e}", file=stderr)
                else:
                    print(f"Error in {wf_path}:\n\t{e}", file=stderr)
                    return 1
            else:
                tg += wf
                results.append(tg)

        self.upload(*results)
        self.write(*results)


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
    summary = cli.SwitchAttr(["--summary"], requires=["--server"],
        help="Write a CSV with a summary of all queries")

    def evaluate(self, path: str) -> Graph:
        """
        Parse and run a single task and add results to the task graph.
        """
        in_graph = Graph()
        in_graph.parse(path)

        query = TransformationQuery(self.language, in_graph, by_io=True,
            by_operators=False, by_types=not self.blackbox,
            by_chronology=self.chronological and not self.blackbox)

        out_graph = query.graph
        out_graph.add((query.root, TF.sparql, Literal(query.sparql())))

        if self.server:
            for match in self.store.run(query):
                out_graph.add((query.root, TF.match, match))
        return out_graph

    def summarize(self, *task_graphs: Graph) -> None:
        """
        Write a CSV summary of the tasks to the output.
        """

        # Collect all tasks
        tasks: dict[URIRef, Graph] = {task: tg for tg in task_graphs
            if (task := tg.value(None, RDF.type, TF.Task, any=False))}

        # Collect all possible workflows
        workflows = set(wf for task, tg in tasks.items() for wf in chain(
            tg.objects(task, TF.implementation), tg.objects(task, TF.match)))

        # Collect results
        n_tpos, n_tneg, n_fpos, n_fneg = 0, 0, 0, 0
        results: dict[Node, dict[str, str]] = defaultdict(dict)
        for task, tg in tasks.items():
            expected = set(tg.objects(task, TF.implementation))
            actual = set(tg.objects(task, TF.match))

            if expected:
                n_fpos += len(actual - expected)
                n_fneg += len(expected - actual)
                n_tpos += len(actual.intersection(expected))
                n_tneg += len(workflows - expected - actual)

            for wf in workflows:
                assert isinstance(wf, URIRef)
                s = "●" if wf in actual else "○"
                if not expected:
                    s += "?"
                elif (wf in actual) ^ (wf in expected):
                    s += "⨯"
                results[wf]["Workflow"] = shorten(wf)
                results[wf][shorten(task)] = s

        # Statistics
        try:
            precision = "{0:.3f}".format(n_tpos / (n_tpos + n_fpos))
        except ZeroDivisionError:
            precision = "?"
        try:
            recall = "{0:.3f}".format(n_tpos / (n_tpos + n_fneg))
        except ZeroDivisionError:
            recall = "?"

        # Write to csv
        if self.summary == "-":
            handle = stdout
        else:
            handle = open(self.summary, 'w', newline='')
        try:
            header = ["Workflow"] + sorted([shorten(t) for t in tasks])
            w = csv.DictWriter(handle, fieldnames=header)
            w.writeheader()
            for wf in sorted(workflows):
                w.writerow(results[wf])
            w.writerow({header[0]: "Precision:", header[1]: precision})
            w.writerow({header[0]: "Recall:", header[1]: recall})
        finally:
            if handle is not stdout:
                handle.close()

    def main(self, *QUERY_FILE):
        if not QUERY_FILE:
            self.help()
            return 1
        else:

            # Windows does not interpret asterisks as globs
            if platform.system() == 'Windows':
                QUERY_FILE = tuple(globbed
                    for original in QUERY_FILE
                    for globbed in glob(original))

            if self.server:
                self.store = TransformationStore.backend(self.backend,
                    self.server, cred=self.cred)
            else:
                self.store = None

            # Parse tasks and optionally run associated queries
            tasks = [self.evaluate(task_file) for task_file in QUERY_FILE]

            self.write(*tasks)
            if self.summary:
                self.summarize(*tasks)


def main():
    CLI.run()


if __name__ == '__main__':
    CLI.run()
