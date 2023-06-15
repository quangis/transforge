"""
Command-line interface for common tasks.
"""

from __future__ import annotations

import sys
import platform
import importlib.machinery
import importlib.util
from sys import stderr
from pathlib import Path
from glob import glob

from plumbum import cli  # type: ignore
from rdflib import Graph
from rdflib.term import Literal
from rdflib.util import guess_format
from transforge.expr import ApplicationError
from transforge.lang import Language, ParseError
from transforge.graph import TransformationGraph, \
    WorkflowCompositionError
from transforge.query import TransformationQuery
from transforge.namespace import TF, EX
from transforge.workflow import WorkflowGraph
from transforge.util.store import TransformationStore
from transforge.util.utils import write_graphs


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


class WithTools:
    tools = Graph()

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

        path = self.output_path
        write_graphs(*graphs,
            file=sys.stdout if path else Path(path),
            format=self.output_format)


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
class VocabBuilder(cli.Application, WithRDF, WithServer):
    "Build vocabulary file for the transformation language"

    def main(self, LANG):
        self.language = lang(LANG)
        if self.output_format == "dot":
            vocab = TransformationGraph(self.language, minimal=True,
                with_labels=True)
            vocab.add_taxonomy()
        else:
            vocab = TransformationGraph(self.language)
            vocab.add_vocabulary()

        self.upload(vocab)
        self.write(vocab)


@CLI.subcommand("graph")
class TransformationGraphBuilder(cli.Application, WithTools, WithRDF, 
        WithServer):
    """
    Generate transformation graphs for entire workflows, concatenating the
    algebra expressions for each individual use of a tool
    """

    expressions = cli.SwitchAttr(["-e", "--expression"], list=True,
        help="Provide an expression to add to the graph.")
    skip_error = cli.Flag(["--skip-error"], default=False,
        help="Skip failed transformation graphs instead of exiting")

    def main(self, LANG, *WORKFLOW_FILE) -> int:
        self.language = lang(LANG)
        results: list[TransformationGraph] = []

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
                with_noncanonical_types=True)
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

            tg = TransformationGraph(self.language)

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
        return 0


@CLI.subcommand("query")
class QueryRunner(cli.Application, WithServer, WithRDF):
    """
    Run transformation queries against a SPARQL endpoint. If no endpoint is
    given, just output the query instead.
    """

    ordered = cli.Flag(["--ordered"],
        default=False, help="Take into account order")

    def evaluate(self, path: str) -> Graph:
        """
        Parse and run a single task and add results to the task graph.
        """
        in_graph = Graph()
        in_graph.parse(path)

        query = TransformationQuery(self.language, in_graph, by_io=True,
            by_operators=False, by_types=True,
            by_chronology=self.ordered)

        out_graph = query.graph
        out_graph.add((query.root, TF.sparql, Literal(query.sparql())))

        if self.server:
            for match in self.store.run(query):
                out_graph.add((query.root, TF.match, match))
        return out_graph

    def main(self, LANG, *QUERY_FILE):
        if not QUERY_FILE:
            self.help()
            return 1
        else:
            self.language = lang(LANG)

            # Windows does not interpret asterisks as globs
            if platform.system() == 'Windows':
                QUERY_FILE = tuple(globbed
                    for original in QUERY_FILE
                    for globbed in glob(original))

            if self.server:
                backend, _, host = self.server.partition("@")

                if self.cred:
                    user, _, password = self.cred.partition(":")
                    cred = user, password
                else:
                    cred = None

                self.store = TransformationStore.backend(backend,
                    host, cred=cred)
            else:
                self.store = None

            # Parse tasks and optionally run associated queries
            tasks = [self.evaluate(task_file) for task_file in QUERY_FILE]

            self.write(*tasks)


def main():
    CLI.run()


if __name__ == '__main__':
    main()
