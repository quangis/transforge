"""
Tools (which a transformation language describes) can be strung together into
*workflows*. This module describes the common interface and provides some
implementations.
"""

from __future__ import annotations

from rdflib import Graph
from rdflib.term import Node, URIRef, Literal
from abc import ABCMeta, abstractmethod
from typing import Iterator

from transforge.namespace import WF, RDF, EX
from transforge.lang import Language


# TODO: Ideally, this should be a protocol, not an abstract method. That way,
# workflow graphs do not need to explicitly inherit from it
class Workflow(object):
    """
    This is the interface that workflows must follow: they consist of
    resources, some of which are source data and some are the output data of a
    tool that takes as input some other resources. Each tool is associated with
    an expression of a transformation language.
    """

    __metaclass__ = ABCMeta

    @property
    @abstractmethod
    def root(self) -> URIRef:
        """
        The identifier of the workflow itself.
        """
        return NotImplemented

    @property
    @abstractmethod
    def sources(self) -> set[Node]:
        """
        Resources that represent source data of the workflow.
        """
        return NotImplemented

    @property
    @abstractmethod
    def tool_outputs(self) -> set[Node]:
        """
        Resources that are the output of tool applications in the workflow.
        """
        return NotImplemented

    @abstractmethod
    def inputs(self, resource: Node) -> Iterator[Node]:
        """
        Find the input resources that were used to obtain an output resource.
        """
        return NotImplemented

    @abstractmethod
    def tool(self, resource: Node) -> URIRef:
        """
        Find the tool that was applied to obtain an output resource.
        """
        return NotImplemented

    @abstractmethod
    def expression(self, tool_or_resource: Node) -> str:
        """
        Expression of a transformation language that is associated with a tool
        (or the resource output by an application of such a tool).
        """
        return NotImplemented

    def target(self) -> Node:
        """
        Output of the workflow, produced by the final tool application.
        """

        # One of the tool applications must be 'last': it represents the one
        # that finally produces the output and so isn't an input to any other.
        targets = self.tool_outputs \
            - set(x for o in self.tool_outputs for x in self.inputs(o))
        if len(targets) == 1:
            target, = targets
            return target
        else:
            raise ValueError("must have exactly one final tool application")

    @staticmethod
    def from_rdf(self, language: Language, graph: Graph) -> Workflow:
        """
        Construct a workflow from an RDF graph. At the moment, only workflow
        graphs described with the `WF` vocabulary are supported.
        """
        if graph.value(None, RDF.type, WF.Workflow):
            g = WorkflowGraph(language)
            g += graph
            g.refresh()
            return g
        else:
            raise ValueError("did not recognize workflow graph")


class WorkflowDict(Workflow):
    """
    A workflow expressed with the absolute minimum of information; mostly for
    testing.
    """

    def __init__(self, root: URIRef,
            tool_apps: dict[Node, tuple[str, list[Node]]],
            sources: set[Node] = set()):
        self._root = root
        self._sources = sources
        self._tool_outputs = set(tool_apps.keys())
        self._tool_apps = tool_apps

    @property
    def root(self) -> URIRef:
        return self._root

    @property
    def sources(self) -> set[Node]:
        return self._sources

    @property
    def tool_outputs(self) -> set[Node]:
        return self._tool_outputs

    def expression(self, resource: Node) -> str:
        return self._tool_apps[resource][0]

    def inputs(self, resource: Node) -> Iterator[Node]:
        return iter(self._tool_apps[resource][1])

    def tool(self, resource: Node) -> URIRef:
        return EX.SomeTool


class WorkflowGraph(Graph, Workflow):
    """
    A workflow using the <http://geographicknowledge.de/vocab/Workflow.rdf#>
    vocabulary.
    """

    def __init__(self, language: Language, tools: Graph | None = None,
            workflow: Graph | None = None):
        super().__init__()
        self.language = language
        self.tools = tools or self
        if workflow:
            self += workflow
            self.refresh()

    def refresh(self) -> None:
        root = self.value(None, RDF.type, WF.Workflow, any=False)
        if root:
            self._root = root
        else:
            raise ValueError("there must be exactly 1 Workflow in the graph")

        self._sources = set(self.objects(self.root, WF.source))
        self._tool_outputs = set()
        self._app_for = dict()

        for tool_app in self.objects(root, WF.edge):
            tool_output = self.value(tool_app, WF.output, any=False)
            assert tool_output is not None
            self._tool_outputs.add(tool_output)
            self._app_for[tool_output] = tool_app

    @property
    def root(self) -> URIRef:
        return self._root

    @property
    def sources(self) -> set[Node]:
        return self._sources

    @property
    def tool_outputs(self) -> set[Node]:
        return self._tool_outputs

    def inputs(self, resource: Node) -> Iterator[Node]:
        for p in (WF.input1, WF.input2, WF.input3):
            input_resource = self.value(self._app_for[resource], p, any=False)
            if input_resource:
                assert (input_resource in self.tool_outputs
                    or input_resource in self.sources)
                yield input_resource

    def tool(self, resource: Node) -> URIRef:
        app = self._app_for[resource]
        tool = self.value(app, WF.applicationOf, any=False)

        if not tool:
            raise ValueError(f"{app} has no associated tool")
        assert isinstance(tool, URIRef)
        return tool

    def expression(self, tool_or_resource: Node) -> str:
        if tool_or_resource in self.tool_outputs:
            tool = self.tool(tool_or_resource)
        else:
            assert isinstance(tool_or_resource, URIRef)
            tool = tool_or_resource

        expr = self.tools.value(tool, self.language.namespace.expression,
            any=False)
        if not expr:
            raise ValueError(
                f"{tool} has no algebra expression in the "
                f"{self.language.namespace} namespace")
        assert isinstance(expr, Literal)
        return str(expr)
