"""
Tools (which a transformation language describes) can be strung together into
*workflows*. This module describes the common interface and provides some
implementations.
"""

from __future__ import annotations

from rdflib import Graph
from rdflib.term import Node, URIRef
from abc import ABCMeta, abstractmethod
from typing import Iterator

from transformation_algebra.namespace import WF, RDF, EX
from transformation_algebra.lang import Language


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
    def inputs(self, app_or_output: Node) -> Iterator[Node]:
        """
        Maps tool applications (or their output resources) to the resources
        that they take as input.
        """
        return NotImplemented

    @abstractmethod
    def tool(self, app_or_output: Node) -> Node:
        """
        The tool associated with a tool application (or its output).
        """
        return NotImplemented

    @abstractmethod
    def expression(self, tool_or_app_or_output: Node) -> str:
        """
        Expression of a transformation language that is associated with a tool
        (or tool application, or the output of a tool application).
        """
        return NotImplemented

    def target(self) -> Node:
        """
        Output of the workflow, produced by the final tool application.
        """

        # One of the tool applications must be 'last': it represents the one
        # that finally produces the output and so isn't an input to any other.
        targets = self.tool_outputs
        targets -= set(x for o in self.tool_outputs for x in self.inputs(o))
        if len(targets) == 1:
            target, = targets
            return target
        else:
            raise RuntimeError("must have exactly one final tool application")


class WorkflowDict(Workflow):
    """
    A workflow expressed with the absolute minimum of information; mostly for
    testing.
    """

    def __init__(self, root: URIRef,
            tool_apps: dict[Node, tuple[str, list[Node]]],
            sources: set[Node] = set()):
        self.root = root
        self._sources = sources
        self._tool_outputs = set(tool_apps.keys())
        self._tool_apps = tool_apps

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

    def __init__(self, language: Language, *nargs, tools: Graph | None = None,
            **kwargs):
        self.language = language
        self.tools = tools or self
        super().__init__(*nargs, **kwargs)

    @property
    def root(self) -> Node:
        root = self.value(None, RDF.type, WF.Workflow, any=False)
        if not root:
            raise RuntimeError("there must be exactly 1 Workflow in the graph")
        return root

    @property
    def sources(self) -> set[Node]:
        if not hasattr(self, '_sources'):
            self._sources = set(self.objects(self.root, WF.source))
        return self._sources

    @property
    def tool_outputs(self) -> set[Node]:
        if not hasattr(self, '_tool_outputs'):
            self._tool_outputs = set(
                self.value(tool_app, WF.output, any=False)
                for tool_app in self.objects(self.root, WF.edge)
            )
        return self._tool_outputs

    def inputs(self, resource: Node) -> Iterator[Node]:
        assert resource in self.tool_outputs
        tool_app = self.value(predicate=WF.output, object=resource, any=False)
        for predicate in (WF.input1, WF.input2, WF.input3):
            input = self.value(tool_app, predicate, any=False)
            if input:
                assert input in self.tool_outputs or input in self.sources
                yield input

    def tool(self, app_or_output: Node) -> URIRef:
        if app_or_output in self.tool_outputs:
            tool_app = self.tool_app(app_or_output)
        else:
            tool_app = app_or_output
        tool = self.value(tool_app, WF.applicationOf, any=False)
        if not tool:
            raise RuntimeError(f"{tool_app} has no associated tool")
        assert isinstance(tool, URIRef)
        return tool

    def expression(self, tool_or_app_or_output: Node) -> str:
        tool = self.tool(tool_or_app_or_output)
        expr = self.tools.value(tool, self.language.namespace.expression,
            any=False)
        if not expr:
            raise RuntimeError(f"{tool} has no algebra expression")
        return str(expr)

    def tool_app(self, resource: Node) -> Node:
        """
        Find the tool application node associated with an output node.
        """
        assert resource in self.tool_outputs
        tool_app = self.value(predicate=WF.output, object=resource, any=False)
        if not tool_app:
            raise RuntimeError(f"{resource} is not output to any tool app")
        return tool_app
