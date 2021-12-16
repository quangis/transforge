"""
This module extends transformation algebras such that its expressions can be
parsed as RDF graphs.
"""

from __future__ import annotations

from transformation_algebra.type import Type, TypeOperation, TypeVariable, \
    Function, TypeOperator, TypeInstance, TypingError
from transformation_algebra.expr import \
    Expr, Operation, Application, Abstraction, Source, Operator, Variable, \
    ApplicationError
from transformation_algebra.lang import Language

from itertools import chain
from rdflib import URIRef, Graph, Namespace, BNode, Literal
from rdflib.term import Node
from rdflib.namespace import RDF, RDFS, ClosedNamespace

from typing import Optional

TA = Namespace("https://github.com/quangis/transformation-algebra#")
TEST = Namespace("https://example.com/#")


class LanguageNamespace(ClosedNamespace):
    """
    A algebra-aware namespace for rdflib. That is, it allows URIs to be written
    as `NS[f]` for an operation or base type `f`. It is also closed: it fails
    when referencing URIs for types or operations that are not part of the
    relevant transformation algebra.
    """

    def __new__(cls, uri, alg: Language):
        terms = chain(
            alg.operators.keys(),
            alg.types.keys()
        )
        rt = super().__new__(cls, uri, terms)
        return rt

    def term(self, value) -> URIRef:
        if isinstance(value, (Operator, TypeOperator)):
            return super().term(value.name)
        else:
            return super().term(value)


class TransformationGraph(Graph):
    """
    A transformation graph represents expressions of a transformation algebra
    as an RDF graph.
    """

    def __init__(self, language: Language,
            with_types: bool = False,
            with_steps: bool = False,
            with_labels: bool = False,
            with_kinds: bool = False,
            *nargs, **kwargs):

        super().__init__(*nargs, **kwargs)

        self.language = language
        self.with_types = with_types
        self.with_labels = with_labels
        self.with_steps = with_steps
        self.with_kinds = with_steps

        self.type_nodes: dict[TypeInstance, Node] = dict()
        self.expr_nodes: dict[Expr, Node] = dict()

        self.bind("ta", TA)
        # self.bind("test", self.namespace)

    def add_vocabulary(self) -> None:
        """
        Add the RDF vocabulary for describing expressions in terms of the types
        and operations defined for this transformation algebra.
        """
        ns = self.language.namespace

        # Add type operators to the vocabulary
        for t in self.language.types.values():
            if t.arity > 0:
                current_uri = ns[t.name]

                if self.with_kinds:
                    self.add((current_uri, RDF.type, TA.Type))
                    self.add((current_uri, RDFS.subClassOf, RDF.Seq))

                if self.with_labels:
                    self.add((current_uri, RDFS.label, Literal(str(t))))
            else:
                previous_uri = None
                current: Optional[TypeOperator] = t
                while current:
                    current_uri = ns[current.name]

                    if self.with_labels:
                        self.add((current_uri, RDFS.label, Literal(str(t))))

                    if self.with_kinds:
                        self.add((current_uri, RDF.type, TA.Type))

                    if previous_uri:
                        self.add((previous_uri, RDFS.subClassOf, current_uri))

                    previous_uri = current_uri
                    current = current.supertype

        # Add operations to the vocabulary
        for op in self.language.operators.values():
            node = ns[op.name]

            if self.with_kinds:
                self.add((node, RDF.type, TA.Operation))

            if self.with_labels:
                self.add((node, RDFS.label, Literal(str(op.name))))
                if op.description:
                    self.add((node, RDFS.comment, Literal(op.description)))

    def add_type(self, type: Type) -> Node:
        """
        Translate and add an RDF representation of the given type. Return the
        top-level node.
        """
        t = type.instance().normalize()

        try:
            return self.type_nodes[t]
        except KeyError:
            if isinstance(t, TypeOperation):
                if t.params:
                    node = BNode()
                    self.add((node, RDFS.subClassOf,
                        self.language.namespace[t._operator.name]))

                    for i, param in enumerate(t.params, start=1):
                        self.add((node, RDF[f"_{i}"], self.add_type(param)))

                    if self.with_labels:
                        self.add((node, RDFS.label, Literal(str(t))))
                else:
                    node = self.language.namespace[t._operator.name]
            else:
                assert isinstance(t, TypeVariable)
                node = BNode()

                if self.with_labels:
                    self.add((node, RDFS.label, Literal(str(t))))

                if self.with_kinds:
                    self.add((node, RDF.type, TA.TypeVariable))

            self.type_nodes[t] = node
            return node

    def add_expr(self, expr: Expr, root: Node, current: Optional[Node] = None,
            sources: dict[str, Node | Expr] = {}) -> Node:
        """
        Translate and add the given expression to a representation in RDF and
        add it to the given graph. Inputs that match the labels in the
        expression are appropriately attached, either as data sources or as
        previously added input expressions.

        This is a lossy conversion, because the order of arguments and the
        exact structure of functions-as-arguments is not preserved.
        """
        assert isinstance(expr.type, TypeInstance)

        if expr in self.expr_nodes:
            return self.expr_nodes[expr]

        current = current or BNode()

        if isinstance(expr, Source):

            if expr.label:
                try:
                    source = sources[expr.label]
                except KeyError as e:
                    msg = f"no input node named '{expr.label}'"
                    raise RuntimeError(msg) from e
            else:
                source = None

            if isinstance(source, Node) or source is None:
                expr.type = expr.type.fix(prefer_lower=False)
                if source:
                    self.add((current, TA.source, source))

                if self.with_types:
                    self.add((current, RDF.type, self.add_type(expr.type)))

                if self.with_labels:
                    self.add((current, RDFS.label,
                        Literal(f"source {expr.type.output()}")))

                if self.with_kinds:
                    self.add((current, RDF.type, TA.SourceData))

            else:
                assert isinstance(source, Expr)
                try:
                    # TODO unification happens as we translate to
                    # RDF, which means some might be outdated
                    # instead match(subtype=False)?
                    source.type.unify(expr.type, subtype=True)
                except TypingError as e:
                    raise SourceError(expr, source) from e

                assert source in self.expr_nodes, \
                    "should already have been calculated in add_workflow"
                current = self.expr_nodes[source]

        elif isinstance(expr, Operation):
            assert not expr.operator.definition, \
                f"{expr.operator} should be a primitive"

            datatype = expr.type.output()

            if self.with_types:
                self.add((current, RDF.type, self.add_type(datatype)))

            if self.with_labels:
                self.add((current, RDFS.label,
                    Literal(f"{datatype} via {expr.operator.name}")))

            if self.with_kinds:
                self.add((current, RDF.type, TA.TransformedData))

            self.add((current, TA.via,
                self.language.namespace[expr.operator.name]))

        else:
            assert isinstance(expr, Application)
            assert not isinstance(expr.f, Abstraction), \
                "abstractions can only occur as parameters, otherwise " \
                "they could have been β-reduced"

            # A function application with multiple arguments will look
            # something like `(f x) y`. By recursively adding the function part
            # to the current node, and attaching a new node for the parameter
            # part, we eventually get a node for the function to which nodes
            # for all parameters are attached.
            f = self.add_expr(expr.f, root, current, sources)

            # For simple data, we can simply attach the node for the parameter
            # directly. But when the parameter is a *function*, we need to be
            # careful.
            # Conceptually, `x` will be some operation that produces data while
            # inside the black box of `f`. Of course, `x` may be used inside
            # `f` many times, producing different data instances, but that is
            # irrelevant for our purposes: we care only that `f` uses at least
            # some input of that type.
            # We don't know exactly what happens internally otherwise --- only
            # that there is some process that passes data to `x` and that may
            # synthesize its own input from any other data known to `f`.
            # Now, if `x` is an abstraction --- an anonymous operation with an
            # inner structure --- then its body represents the data that `f`
            # may use. For primitive functions, we add a data node between `x`
            # and `f` to represent this data. The mysterious input to `x` (or,
            # in the case of abstractions, the values of `x`'s parameters) is
            # then a single unspecified internal data node produced by an
            # internal operation. All inputs to `f` should be made inputs to
            # this operation also.
            current_internal = None
            if expr.x.type.operator == Function:
                internal = BNode()
                current_internal = internal
                self.add((f, TA.internal, internal))

                if self.with_kinds:
                    self.add((internal, RDF.type, TA.InternalData))

                if self.with_steps:
                    self.add((root, TA.step, internal))

                if self.with_labels:
                    self.add((internal, RDFS.label, Literal("internal")))

                if isinstance(expr.x, Abstraction):
                    for p in expr.x.params:
                        self.expr_nodes[p] = internal

                    x = self.add_expr(expr.x.body, root, BNode(), sources)
                else:
                    x = self.add_expr(expr.x, root, BNode(), sources)
                    self.add((internal, TA.feeds, x))
            else:
                x = self.add_expr(expr.x, root, BNode(), sources)
            self.add((x, TA.feeds, f))

            # If `x` has internal operations of its own, then those inner
            # operations should be fed by the current (outer) internal
            # operation, which has access to additional parameters that may be
            # used by the inner one. See issues #37 and #41.
            if current_internal:
                for internal in self.objects(x, TA.internal):
                    self.add((current_internal, TA.feeds, internal))

            # Every operation that is internal to `f` should also take `x`'s
            # output as input
            for internal in self.objects(f, TA.internal):
                if internal != current_internal:
                    self.add((x, TA.feeds, internal))

            # ... and every input to `f` should be an input to this internal
            # operation
            if current_internal:
                for data_input in self.subjects(TA.feeds, f):
                    if x != data_input:
                        self.add((data_input, TA.feeds, current_internal))

        if self.with_steps:
            self.add((root, TA.step, current))

        return current

    def add_workflow(self, root: Node,
            steps: dict[Expr, list[Node | Expr]]) -> Node:
        """
        Convert a workflow to a full transformation graph by converting its
        individual steps to representations of expressions in RDF and combining
        them. Return the final 'output' node of the expression.

        A workflow consists of a collection of algebra expressions for each
        step (e.g. application of a tool), paired with the inputs to those
        expressions. Every input must be either a source node or an expression.
        """
        # TODO cycles can occur

        # One of the tool expressions must be 'last': it represents the tool
        # finally producing the output and so isn't an input to another.
        all_inputs = set(i for s in steps.values() for i in s)
        final_expressions = [e for e in steps.keys() if e not in all_inputs]
        assert len(final_expressions) == 1, "must have exactly one final step"
        top_level_expression = final_expressions[0]

        def to_expr_node(expr: Expr) -> Node:
            try:
                return self.expr_nodes[expr]
            except KeyError:
                for source in steps[expr]:
                    if isinstance(source, Expr):
                        to_expr_node(source)

                self.expr_nodes[expr] = node = self.add_expr(
                    expr, root, sources={
                        f"x{i}": source
                        for i, source in enumerate(steps[expr], start=1)
                    }
                )
                return node

        result_node = to_expr_node(top_level_expression)
        self.add((root, TA.result, result_node))

        if self.with_kinds:
            self.add((root, RDF.type, TA.Transformation))

        return result_node


# Errors #####################################################################

class SourceError(Exception):
    "Raised when a source"

    def __init__(self, source: Expr, attachment: Expr):
        self.source = source
        self.attachment = attachment

    def __str__(self) -> str:
        assert self.__cause__, "must caused by another error"
        return f"Could not attach {self.attachment} to a source of type " \
            f"{self.source.type}: " \
            f"{self.__cause__}"
