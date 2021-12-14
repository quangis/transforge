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

    def __init__(self, algebra: Language, namespace: Namespace,
            include_types: bool = True,
            include_steps: bool = False,
            include_labels: bool = True,
            include_kinds: bool = False,
            *nargs, **kwargs):

        super().__init__(*nargs, **kwargs)

        self.algebra = algebra
        self.namespace = namespace
        self.include_types = include_types
        self.include_labels = include_labels
        self.include_steps = include_steps
        self.include_kinds = include_steps

        self.type_nodes: dict[TypeInstance, Node] = dict()
        self.expr_nodes: dict[Expr, Node] = dict()

        self.bind("ta", TA)
        # self.bind("test", self.namespace)

    @staticmethod
    def vocabulary(algebra: Language, namespace: Namespace) -> Graph:
        """
        Produce an RDF vocabulary for describing expressions in terms of the
        types and operations defined for this transformation algebra.
        """
        vocab = TransformationGraph(algebra, namespace)
        vocab.include_labels = False  # TODO

        # Add type operators to the vocabulary
        for t in algebra.types.values():
            if t.arity > 0:
                current_uri = namespace[t.name]

                if vocab.include_kinds:
                    vocab.add((current_uri, RDF.type, TA.Type))
                    vocab.add((current_uri, RDFS.subClassOf, RDF.Seq))

                if vocab.include_labels:
                    vocab.add((current_uri, RDFS.label, Literal(str(t))))
            else:
                previous_uri = None
                current: Optional[TypeOperator] = t
                while current:
                    current_uri = namespace[current.name]

                    if vocab.include_labels:
                        vocab.add((current_uri, RDFS.label, Literal(str(t))))

                    if vocab.include_kinds:
                        vocab.add((current_uri, RDF.type, TA.Type))

                    if previous_uri:
                        vocab.add((previous_uri, RDFS.subClassOf, current_uri))

                    previous_uri = current_uri
                    current = current.supertype

        # Add operations to the vocabulary
        for d in algebra.operators.values():
            node = namespace[d]

            if vocab.include_kinds:
                type_node = TA.Operation
                vocab.add((node, RDF.type, type_node))

            if vocab.include_labels:
                vocab.add((node, RDFS.label, Literal(str(d.name))))
                if d.description:
                    vocab.add((node, RDFS.comment, Literal(d.description)))

        return vocab

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
                        self.namespace[t._operator.name]))

                    for i, param in enumerate(t.params, start=1):
                        self.add((node, RDF[f"_{i}"], self.add_type(param)))

                    if self.include_labels:
                        self.add((node, RDFS.label, Literal(str(t))))
                else:
                    node = self.namespace[t._operator.name]
            else:
                assert isinstance(t, TypeVariable)
                node = BNode()

                if self.include_labels:
                    self.add((node, RDFS.label, Literal(str(t))))

                if self.include_kinds:
                    self.add((node, RDF.type, TA.TypeVariable))

            self.type_nodes[t] = node
            return node

    def add_expr(self, expr: Expr, root: Node, current: Optional[Node] = None,
            sources: dict[str, Node | Expr] = {}) -> Node:
        """
        Translate and add the given expression to a representation in RDF and
        add it to the given graph. Inputs that match the labels in the
        expression are appropriately attached, either as data sources or as
        input expressions.

        This is a lossy conversion, because the order of arguments and the
        exact structure of functions-as-arguments is not preserved.
        """
        assert isinstance(expr.type, TypeInstance)

        if expr in self.expr_nodes:
            return self.expr_nodes[expr]

        current = current or BNode()

        # always label transformation; ignore self.include_kinds because this
        # information is actually used by our queries
        self.add((root, RDF.type, TA.Transformation))

        if self.include_steps:
            self.add((root, TA.step, current))

        if isinstance(expr, (Operation, Source)):
            datatype = expr.type.output()

            if self.include_types:
                self.add((current, RDF.type, self.add_type(datatype)))

            if isinstance(expr, Operation):
                assert not expr.operator.definition, \
                    f"{expr.operator} should be a primitive"

                if self.include_labels:
                    self.add((current, RDFS.label,
                        Literal(f"{datatype} via {expr.operator.name}")))

                if self.include_kinds:
                    self.add((current, RDF.type, TA.TransformedData))

                self.add((current, TA.via, self.namespace[expr.operator.name]))
            else:
                assert isinstance(expr, Source)

                if self.include_labels:
                    self.add((current, RDFS.label,
                        Literal(f"source {datatype}")))

                if self.include_kinds:
                    self.add((current, RDF.type, TA.SourceData))

                if expr.label:
                    try:
                        source = sources[expr.label]
                    except KeyError as e:
                        msg = f"no input node named '{expr.label}'"
                        raise RuntimeError(msg) from e
                    else:
                        if isinstance(source, Node):
                            self.add((current, TA.source, source))
                        else:
                            assert isinstance(source, Expr)
                            try:
                                # TODO unification happens as we translate to
                                # RDF, which means some might be outdated
                                # instead match(subtype=False)?
                                source.type.unify(expr.type, subtype=True)
                            except TypingError as e:
                                raise SourceError(expr, source) from e
                            return self.expr_nodes[source]

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

                if self.include_kinds:
                    self.add((internal, RDF.type, TA.InternalData))

                if self.include_steps:
                    self.add((root, TA.step, internal))

                if self.include_labels:
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

        return current

    def add_workflow(self, root: Node,
            steps: dict[Expr, list[Node | Expr]]) -> Node:
        """
        Convert a workflow to a full transformation graph by converting its
        individual steps to representations of expressions in RDF and combining
        them. Return the final 'output' node of the expression.

        A workflow consists of:

        -   A collection of algebra expressions for each step (e.g. application
            of a tool), paired with the inputs to those expressions. Every
            input must be either a source node or an expression.
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

                self.expr_nodes[expr] = n = self.add_expr(expr, root, sources={
                    f"x{i}": source
                    for i, source in enumerate(steps[expr], start=1)
                })
                return n

        node = to_expr_node(top_level_expression)
        self.add((root, TA.result, node))
        return node


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
