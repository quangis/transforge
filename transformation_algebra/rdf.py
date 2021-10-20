"""
This module extends transformation algebras such that its expressions can be
parsed as RDF graphs.
"""

from __future__ import annotations

from transformation_algebra import error
from transformation_algebra.type import Type, TypeOperation, TypeVariable, \
    Function, TypeOperator, TypeInstance
from transformation_algebra.expr import \
    Expr, Operation, Application, Abstraction, Source, Operator, Variable
from transformation_algebra.alg import TransformationAlgebra

from itertools import count, chain
from rdflib import URIRef, Graph, Namespace, BNode, Literal
from rdflib.term import Node
from rdflib.namespace import RDF, RDFS, ClosedNamespace
from rdflib.plugins import sparql

from typing import Dict, Union, Iterator, Optional, Tuple

TA = Namespace("https://github.com/quangis/transformation-algebra#")


class AlgebraNamespace(ClosedNamespace):
    """
    A algebra-aware namespace for rdflib. That is, it allows URIs to be written
    as `NS[f]` for an operation or base type `f`. It is also closed: it fails
    when referencing URIs for types or operations that are not part of the
    relevant transformation algebra.
    """

    def __new__(cls, uri, alg: TransformationAlgebra):
        terms = chain(
            alg.definitions.keys(),
            (t.name for t in alg.types)
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

    def __init__(self, algebra: TransformationAlgebra,
            namespace: AlgebraNamespace,
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

        self.type_nodes: Dict[TypeInstance, Node] = dict()
        self.var_nodes: Dict[Variable, Node] = dict()

        self.bind("ta", TA)
        # self.bind("test", self.namespace)

    @staticmethod
    def vocabulary(algebra: TransformationAlgebra, namespace:
            AlgebraNamespace) -> Graph:
        """
        Produce an RDF vocabulary for describing expressions in terms of the
        types and operations defined for this transformation algebra.
        """
        vocab = TransformationGraph(algebra, namespace)

        # Add type operators to the vocabulary
        for t in algebra.types:
            if t.arity > 0:
                current_uri = namespace[t]
                vocab.add((current_uri, RDF.type, TA.Type))
                vocab.add((current_uri, RDFS.subClassOf, RDF.Seq))
                vocab.add((current_uri, RDFS.label, Literal(str(t))))
            else:
                previous_uri = None
                current: Optional[TypeOperator] = t
                while current:
                    current_uri = namespace[current]
                    vocab.add((current_uri, RDFS.label, Literal(str(t))))
                    vocab.add((current_uri, RDF.type, TA.Type))
                    if previous_uri:
                        vocab.add((previous_uri, RDFS.subClassOf, current_uri))
                    previous_uri = current_uri
                    current = current.supertype

        # Add operations to the vocabulary
        for d in algebra.definitions.values():
            node = namespace[d]
            type_node = TA.Operation
            vocab.add((node, RDF.type, type_node))
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
                        self.namespace[t._operator]))

                    for i, param in enumerate(t.params, start=1):
                        self.add((node, RDF[f"_{i}"], self.add_type(param)))

                    if self.include_labels:
                        self.add((node, RDFS.label, Literal(str(t))))
                else:
                    node = self.namespace[t._operator]
            else:
                assert isinstance(t, TypeVariable)
                # node = BNode()

                # if self.include_labels:
                #     self.add((node, RDFS.label, Literal(str(t))))

                # if self.include_kinds:
                #     self.add((node, RDF.type, TA.TypeVariable))

            self.type_nodes[t] = node
            return node

    def add_expr(self, expr: Expr, root: Node, current: Optional[Node] = None,
            sources: Dict[str, Union[Node, Tuple[Node, Expr]]] = {}) -> Node:
        """
        Translate and add the given expression to a representation in RDF and
        add it to the given graph. Inputs that match the labels in the
        expression are appropriately attached, either as data sources or as
        input expressions.

        This is a lossy conversion, because the order of arguments and the
        exact structure of functions-as-arguments is not preserved.
        """
        assert isinstance(expr.type, TypeInstance)

        if isinstance(expr, Variable):
            assert expr in self.var_nodes
            return self.var_nodes[expr]

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

            if self.include_labels:
                self.add((current, RDFS.label,
                    Literal(f"source : {datatype}")))

            if isinstance(expr, Operation):
                assert expr.operator.is_primitive(), \
                    f"{expr.operator} is not a primitive"

                if self.include_kinds:
                    self.add((current, RDF.type, TA.TransformedData))

                self.add((current, TA.via, self.namespace[expr.operator]))
            else:
                assert isinstance(expr, Source)

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
                            source_node, source_expr = source
                            assert isinstance(source_node, Node) and \
                                isinstance(source_expr, Expr)
                            try:
                                # TODO unification happens as we translate to
                                # RDF, which means some might be outdated
                                # instead match(subtype=False)?
                                source_expr.type.unify(expr.type, subtype=True)
                            except error.TATypeError as e:
                                e.while_unifying(source_expr, expr)
                                raise
                            return source_node

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
                        self.var_nodes[p] = internal

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

    def add_workflow(self, root: Node, sources: set[Node],
            steps: dict[Node, tuple[Expr, list[Node]]]) -> Node:
        """
        Convert a workflow to a full transformation graph by converting its
        individual steps to representations of expressions in RDF and combining
        them. Return the final 'output' node of the expression.

        A workflow consists of:

        -   A collection of RDF nodes representing data sources.
        -   A collection of algebra expressions for each step (e.g. application
            of a tool), paired with the inputs to those expressions. Every
            input must be either a source node or a node representing another
            step from the same collection.
        """
        # TODO cycles can occur

        cache: Dict[Node, Node] = {}

        def to_expr_node(step_node: Node) -> Node:
            try:
                return cache[step_node]
            except KeyError:
                expr, inputs = steps[step_node]
                assert all(x in steps or x in sources for x in inputs), \
                    "unknown input data source"
                cache[step_node] = self.add_expr(expr, root, sources={
                    f"x{i}": (to_expr_node(x), steps[x][0])
                    if x in steps else x
                    for i, x in enumerate(inputs, start=1)
                })
                return cache[step_node]

        # One of the tool expressions must be 'last', in that it represents the
        # tool finally producing the output and so isn't an input to another.
        all_inputs = set(input_node
            for step in steps.values() for input_node in step[1])
        final_tool_node = [step_node for step_node in steps.keys()
            if step_node not in all_inputs]
        assert len(final_tool_node) == 1, \
            "workflow must have exactly one final step"

        final_expr_node = to_expr_node(final_tool_node[0])
        self.add((root, TA.result, final_expr_node))
        return final_expr_node
