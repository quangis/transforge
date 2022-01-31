"""
This module extends transformation algebras such that its expressions can be
parsed as RDF graphs.
"""

from __future__ import annotations

from transformation_algebra.type import Type, TypeOperation, TypeVariable, \
    Function, TypeInstance
from transformation_algebra.expr import \
    Expr, Operation, Application, Abstraction, Source
from transformation_algebra.lang import Language

from itertools import chain
from rdflib import Graph, Namespace, BNode, Literal
from rdflib.term import Node
from rdflib.namespace import RDF, RDFS

TA = Namespace("https://github.com/quangis/transformation-algebra#")
TEST = Namespace("https://example.com/#")


class TransformationGraph(Graph):
    """
    A transformation graph represents expressions of a transformation algebra
    as an RDF graph.
    """

    def __init__(self, language: Language,
            minimal: bool = False,
            with_operators: bool | None = None,
            with_types: bool | None = None,
            with_output: bool | None = None,
            with_inputs: bool | None = None,
            with_membership: bool | None = None,
            with_type_parameters: bool | None = None,
            with_labels: bool | None = None,
            with_classes: bool | None = None,
            with_transitive_closure: bool | None = None,
            with_noncanonical_types: bool | None = None,
            *nargs, **kwargs):

        super().__init__(*nargs, **kwargs)

        def default(switch: bool | None) -> bool:
            assert isinstance(minimal, bool)
            return (not minimal) if switch is None else switch

        self.language = language
        self.with_operators = default(with_operators)
        self.with_types = default(with_types)
        self.with_labels = default(with_labels)
        self.with_output = default(with_output)
        self.with_inputs = default(with_inputs)
        self.with_membership = default(with_membership)
        self.with_type_parameters = default(with_type_parameters)
        self.with_classes = default(with_classes)
        self.with_transitive_closure = default(with_transitive_closure)
        self.with_noncanonical_types = default(with_noncanonical_types)

        self.type_nodes: dict[TypeInstance, Node] = dict()
        self.expr_nodes: dict[Expr, Node] = dict()

        for t in self.language.taxonomy:
            self.type_nodes[t] = self.language.namespace[
                t.text(sep=".", lparen="_", rparen="")]

        self.bind("ta", TA)
        self.bind("lang", self.language.namespace)

    def add_vocabulary(self) -> None:
        """
        Add the RDF vocabulary for describing expressions in terms of the types
        and operations defined for this transformation algebra.
        """
        self.add_taxonomy()
        self.add_operators()

    def add_taxonomy(self) -> None:
        taxonomy = self.language.taxonomy

        for t, subtypes in taxonomy.items():
            uri = self.type_nodes[t]

            if self.with_classes:
                self.add((uri, RDF.type, TA.Type))

            if self.with_labels:
                self.add((uri, RDFS.label, Literal(t.text())))

            if self.with_type_parameters:
                for i, param in enumerate(t.params, start=1):
                    self.add((uri, RDF[f"_{i}"], self.type_nodes[param]))

            for s in subtypes:
                self.add((self.type_nodes[s], RDFS.subClassOf, uri))

        # Connect top-level type nodes (i.e. compound type operators) to the
        # roots of their respective trees
        for root in set(taxonomy) - set.union(*taxonomy.values()):
            if root._operator.arity > 0:
                self.add((self.type_nodes[root], RDFS.subClassOf,
                    self.language.namespace[root._operator.name]))

        if self.with_transitive_closure:
            nodes = set(chain(
                (self.type_nodes[t] for t in taxonomy),
                (self.language.namespace[op] for op in self.language.types)
            ))
            for node in nodes:
                for desc in self.transitive_subjects(RDFS.subClassOf, node):
                    self.add((desc, RDFS.subClassOf, node))

    def add_operators(self) -> None:
        for op in self.language.operators.values():
            node = self.language.namespace[op.name]

            if self.with_classes:
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

                    if self.with_type_parameters:
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

                if self.with_classes:
                    self.add((node, RDF.type, TA.TypeVariable))

            self.type_nodes[t] = node
            return node

    def add_expr(self, expr: Expr, root: Node,
            current: Node | None = None) -> Node:
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
            # Once we arrive at a data source, we just assume its type to be
            # the most general possible. This is so that we don't have to give
            # a specific type for data source 1 when we have a tool with
            # expression `f (1: A)`. See also issue #72. The type inference has
            # already run its course at this point, so it should be safe.
            expr.type = expr.type.fix(prefer_lower=False)

            if self.with_inputs:
                self.add((root, TA.input, current))

            if self.with_types and (self.with_noncanonical_types or
                    expr.type in self.language.taxonomy):

                type_node = self.add_type(expr.type)
                self.add((current, RDF.type, type_node))

                if self.with_membership:
                    self.add((root, TA.member, type_node))

            if self.with_labels:
                self.add((current, RDFS.label,
                    Literal(f"{expr.type} (source data)")))

            if self.with_classes:
                self.add((current, RDF.type, TA.SourceData))

        elif isinstance(expr, Operation):
            # assert not expr.operator.definition, \
            #     f"{expr.operator} should be a primitive"

            datatype = expr.type.output().normalize()

            if self.with_operators:
                op_node = self.language.namespace[expr.operator.name]
                self.add((current, TA.via, op_node))

                if self.with_membership:
                    self.add((root, TA.member, op_node))

            if self.with_types and (self.with_noncanonical_types or
                    datatype in self.language.taxonomy):
                type_node = self.add_type(datatype)
                self.add((current, RDF.type, type_node))

                if self.with_membership:
                    self.add((root, TA.member, type_node))

            if self.with_labels:
                self.add((current, RDFS.label,
                    Literal(f"{datatype} (via {expr.operator.name})")))

            if self.with_classes:
                self.add((current, RDF.type, TA.TransformedData))

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
            f = self.add_expr(expr.f, root, current)

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

                if self.with_classes:
                    self.add((internal, RDF.type, TA.InternalData))

                if self.with_labels:
                    self.add((internal, RDFS.label, Literal(
                        "unknown (internal)")))

                if isinstance(expr.x, Abstraction):
                    for p in expr.x.params:
                        self.expr_nodes[p] = internal

                    x = self.add_expr(expr.x.body, root, BNode())
                else:
                    x = self.add_expr(expr.x, root, BNode())
                    self.add((internal, TA.feeds, x))
            else:
                x = self.add_expr(expr.x, root, BNode())
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
            steps: dict[Node, tuple[str, list[Node]]],
            sources: set[Node] = set()) -> Node:
        """
        Convert a workflow to a full transformation graph by converting its
        individual steps to representations of expressions in RDF and combining
        them. Return the final 'output' node of the expression.

        A workflow consists of a collection of algebra expressions for each
        step (e.g. application of a tool), paired with the inputs to those
        expressions. Every input must be either a source node or an expression.
        """

        # One of the steps must be 'last': it represents the tool finally
        # producing the output and so isn't an input to another.
        all_inputs = set(i for tool in steps.values() for i in tool[1])
        try:
            final_tool, = [e for e in steps.keys() if e not in all_inputs]
        except ValueError:
            raise RuntimeError("must have exactly one final step")

        # 1. Construct expressions for each step, possibly using expressions
        # from previous steps
        step_exprs: dict[Node, Expr] = dict()

        def step2expr(step: Node) -> Expr:
            try:
                return step_exprs[step]
            except KeyError:
                string, input_nodes = steps[step]
                step_exprs[step] = expr = self.language.parse(string,
                    *(step2expr(n) for n in input_nodes))
                return expr

        for node in sources:
            step_exprs[node] = Source(type=TypeVariable())

        step2expr(final_tool)

        # 2. Convert tool application & source nodes to expression nodes. We
        # must do this in the proper order, so that expression nodes for the
        # same tool/source nodes get saved in self.expr_nodes and reused rather
        # than regenerated
        def step2node(step: Node) -> Node:
            expr = step2expr(step)
            try:
                node = self.expr_nodes[expr]
            except KeyError:
                if step not in sources:
                    for input_step in steps[step][1]:
                        step2node(input_step)
                self.expr_nodes[expr] = node = self.add_expr(expr, root)

            return node

        result_node = step2node(final_tool)

        for node in sources:
            expr_node = step2node(node)
            self.add((expr_node, TA.origin, node))

        if self.with_output:
            self.add((root, TA.output, result_node))

        if self.with_classes:
            self.add((root, RDF.type, TA.Transformation))

        return result_node
