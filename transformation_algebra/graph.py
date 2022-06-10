"""
This module extends transformation algebras such that its expressions can be
parsed as RDF graphs.
"""

from __future__ import annotations

from transformation_algebra.type import Type, TypeOperation, TypeVariable, \
    Function, Product, Unit, TypeInstance
from transformation_algebra.expr import \
    Expr, Operation, Application, Abstraction, Source
from transformation_algebra.lang import Language

from itertools import chain, count
from rdflib import Graph, Namespace, BNode, Literal
from rdflib.term import Node
from rdflib.namespace import RDF, RDFS

from typing import Iterator, Iterable

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
            with_intermediate_types: bool | None = None,
            with_output: bool | None = None,
            with_inputs: bool | None = None,
            with_membership: bool | None = None,
            with_type_parameters: bool | None = None,
            with_labels: bool | None = None,
            with_classes: bool | None = None,
            with_transitive_closure: bool | None = None,
            with_noncanonical_types: bool | None = None,
            passthrough: bool = True,
            *nargs, **kwargs):

        super().__init__(*nargs, **kwargs)

        def default(switch: bool | None, inherit: bool = not minimal) -> bool:
            return inherit if switch is None else switch

        self.language = language
        self.passthrough = passthrough
        self.with_operators = default(with_operators)
        self.with_types = default(with_types)
        self.with_intermediate_types = default(with_intermediate_types,
            self.with_types)
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

        self.identifiers: Iterator[int] | None = None

        for t in self.language.taxonomy:
            self.type_nodes[t] = self.language.namespace[
                t.text(sep="-", lparen="-", rparen="", prod="")]

        self.bind("", TA)

    def traverse(self, predicate: Node, start: Node,
            path: list[Node] = [],
            visited: set[Node] | None = None) -> Iterator[Node]:
        """
        Follow starting node along the given predicate in a depth-first manner.
        Raise an error when a cycle is detected.
        """
        if start in path:
            raise ValueError("cycle")
        visited = visited or set()
        if start not in visited:
            visited.add(start)
            yield start
            yield from chain.from_iterable(
                self.traverse(predicate, n, path + [start], visited)
                for n in self.objects(start, predicate))

    def ref(self) -> str:
        return f"{next(self.identifiers)}. " if self.identifiers else ""

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
                    TA.Product if root._operator is Product else
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
                    if t._operator is Product:
                        self.add((node, RDFS.subClassOf, TA.Product))
                    else:
                        self.add((node, RDFS.subClassOf,
                            self.language.namespace[t._operator.name]))

                    if self.with_type_parameters:
                        for i, param in enumerate(t.params, start=1):
                            self.add((node, RDF[f"_{i}"], self.add_type(param)))

                    if self.with_labels:
                        self.add((node, RDFS.label, Literal(str(t))))
                else:
                    if t._operator is Unit:
                        node = TA.Unit
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
            current: Node | None = None, intermediate: bool = False) -> Node:
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
            intermediate = False

            # Once we arrive at a data source, we just assume its type to be
            # the most general possible. This is so that we don't have to give
            # a specific type for data source 1 when we have a tool with
            # expression `f (1: A)`. See also issue #72. The type inference has
            # already run its course at this point, so it should be safe.
            expr.type = expr.type.fix(prefer_lower=False).normalize()

            # Sources are always saved since they must always be reused when
            # used multiple times, especially if passthrough is disabled
            self.expr_nodes[expr] = current

            if self.with_inputs:
                self.add((root, TA.input, current))

            if self.with_types and (self.with_noncanonical_types or
                    expr.type in self.language.taxonomy):

                type_node = self.add_type(expr.type)
                self.add((current, RDF.type, type_node))

                if self.with_membership:
                    self.add((root, TA.contains, type_node))

            if self.with_labels:
                self.add((current, RDFS.label,
                    Literal(f"{self.ref()}{expr.type}")))

            if self.with_classes:
                self.add((current, RDF.type, TA.SourceConcept))

        elif isinstance(expr, Operation):
            # assert not expr.operator.definition, \
            #     f"{expr.operator} should be a primitive"

            output_type = expr.type.output().normalize()

            if self.with_operators:
                op_node = self.language.namespace[expr.operator.name]
                self.add((current, TA.via, op_node))

                if self.with_membership:
                    self.add((root, TA.contains, op_node))

            if (self.with_types
                    and (self.with_noncanonical_types or
                        output_type in self.language.taxonomy)
                    and (self.with_intermediate_types or not intermediate)):

                type_node = self.add_type(output_type)
                self.add((current, RDF.type, type_node))

                if self.with_membership:
                    self.add((root, TA.contains, type_node))

            if self.with_labels:
                if ((self.with_intermediate_types or not intermediate)
                        and (self.with_noncanonical_types or
                        output_type in self.language.taxonomy)):
                    type_str = str(output_type)
                else:
                    type_str = f"{output_type} (non-canonical)"

                self.add((current, RDFS.label, Literal(
                    f"{self.ref()}{type_str} via {expr.operator.name}")))

            if self.with_classes:
                self.add((current, RDF.type, TA.TransformedConcept))

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
            f = self.add_expr(expr.f, root, current, intermediate=intermediate)

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
                    self.add((internal, RDF.type, TA.InternalConcept))

                if self.with_labels:
                    self.add((internal, RDFS.label, Literal(
                        "(internal)")))

                if isinstance(expr.x, Abstraction):
                    for p in expr.x.params:
                        self.expr_nodes[p] = internal

                    x = self.add_expr(expr.x.body, root, BNode(),
                        intermediate=True)
                else:
                    x = self.add_expr(expr.x, root, BNode(), intermediate=True)
                    self.add((internal, TA.to, x))
            else:
                x = self.add_expr(expr.x, root, BNode(), intermediate=True)
            self.add((x, TA.to, f))

            # If `x` has internal operations of its own, then those inner
            # operations should be fed by the current (outer) internal
            # operation, which has access to additional parameters that may be
            # used by the inner one. See issues #37 and #41.
            if current_internal:
                for internal in self.objects(x, TA.internal):
                    self.add((current_internal, TA.to, internal))

            # Every operation that is internal to `f` should also take `x`'s
            # output as input
            for internal in self.objects(f, TA.internal):
                if internal != current_internal:
                    self.add((x, TA.to, internal))

            # ... and every input to `f` should be an input to this internal
            # operation
            if current_internal:
                for f_input in self.subjects(TA.to, f):
                    if x != f_input:
                        self.add((f_input, TA.to, current_internal))

        return current

    def add_workflow(self, root: Node,
            tool_apps: dict[Node, tuple[str, list[Node]]],
            sources: set[Node] = set()) -> dict[Node, Node]:
        """
        Convert a workflow to a full transformation graph by converting its
        individual workflow steps to representations of expressions in RDF.
        Return a mapping between tool application/source nodes and their
        expression nodes.

        A workflow consists of a dictionary of transformation expressions for
        each step (e.g. application of a tool), paired with the inputs to those
        expressions. Every input must be either a source node or a node for the
        output of another step.
        """

        self.identifiers = count(start=1)

        # One of the steps must be 'last': it represents the tool application
        # that finally produces the output and so isn't an input to another.
        all_inputs = set(i for _, inputs in tool_apps.values() for i in inputs)
        try:
            final_tool_app, = [app for app in tool_apps.keys()
                if app not in all_inputs]
        except ValueError:
            raise RuntimeError("must have exactly one final tool application")

        # 1. Construct expressions for each wfnode step (source or tool
        # application), possibly using expressions from previous steps
        exprs: dict[Node, Expr] = dict()

        # If passthrough is disabled, we record connections between sources and
        # the expressions they are derived from (which may be more specific)
        indirection: dict[Source, Expr] = dict()

        for source in sources:
            exprs[source] = Source()

        def wfnode2expr(wfnode: Node) -> Expr:
            try:
                return exprs[wfnode]
            except KeyError:
                tool_description, input_nodes = tool_apps[wfnode]
                input_exprs = [wfnode2expr(n) for n in input_nodes]
                if not self.passthrough:
                    for i in range(len(input_exprs)):
                        if input_nodes[i] not in sources:
                            e = input_exprs[i]
                            s = input_exprs[i] = Source()
                            indirection[s] = e
                exprs[wfnode] = expr = self.language.parse(tool_description,
                    *input_exprs)
                return expr

        # 2. Convert individual transformation expressions to nodes and add
        # them. We must do this in the proper order, so that expression nodes
        # for the same tool/source nodes get saved in self.expr_nodes and
        # reused rather than regenerated
        def wfnode2tfmnode(wfnode: Node) -> Node:
            expr = wfnode2expr(wfnode)
            try:
                tfmnode = self.expr_nodes[expr]
            except KeyError:
                if wfnode not in sources:
                    for input_wfnode in tool_apps[wfnode][1]:
                        # Running this has the side effect that the wfnode for
                        # every input will already have been added
                        wfnode2tfmnode(input_wfnode)
                self.expr_nodes[expr] = tfmnode = self.add_expr(expr, root)
            return tfmnode

        result_node = wfnode2tfmnode(final_tool_app)

        # If passthrough is disabled, connections between the outputs of one
        # tool and the inputs of the next are not established; do that now
        for source_expr, ref_expr in indirection.items():
            src_tfmnode = self.add_expr(source_expr, root)
            ref_tfmnode = self.expr_nodes[ref_expr]
            self.add((ref_tfmnode, TA.to, src_tfmnode))

        for source in sources:
            self.add((wfnode2tfmnode(source), TA.origin, source))

        if self.with_output:
            self.add((root, TA.output, result_node))

        if self.with_classes:
            self.add((root, RDF.type, TA.Transformation))

        self.identifiers = None  # reset the identifiers

        return {wfnode: self.expr_nodes[expr]
            for wfnode, expr in exprs.items()}

    def parse_shortcuts(self, remove: bool = True) -> None:
        """
        For convenience, types and operators may be specified as string
        literals in RDF using the `lang:type` and `lang:via` predicates. This
        method automatically parses these strings and replaces them with
        `ta:type` and `ta:via` predicates with the corresponding nodes as
        object. Example:

            [] cct:type "R(Obj, Reg)".

        Becomes:

            [] ta:type cct:R-Obj-Reg.
        """
        ns = self.language.namespace
        # TODO handle collections of types/operators
        for subj, obj in self[:ns.type:]:
            type = self.language.parse_type(str(obj))
            node = self.type_nodes[type]
            self.add((subj, RDF.type, node))

        for subj, obj in self[:ns.via:]:
            operator = self.language.parse_operator(str(obj))
            self.add((subj, TA.via, ns[operator.name]))

        if remove:
            self.remove((None, ns.type, None))
            self.remove((None, ns.via, None))
