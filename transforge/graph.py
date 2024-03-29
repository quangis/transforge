"""
This module extends transformation algebras such that its expressions can be
parsed as RDF graphs.
"""

from __future__ import annotations

from transforge.namespace import TF, RDF, RDFS, WF, shorten
from transforge.type import (Type, TypeOperation, Function,
    TypeInstance, TypingError)
from transforge.expr import (Expr, Operation, Application,
    Abstraction, Source)
from transforge.lang import (Language, ParseError,
    NonCanonicalTypeError, TypeAnnotationError)
from transforge.workflow import Workflow

import html
from pathlib import Path
from io import StringIO
from itertools import chain
from collections import defaultdict
from rdflib import Graph, Namespace, BNode, Literal
from rdflib.term import Node, URIRef

TEST = Namespace("https://example.com/#")


def escape(string: Literal | str | None) -> str:
    return html.escape(str(string))


class TransformationGraph(Graph):
    """
    A transformation graph represents expressions of a transformation algebra
    as an RDF graph.
    """

    def __init__(self, language: Language,
            minimal: bool = False,
            with_operators: bool | None = None,
            with_types: bool | None = None,
            with_supertypes: bool | None = None,
            with_intermediate_types: bool | None = None,
            with_output: bool | None = None,
            with_inputs: bool | None = None,
            with_workflow_origin: bool | None = None,
            with_membership: bool | None = None,
            with_membership_supertypes: bool | None = None,
            with_type_parameters: bool | None = None,
            with_labels: bool | None = None,
            with_classes: bool | None = None,
            with_transitive_closure: bool | None = None,
            with_canonical_types: bool | None = None,
            with_noncanonical_types: bool | None = None,
            with_supertype_classes: bool | None = None,
            with_dependencies: bool | None = None,
            passthrough: bool = True,
            *nargs, **kwargs):

        super().__init__(*nargs, **kwargs)

        def default(switch: bool | None, inherit: bool = not minimal) -> bool:
            return inherit if switch is None else switch

        self.uri: URIRef | None = None
        self.language = language
        self.passthrough = passthrough
        self.with_operators = default(with_operators)
        self.with_types = default(with_types)
        self.with_supertypes = default(with_supertypes)
        self.with_supertype_classes = default(with_supertype_classes, False)
        self.with_intermediate_types = default(with_intermediate_types,
            self.with_types)
        self.with_labels = default(with_labels)
        self.with_output = default(with_output)
        self.with_inputs = default(with_inputs)
        self.with_membership = default(with_membership)
        self.with_membership_supertypes = default(with_membership_supertypes)
        self.with_type_parameters = default(with_type_parameters)
        self.with_classes = default(with_classes)
        self.with_transitive_closure = default(with_transitive_closure)
        self.with_canonical_types = default(with_canonical_types, False)
        self.with_noncanonical_types = default(with_noncanonical_types)
        self.with_workflow_origin = default(with_workflow_origin)
        self.with_dependencies = default(with_dependencies)

        self.type_nodes: dict[TypeInstance, Node] = dict()
        self.expr_nodes: dict[Expr, Node] = dict()

        self.bind("tf", TF)
        if self.language.prefix and self.language.namespace:
            self.bind(self.language.prefix, self.language.namespace)

        # types that are connected to all sub cq. supertypes
        self.subtyped: set[TypeOperation] = set()
        self.supertyped: set[TypeOperation] = set()

        # In general, we can assume that the canonical types are defined in the 
        # vocabulary of the transformation language, so we do not need to add 
        # their structure to the transformation graph unless explicitly told to 
        # do so with with_canonical_types
        if not self.with_canonical_types:
            self.language.expand_canon()
            for t in self.language.canon:
                self.type_nodes[t] = self.language.uri(t)

    def add_vocabulary(self) -> None:
        """
        Add the RDF vocabulary for describing expressions in terms of the types
        and operations defined for this transformation algebra.
        """
        ns = self.language.namespace
        self.add_taxonomy()
        self.add_operators()
        self.add((ns["signature"], RDFS.subPropertyOf, TF["signature"]))
        self.add((ns["expression"], RDFS.subPropertyOf, TF["expression"]))
        self.add((ns["type"], RDFS.subPropertyOf, TF["type"]))
        self.add((ns["via"], RDFS.subPropertyOf, TF["via"]))

    def add_supertypes(self, t: TypeOperation,
            recursive: bool = False) -> None:
        if t not in self.supertyped:
            ref = self.language.uri(t)
            for s in self.language.supertypes(t):
                self.add((ref, RDFS.subClassOf, self.language.uri(s)))
                if recursive:
                    self.add_supertypes(s, recursive=True)
            if recursive:
                self.supertyped.add(t)

    def add_subtypes(self, t: TypeOperation, recursive: bool = False) -> None:
        if t not in self.subtyped:
            ref = self.language.uri(t)
            for s in self.language.subtypes(t):
                self.add((self.language.uri(s), RDFS.subClassOf, ref))
                if recursive:
                    self.add_subtypes(s, recursive=True)
            if recursive:
                self.subtyped.add(t)

    def add_taxonomy(self) -> None:
        """
        Add a taxonomy of types.
        """

        assert self.with_canonical_types
        self.language.expand_canon()
        for t in self.language.canon:
            self.add_type(t)
            self.add_subtypes(t)
            self.add_supertypes(t)

        if self.with_transitive_closure:
            for t in self.language.canon:
                ref = self.language.uri(t)
                for sub in self.transitive_subjects(RDFS.subClassOf, ref):
                    assert sub is not None
                    self.add((sub, RDFS.subClassOf, ref))

    def add_operators(self) -> None:
        for op in self.language.operators.values():
            node = self.language.uri(op)

            if self.with_classes:
                self.add((node, RDF.type, TF.Operation))

            if self.with_labels:
                self.add((node, RDFS.label, Literal(str(op.name))))
                self.add((node, self.language.namespace["signature"],
                    Literal(str(op.type))))
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
            node: Node
            try:
                node = self.language.uri(t)  # type: ignore
            except NonCanonicalTypeError:
                if self.with_noncanonical_types:
                    node = BNode()
                else:
                    raise

            if self.with_classes:
                self.add((node, RDF.type, TF.Type))

            if self.with_labels:
                self.add((node, RDFS.label, Literal(t.text())))

            if isinstance(t, TypeOperation) and t.operator.arity > 0 \
                    and self.with_type_parameters:
                self.add((node, RDFS.subClassOf,
                    self.language.uri(t.operator)))
                for i, param in enumerate(t.params, start=1):
                    pred = RDF[f"_{i}"]  # type: ignore
                    self.add((node, pred, self.add_type(param)))

            if self.with_supertype_classes and t in self.language.canon:
                assert isinstance(t, TypeOperation)
                self.add_supertypes(t, recursive=True)

        self.type_nodes[t] = node

        return node

    def add_expr(self, expr: Expr, root: Node,
            current: Node | None = None, intermediate: bool = False,
            origin: Node | None = None) -> Node:
        """
        Translate and add the given expression to a representation in RDF and
        add it to the given graph. Inputs that match the labels in the
        expression are appropriately attached, either as data sources or as
        previously added input expressions.

        This is a lossy conversion, because the order of arguments and the
        exact structure of functions-as-arguments is not preserved.

        Note also that, when recording supertypes in the `:subtypeOf` or 
        `:containsType` predicates using the `with_supertypes` and 
        `with_membership_supertypes` options respectively, these supertypes 
        only include *canonical* supertypes, even if `with_noncanonical_types` 
        is on.
        """
        assert isinstance(expr.type, TypeInstance)

        if expr in self.expr_nodes:
            return self.expr_nodes[expr]

        current = current or BNode()

        if isinstance(expr, Source):
            intermediate = False

            # Sources are always saved since they must always be reused when
            # used multiple times, especially if passthrough is disabled
            self.expr_nodes[expr] = current

            canonical = expr.type in self.language.canon

            if self.with_types and (canonical or self.with_noncanonical_types):

                type_node = self.add_type(expr.type)
                self.add((current, TF.type, type_node))

                if self.with_supertypes and canonical:
                    self.add((current, TF.subtypeOf, type_node))

                if self.with_membership:
                    self.add((root, TF.containsType, type_node))

                if (isinstance(expr.type, TypeOperation) and canonical):
                    for stype in self.language.supertypes(expr.type, 
                            transitive=True):
                        stype_node = self.add_type(stype)
                        if self.with_membership_supertypes:
                            self.add((root, TF.containsType, stype_node))
                        if self.with_supertypes:
                            self.add((current, TF.subtypeOf, stype_node))

            if self.with_labels:
                self.add((current, RDFS.label,
                    Literal(f"{expr.type} from source")))

        elif isinstance(expr, Operation):
            # assert not expr.operator.definition, \
            #     f"{expr.operator} should be a primitive"

            output_type = expr.type.output().normalize()

            canonical = self.with_noncanonical_types or \
                output_type in self.language.canon

            essential = self.with_intermediate_types or \
                not intermediate

            if self.with_operators:
                op_node = self.language.namespace[expr.operator.name]
                self.add((current, TF.via, op_node))

                if self.with_membership:
                    self.add((root, TF.containsOperator, op_node))

            if self.with_types and canonical and essential:

                type_node = self.add_type(output_type)
                self.add((current, TF.type, type_node))

                canonical = output_type in self.language.canon

                if self.with_membership:
                    self.add((root, TF.containsType, type_node))

                if self.with_supertypes and canonical:
                    self.add((current, TF.subtypeOf, type_node))

                if (isinstance(output_type, TypeOperation) and canonical):
                    for stype in self.language.supertypes(output_type, 
                            transitive=True):
                        stype_node = self.add_type(stype)
                        if self.with_membership_supertypes:
                            self.add((root, TF.containsType, stype_node))
                        if self.with_supertypes:
                            self.add((current, TF.subtypeOf, stype_node))

            if self.with_labels:
                self.add((current, RDFS.label, Literal(
                    f"{output_type} via {expr.operator.name}")))

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
            f = self.add_expr(expr.f, root, current, intermediate, origin)

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
            if isinstance(expr.x.type, TypeOperation) and \
                    expr.x.type.operator == Function:
                internal: Node = BNode()
                current_internal = internal
                self.add((f, TF.internal, internal))

                if self.with_labels:
                    self.add((internal, RDFS.label, Literal(
                        "(internal)")))

                if isinstance(expr.x, Abstraction):
                    for p in expr.x.params:
                        self.expr_nodes[p] = internal

                    x = self.add_expr(expr.x.body, root, BNode(),
                        intermediate=True)
                else:
                    x = self.add_expr(expr.x, root, BNode(), intermediate=True,
                        origin=origin)
                    self.add_from(x, internal)
            else:
                x = self.add_expr(expr.x, root, BNode(), intermediate=True,
                    origin=origin)
            self.add_from(f, x)

            # If `x` has internal operations of its own, then those inner
            # operations should be fed by the current (outer) internal
            # operation, which has access to additional parameters that may be
            # used by the inner one. See issues #37 and #41.
            if current_internal:
                for internal in self.objects(x, TF.internal):
                    self.add_from(internal, current_internal)

            # Every operation that is internal to `f` should also take `x`'s
            # output as input
            for internal in self.objects(f, TF.internal):
                if internal != current_internal:
                    self.add_from(internal, x)

            # ... and every input to `f` should be an input to this internal
            # operation
            if current_internal:
                for f_input in self.objects(f, TF["from"]):
                    if x != f_input:
                        self.add_from(current_internal, f_input)

                if origin and self.with_workflow_origin:
                    self.add((current_internal, TF["origin"], origin))

        if origin and self.with_workflow_origin:
            self.add((current, TF["origin"], origin))
        return current

    def add_from(self, a: Node, b: Node, recursive: bool = False) -> None:
        """Add predicate to say that `b` feeds into `a`. If the option is on, 
        all transitive dependencies will also be added."""
        self.add((a, TF["from"], b))
        if self.with_dependencies:

            self.add((a, TF.depends, b))
            if recursive:
                for bdep in self.transitive_objects(b, TF["from"]):
                    assert bdep
                    self.add((a, TF.depends, bdep))
            else:
                for bdep in self.objects(b, TF.depends):
                    self.add((a, TF.depends, bdep))

    def add_workflow(self, wf: Workflow) -> dict[Node, Node]:
        """
        Convert a `Workflow` to a full transformation graph by converting its
        individual workflow steps (source data and tool applications) to
        representations of expressions in RDF.
        """
        # 1. Construct expressions for each workflow resource (source data or
        # tool output), possibly using expressions from previous steps
        exprs: dict[Node, Expr] = dict()

        # If passthrough is disabled, we record connections between sources and
        # the expressions they are derived from (which may be more specific)
        indirection: dict[Source, Expr] = dict()

        for source, type in wf.source_types(self.language):
            exprs[source] = Source(type)

        def wfnode2expr(wfnode: Node) -> Expr:
            try:
                return exprs[wfnode]
            except KeyError:
                assert wfnode in wf.tool_outputs
                input_nodes = list(wf.inputs(wfnode))
                input_exprs = [wfnode2expr(n) for n in input_nodes]
                if not self.passthrough:
                    for i in range(len(input_exprs)):
                        if input_nodes[i] not in wf.sources:
                            e = input_exprs[i]
                            s = input_exprs[i] = Source()
                            e.fix()
                            indirection[s] = e

                try:
                    exprs[wfnode] = expr = self.language.parse_expr(
                        wf.expression(wfnode), *input_exprs, fix=True)
                except (TypingError, ParseError) as e:
                    raise WorkflowCompositionError(wf, wfnode) from e
                return expr

        wfnode2expr(wf.target()).fix()

        # 2. Convert individual transformation expressions to nodes and add
        # them. We must do this in the proper order, so that expression nodes
        # for the same tool/source nodes get saved in self.expr_nodes and
        # reused rather than regenerated
        def wfnode2tfmnode(wfnode: Node) -> Node:
            expr = wfnode2expr(wfnode)
            try:
                tfmnode = self.expr_nodes[expr]
            except KeyError:
                if wfnode not in wf.sources:
                    for input_wfnode in wf.inputs(wfnode):
                        # Running this has the side effect that the wfnode for
                        # every input will already have been added
                        wfnode2tfmnode(input_wfnode)

                self.expr_nodes[expr] = tfmnode = self.add_expr(expr, wf.root,
                    origin=wfnode)

            return tfmnode

        result_node = wfnode2tfmnode(wf.target())

        # If passthrough is disabled, connections between the outputs of one
        # tool and the inputs of the next are not established; do that now
        for source_expr, ref_expr in indirection.items():
            src_tfmnode = self.add_expr(source_expr, wf.root)
            ref_tfmnode = self.expr_nodes[ref_expr]
            self.add_from(src_tfmnode, ref_tfmnode, recursive=True)

        if self.with_inputs:
            for wfnode in wf.sources:
                tfmnode = wfnode2tfmnode(wfnode)
                self.add((wf.root, TF.input, tfmnode))

        if self.with_output:
            self.add((wf.root, TF.output, result_node))

        assert not self.uri
        self.uri = wf.root

        if self.with_classes:
            self.add((wf.root, RDF.type, TF.Transformation))

        return {wfnode: self.expr_nodes[expr]
            for wfnode, expr in exprs.items()}

    def parse_shortcuts(self, remove: bool = True) -> None:
        """
        For convenience, types and operators may be specified as string
        literals in RDF using the `type` and `via` predicates in the language's
        own namespace. This method automatically parses these strings and
        replaces them with `tf:type` and `tf:via` predicates with the
        corresponding nodes as object. Example:

            [] lang:type "F(A, _)".

        Becomes:

            [] tf:type lang:F-A-Top.
        """
        ns = self.language.namespace

        for subj, obj in self[:ns.type:]:
            t = self.language.parse_type(str(obj))
            self.add((subj, TF.type, self.language.uri(t)))

        for subj, obj in self[:ns.via:]:
            operator = self.language.parse_operator(str(obj))
            self.add((subj, TF.via, self.language.uri(operator)))

        if remove:
            self.remove((None, ns.type, None))
            self.remove((None, ns.via, None))

    def visualize(self, path: Path | str | None) -> str | None:
        """
        Produce a GraphViz visualization of the concept graph, with subgraphs
        for every tool application.
        """
        # `rdflib`'s `rdf2dot` is useful, but produces output that is too messy
        # for complex graphs

        # TODO maybe use a dedicated graph library for this
        # TODO separate from WF.output
        concepts_in: set[Node] = set(self.objects(self.uri, TF.input))
        concepts_app: set[Node] = set(self.subjects(TF["from"])).union(
            self.objects(self.uri, TF.output))
        concepts_in_internal: set[Node] = set(x
            for x in self.objects(None, TF["from"])
            if x not in concepts_in)

        # Map tool applications to constituent concepts.
        app2concepts: dict[Node | None, set[Node]] = defaultdict(set)
        for c in chain(concepts_app, concepts_in_internal):
            app = self.value(c, TF.origin, any=False)
            app2concepts[app].add(c)

        h = open(path, 'w', encoding="utf-8") if path else StringIO()
        try:
            h.write("digraph G {\n")
            h.write(f"label=<<font face=\"monospace\">{self.uri}</font>>;\n")
            h.write("\tcompound=true;\n")
            h.write("\tnode [shape=rectangle];\n")

            # Input concepts
            for c in concepts_in:
                type = self.value(c, TF.type, any=False)
                origin = self.value(c, TF.origin, any=False)
                assert type
                # typelabel = escape(self.value(type, RDFS.label, any=False))
                typelabel = escape(shorten(type))
                datalabel = escape(self.value(origin, RDFS.label, any=False))
                h.write(f"\tsubgraph cluster{c} {{\n")
                h.write(f"\t\tlabel=<<u><i>{datalabel or origin}</i></u>>;\n")
                h.write(f"\t\t{c} [shape=none, label=<{typelabel}> ];\n")
                h.write("\t}\n")

            # Transformed concepts and internal input concepts
            for wf_out, tfm_concepts in app2concepts.items():
                datalabel = escape(self.value(wf_out, RDFS.label, any=False))
                wf_app = self.value(None, WF.output, wf_out, any=False)
                tool = self.value(wf_app, WF.applicationOf, any=False)
                h.write(f"\tsubgraph cluster{wf_app} {{\n")
                if tool:
                    h.write(f"\t\tlabel=<<b>{shorten(tool)}</b><br/>"
                        f"<i>{datalabel}</i>>;\n")
                else:
                    h.write("\t\tlabel=<<b>anonymous tool</b>>;\n")
                for c in tfm_concepts:
                    label = self.value(c, RDFS.label, any=False)
                    type = self.value(c, TF.type, any=False)
                    if "internal" in label:
                        h.write(f"\t\t{c} [shape=circle, "
                            f"style=dashed, label=\"\"];\n")
                        # Show which node is internal to which
                        # for x in self.subjects(TF.internal, c):
                        #     h.write(f"\t\t{x} -> {c} [style=dashed];\n")
                    else:
                        via = self.value(c, TF.via, any=False)
                        op = shorten(escape(via))
                        if type:
                            typelabel = escape(shorten(type))
                            # self.value(type, RDFS.label))
                            if via:
                                h.write(f"\t\t{c} [label=<{typelabel}<br/>"
                                    f"via {op}>];\n")
                            else:
                                h.write(f"\t\t{c} [label=<{typelabel}>];\n")
                        else:
                            # If no type is found, then this must have been a
                            # non-canonical type. We use the label on the
                            # resource as a fallback and make it red
                            errorlabel = escape(label).replace(' via ',
                                '<br/>via ')
                            h.write(f"\t\t{c} [label=<"
                                f"<font color=\"red\">non-canonical<br/>"
                                f"{errorlabel}</font>>];\n")
                h.write("\t}\n")

            # Connect all the nodes
            for node2, node1 in self.subject_objects(TF["from"]):
                if node1 in concepts_in:
                    h.write(f"\t{node1} -> {node2} [ltail=cluster{node1}];\n")
                else:
                    h.write(f"\t{node1} -> {node2};\n")

            h.write("}\n\n")

            if isinstance(h, StringIO):
                return h.getvalue()
            else:
                return None

        finally:
            h.close()


class WorkflowCompositionError(Exception):
    """
    Raised when an errors occurs while trying to compose a transformation graph
    for a workflow.
    """

    def __init__(self, wf: Workflow, node: Node):
        self.wf = wf
        self.node = node

    def __str__(self) -> str:
        assert self.__cause__, "must be caused by another error"
        tool = self.wf.tool(self.node)

        # Add information on where it came from
        origin = ""
        if isinstance(self.__cause__, TypeAnnotationError):
            input_number = self.__cause__.input
            if input_number:
                input_node = list(self.wf.inputs(self.node))[input_number - 1]
                if input_node in self.wf.sources:
                    origin = f"source <{shorten(input_node)}>"
                else:
                    app = self.wf.tool(input_node)
                    origin = f"an application of <{shorten(app)}>"
            if origin:
                origin = f"\n\tthat gets input #{input_number} from {origin}, "

        return (
            f"In workflow <{shorten(self.wf.root)}>\n"
            f"\tin an application of <{shorten(tool)}>{origin}\n"
            f"\t{self.__cause__}"
        )


class CyclicTransformationGraphError(Exception):
    """
    Raised when a transformation is cyclical.
    """
