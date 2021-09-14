"""
This module extends transformation algebras such that its expressions can be
parsed as RDF graphs.
"""

from __future__ import annotations

from transformation_algebra import error, flow
from transformation_algebra.type import Type, TypeOperation, TypeVar, \
    Function, TypeOperator, TypeInstance
from transformation_algebra.expr import \
    TransformationAlgebra, Expr, Base, Application, Abstraction, Data, \
    Operation, Variable, Definition

from itertools import count, chain
from rdflib import URIRef, Graph, Namespace, BNode, Literal
from rdflib.term import Node
from rdflib.namespace import RDF, RDFS
from rdflib.plugins import sparql

from typing import Dict, Union, Iterator, Optional, Tuple

TA = Namespace("https://github.com/quangis/transformation-algebra#")


class TransformationAlgebraRDF(TransformationAlgebra):
    def __init__(self, prefix: str, namespace: Union[Namespace, str]):
        self.prefix = prefix
        self.namespace = Namespace(namespace) \
            if isinstance(namespace, str) else namespace
        super().__init__()

    def uri(self, value: Union[TypeOperator, Definition]) -> URIRef:
        """
        Obtain the URI node for an operation or type operator.
        """
        if value == Function:
            return TA.Function

        assert value in self, f"{value} is not in algebra"
        if isinstance(value, TypeOperator):
            return self.namespace.term(value.name)
        else:
            assert isinstance(value, Definition) and value.name
            return self.namespace.term(value.name)

    def vocabulary(self) -> Graph:
        """
        Produce an RDF vocabulary for describing expressions in terms of the
        types and operations defined for this transformation algebra.
        """
        vocab = Graph()
        self.bindings(vocab)

        # Add type operators to the vocabulary
        for t in self.types:
            if t.arity > 0:
                current_uri = self.uri(t)
                vocab.add((current_uri, RDF.type, TA.Type))
                vocab.add((current_uri, RDFS.subClassOf, RDF.Seq))
                vocab.add((current_uri, RDFS.label, Literal(str(t))))
            else:
                previous_uri = None
                current: Optional[TypeOperator] = t
                while current:
                    current_uri = self.uri(current)
                    vocab.add((current_uri, RDFS.label, Literal(str(t))))
                    vocab.add((current_uri, RDF.type, TA.Type))
                    if previous_uri:
                        vocab.add((previous_uri, RDFS.subClassOf, current_uri))
                    previous_uri = current_uri
                    current = current.supertype

        # Add operations to the vocabulary
        for d in self.definitions.values():
            node = self.uri(d)
            type_node = TA.Data if isinstance(d, Data) else TA.Operation
            vocab.add((node, RDF.type, type_node))
            vocab.add((node, RDFS.label, Literal(str(d.name))))
            if d.description:
                vocab.add((node, RDFS.comment, Literal(d.description)))

        return vocab

    def parse_rdf(self, graph: Graph, string: str) -> BNode:
        """
        Convenience function to parse an expression and add it to an RDF graph
        in one go.
        """
        root = BNode()
        self.rdf_expr(graph, root, self.parse(string).primitive())
        return root

    def bindings(self, g: Graph) -> None:
        """
        Add namespace bindings to RDF.
        """
        g.bind("ta", TA)
        g.bind(self.prefix, self.namespace)

    def rdf_type(self, output: Graph, type: Type) -> Node:
        """
        Translate the given type to a representation in RDF and add it to the
        given graph. Return the top-level node.
        """
        t = type.instance()
        if isinstance(t, TypeOperation):

            if t.params:
                node = BNode()
                output.add((node, RDFS.label, Literal(str(t))))
                output.add((node, RDF.type, self.uri(t._operator)))
                for i, param in enumerate(t.params, start=1):
                    param_node = self.rdf_type(output, param)
                    output.add((node, RDF.term(f"_{i}"), param_node))
            else:
                return self.uri(t._operator)
        else:
            # TODO don't make new node if we already encountered this variable
            assert isinstance(t, TypeVar)
            node = BNode()
            output.add((node, RDF.type, TA.TypeVar))
            output.add((node, RDFS.label, Literal(str(t))))
        return node

    def rdf_expr(self,
            output: Graph,
            root: Node,
            expr: Expr,
            inputs: Dict[str, Union[Node, Tuple[Node, Expr]]] = {},
            current: Optional[Node] = None,
            variables: Dict[Variable, Node] = {},
            include_types: bool = True,
            include_labels: bool = True) -> Node:
        """
        Translate the given expression to  a representation in RDF and add it
        to the given graph, connecting all intermediary data and operation
        nodes to the given root. Inputs that match the labels in the expression
        are appropriately attached, either as data sources or as input
        expressions.
        """
        assert isinstance(expr.type, TypeInstance)

        # Ensure some basic properties of the graph
        output.bind("ta", TA)
        output.bind(self.prefix, self.namespace)
        output.add((root, RDF.type, TA.Transformation))

        current = current or BNode()

        # Add connections to input or output nodes
        if isinstance(expr, Base):
            if include_labels:
                label = Literal(f"instance of {expr.definition.name}")
                output.add((current, RDFS.label, label))

            if isinstance(expr.definition, Operation):
                assert expr.definition.is_primitive(), \
                    f"{expr.definition} is not a primitive"
                output.add((root, TA.operation, current))
                output.add((current, RDF.type, self.uri(expr.definition)))
            else:
                assert isinstance(expr.definition, Data)
                if expr.label:
                    # Source nodes are attached to inputs. Blank nodes are from
                    # other expressions, URIs are from data sources.
                    try:
                        source = inputs[expr.label]
                    except KeyError as e:
                        msg = f"no input node named '{expr.label}'"
                        raise RuntimeError(msg) from e
                    else:
                        if isinstance(source, Node):
                            output.add((current, TA.source, source))
                        else:
                            source_node, source_expr = source
                            assert isinstance(source_node, Node) and \
                                isinstance(source_expr, Expr)
                            try:
                                # TODO unification happens as we translate to
                                # RDF, which means some might be outdated
                                source_expr.type.unify(expr.type, subtype=True)
                            except error.TATypeError as e:
                                e.while_unifying(source_expr, expr)
                                raise
                            return source_node
                output.add((root, TA.data, current))
                output.add((current, RDF.type, TA.Data))

        elif isinstance(expr, Application):

            assert not isinstance(expr.f, Abstraction), \
                "abstractions can only occur as parameters, otherwise " \
                "they could have been β-reduced"

            # A function application with multiple arguments will look
            # something like `(f x) y`. By recursively adding the function part
            # to the current node, and attaching a new node for the argument
            # part, we eventually get a node for the function to which nodes
            # for all parameters are attached.
            f = self.rdf_expr(output, root, expr.f, inputs, current, variables,
                include_types, include_labels)

            # Attaching that parameter is straightforward for simple data:
            if expr.x.type.operator != Function:
                x = self.rdf_expr(output, root, expr.x, inputs, BNode(),
                    variables, include_types, include_labels)
                output.add((f, TA.input, x))
                x_data = x
                current_internal_operation = None

            # But when the parameter is a *function*, we need to be careful.
            # Conceptually, `x` will be some operation that produces data while
            # inside the black box of `f`. We don't know exactly what happens
            # internally --- only that there is some process that passes data
            # to `x` and that may synthesize its own input from any other data
            # known to `f`.
            # Now, if `x` is an abstraction --- an anonymous operation with an
            # inner structure --- then its body represents the data (TODO or
            # the function) that `f` may use. For primitive functions, we add a
            # data node between `x` and `f` to represent this data.
            # The mysterious input to `x` (or, in the case of abstractions, the
            # values of `x`'s parameters) is then a single unspecified internal
            # data node produced by an internal operation. All inputs to `f`
            # should be made inputs to this operation also.
            else:
                internal_data, internal_op = BNode(), BNode()
                output.add((internal_data, RDF.type, TA.InternalData))
                output.add((internal_op, RDF.type, TA.InternalOperation))
                output.add((internal_op, TA.output, internal_data))
                # output.add((root, TA.data, internal_data))
                # output.add((root, TA.operation, internal_op))

                if include_labels:
                    output.add((internal_op, RDFS.label,
                        Literal("internal operation")))
                    output.add((internal_data, RDFS.label,
                        Literal("internal data")))

                if isinstance(expr.x, Abstraction):
                    assert expr.x.body.type.operator != Function  # TODO why

                    variables = dict(variables)
                    for param in expr.x.params:
                        variables[param] = internal_data

                    x_data = self.rdf_expr(output, root, expr.x.body, inputs,
                        BNode(), variables, include_types, include_labels)
                else:
                    x_data = BNode()
                    output.add((x_data, RDF.type, TA.Data))  # TODO add type

                    x = self.rdf_expr(output, root, expr.x, inputs, BNode(),
                        variables, include_types, include_labels)
                    output.add((x, TA.input, internal_data))
                    output.add((x, TA.output, x_data))

                output.add((f, TA.input, x_data))
                output.add((f, TA.internal, internal_op))
                current_internal_operation = internal_op

            # Every operation that is internal to `f` should also take `x` (or
            # the output of `x`) as input
            for internal_op in output.objects(f, TA.internal):
                if internal_op != current_internal_operation:
                    output.add((internal_op, TA.input, x_data))

            # If the *output* of this application is data (that is, not another
            # function), move on to a new node representing that output
            if expr.type.operator != Function:
                current = BNode()
                output.add((current, RDF.type, TA.Data))
                if include_labels:
                    label = Literal("output of operation")
                    output.add((current, RDFS.label, label))
                output.add((root, TA.data, current))
                output.add((f, TA.output, current))

        else:
            assert isinstance(expr, Variable) and expr in variables
            return variables[expr]

        # Add information on the type of node, but only for data nodes
        if include_types and expr.type.operator != Function:
            output.add((current, TA.type, self.rdf_type(output, expr.type)))

        return current

    def rdf_workflow(self,
            output: Graph,
            root: Node,
            sources: set[Node],
            steps: dict[Node, tuple[Expr, list[Node]]]) -> Node:
        """
        Convert a workflow to an RDF graph by converting its individual steps
        to representations of expressions in RDF and combining them. Return the
        final 'output' node of the expression.

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
                cache[step_node] = self.rdf_expr(output, root, expr, inputs={
                    f"x{i}": (to_expr_node(x), steps[x][0]) if x in steps else x
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
        output.add((root, TA.target, final_expr_node))
        return final_expr_node

    def sparql_type(self, name: str, type: Type,
            name_generator: Iterator[str],
            index: Optional[int] = None) -> Iterator[str]:
        """
        Produce SPARQL constraints for the given (non-function) type.
        """

        t = type.instance()

        if isinstance(t, TypeVar):
            # If a type in a trace query contains variables, it must be a
            # wildcard --- because we don't do anything with it
            assert t.wildcard
        else:
            assert isinstance(t, TypeOperation) and t.operator != Function

            pred = "ta:type" if index is None else f"rdf:_{index}"
            if t.params:
                bnode = next(name_generator)
                yield f"?{name} {pred} ?{bnode}."
                yield f"?{bnode} rdf:type <{self.uri(t._operator)}>."
                for i, param in enumerate(t.params, start=1):
                    yield from self.sparql_type(bnode, param, name_generator,
                        index=i)
            else:
                yield f"?{name} {pred}/(rdfs:subClassOf*) <{self.uri(t._operator)}>."

    def trace(self,
            name: str,
            current: flow.Flow,
            previous: Optional[Tuple[str, flow.Unit, bool]] = None,
            name_generator: Optional[Iterator[str]] = None) -> \
            Iterator[str]:
        """
        Trace the paths between each node in a chain to produce SPARQL
        constraints.
        """
        name_generator = name_generator or iter(f"n{i}" for i in count())

        if isinstance(current, flow.Unit):
            if previous:
                yield (
                    f"?{previous[0]} "
                    f"({self.path(previous[1], previous[2], current)}) "
                    f"?{name}.")

            if isinstance(current, Operation):
                assert current.is_primitive(), \
                    "operation in a flow query must be primitive"
                yield f"?{name} rdf:type <{self.uri(current)}>."
            elif isinstance(current, Type):
                yield from self.sparql_type(name, current, name_generator)
            else:
                raise NotImplementedError

        elif isinstance(current, flow.Parallel):
            for sub in current.branches:
                yield from self.trace(next(name_generator), sub, previous,
                    name_generator)

        else:
            assert isinstance(current, flow.Serial)

            # TODO remove this assumption when possible
            assert all(not isinstance(x, flow.Parallel)
                for x in current.sequence[:-1])

            for n, x in zip(chain([name], name_generator), current.sequence):
                if x is None:
                    if previous is not None:
                        previous = previous[0], previous[1], True
                else:
                    yield from self.trace(n, x, previous, name_generator)
                    if isinstance(x, flow.Unit):
                        previous = n, x, False
                    else:
                        break

    def sparql_flow(self, flow: flow.Flow) -> sparql.Query:
        """
        Convert this Flow to a SPARQL query.
        """

        query = [
            "SELECT ?workflow ?description WHERE {",
            "?workflow rdf:type ta:Transformation.",
            "?workflow rdfs:comment ?description.",
            "?workflow ta:target ?output_node.",
            # "?workflow ta:data ?output_node.",
            # "FILTER NOT EXISTS {?next_step ta:input ?output_node}."
        ]
        query.extend(self.trace("output_node", flow))
        query.append("} GROUP BY ?workflow")

        print("Query is:")
        print("\n".join(query))
        print()

        return sparql.prepareQuery("\n".join(query),
                initNs={'ta': TA, 'rdf': RDF, 'rdfs': RDFS,
                    self.prefix: self.namespace}
        )

    def path(self, a: flow.Unit, skip: bool, b: flow.Unit) -> str:
        """
        Produce a SPARQL property path describing the connection between two
        nodes that represent either a data type or an operation. `skip`
        indicates that multiple intermediaries may lie between.
        """
        # See also rdflib.paths

        repeat = "+" if skip else ""
        if isinstance(a, Type):
            if isinstance(b, Type):
                return f"(^ta:output/ta:input+){repeat}"
            else:
                assert isinstance(b, Operation)
                return "^ta:output/((ta:input+/^ta:output)*)" if repeat else "^ta:output"
        else:
            assert isinstance(a, Operation)
            if isinstance(b, Type):
                return "ta:input/((^ta:output/ta:input+)*)" if repeat else "ta:input+"
            else:
                assert isinstance(b, Operation)
                return f"(ta:input+/^ta:output){repeat}"

    def query(self, g: Graph, flow: flow.Flow) -> sparql.QueryResult:
        return g.query(self.sparql_flow(flow))
