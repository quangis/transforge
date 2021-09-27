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
                output.add((node, RDFS.subClassOf, self.uri(t._operator)))
                for i, param in enumerate(t.params, start=1):
                    param_node = self.rdf_type(output, param)
                    output.add((node, RDF[f"_{i}"], param_node))
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
            expr: Expr,
            root: Node,
            current: Optional[Node] = None,
            sources: Dict[str, Union[Node, Tuple[Node, Expr]]] = {},
            variables: Dict[Variable, Node] = {},
            include_types: bool = True,
            include_labels: bool = True,
            include_steps: bool = True) -> Node:
        """
        Translate the given expression to a representation in RDF and add it
        to the given graph, connecting all steps to root. Inputs that match the
        labels in the expression are appropriately attached, either as data
        sources or as input expressions.

        This is a lossy conversion, because the order of arguments and the
        exact structure of functions-as-arguments is not preserved.
        """
        assert isinstance(expr.type, TypeInstance)

        if isinstance(expr, Variable):
            assert expr in variables
            return variables[expr]

        current = current or BNode()
        output.add((root, RDF.type, TA.Transformation))

        if include_steps:
            output.add((root, TA.step, current))

        if isinstance(expr, Base):
            datatype = expr.type.output()

            if include_types:
                output.add((current, RDF.type,
                    self.rdf_type(output, datatype)))

            if include_labels:
                output.add((current, RDFS.label,
                    Literal(f"{datatype} via {expr.definition.name}")))

            if isinstance(expr.definition, Operation):
                assert expr.definition.is_primitive(), \
                    f"{expr.definition} is not a primitive"

                output.add((current, RDF.type, TA.TransformedData))
                output.add((current, TA.transformer,
                    self.uri(expr.definition)))
            else:
                assert isinstance(expr.definition, Data)

                output.add((current, RDF.type, TA.SourceData))

                if expr.label:
                    try:
                        source = sources[expr.label]
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
            f = self.rdf_expr(output, expr.f, root, current,
                sources, variables, include_types, include_labels,
                include_steps)

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
                output.add((internal, RDF.type, TA.InternalData))
                output.add((f, TA.internal, internal))

                if include_labels:
                    output.add((internal, RDFS.label, Literal("internal")))

                if isinstance(expr.x, Abstraction):
                    variables = variables | \
                        {p: internal for p in expr.x.params}

                    x = self.rdf_expr(output, expr.x.body, root, BNode(),
                        sources, variables, include_types, include_labels,
                        include_steps)
                else:
                    x = self.rdf_expr(output, expr.x, root, BNode(),
                        sources, variables, include_types, include_labels,
                        include_steps)
                    output.add((internal, TA.feeds, x))
            else:
                x = self.rdf_expr(output, expr.x, root, BNode(),
                    sources, variables, include_types, include_labels,
                    include_steps)
            output.add((x, TA.feeds, f))

            # If `x` has internal operations of its own, then those inner
            # operations should be fed by the current (outer) internal
            # operation, which has access to additional parameters that may be
            # used by the inner one. See issues #37 and #41.
            if current_internal:
                for internal in output.objects(x, TA.internal):
                    output.add((current_internal, TA.feeds, internal))

            # Every operation that is internal to `f` should also take `x`'s
            # output as input
            for internal in output.objects(f, TA.internal):
                if internal != current_internal:
                    output.add((x, TA.feeds, internal))

            # ... and every input to `f` should be an input to this internal
            # operation
            if current_internal:
                for data_input in output.subjects(TA.feeds, f):
                    if x != data_input:
                        output.add((data_input, TA.feeds, current_internal))

        return current

    def rdf_workflow(self,
            output: Graph,
            root: Node,
            sources: set[Node],
            steps: dict[Node, tuple[Expr, list[Node]]],
            **kwargs) -> Node:
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

        output.bind("ta", TA)
        output.bind(self.prefix, self.namespace)

        cache: Dict[Node, Node] = {}

        def to_expr_node(step_node: Node) -> Node:
            try:
                return cache[step_node]
            except KeyError:
                expr, inputs = steps[step_node]
                assert all(x in steps or x in sources for x in inputs), \
                    "unknown input data source"
                cache[step_node] = self.rdf_expr(output, expr, root, sources={
                    f"x{i}": (to_expr_node(x), steps[x][0])
                    if x in steps else x
                    for i, x in enumerate(inputs, start=1)
                }, **kwargs)
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
        output.add((root, TA.result, final_expr_node))
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

            pred = "rdf:type" if index is None else f"rdf:_{index}"
            if t.params:
                bnode = next(name_generator)
                yield f"?{name} {pred} ?{bnode}."
                yield f"?{bnode} rdfs:subClassOf <{self.uri(t._operator)}>."
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
        # See rdflib.paths: (~TA.feeds * OneOrMore).n3(g.namespace_manager)

        name_generator = name_generator or iter(f"n{i}" for i in count())

        if isinstance(current, flow.Unit):
            if previous:
                yield (
                    f"?{previous[0]} "
                    f"({'^ta:feeds' + ('+' if previous[2] else '')}) "
                    f"?{name}.")

            if isinstance(current, Operation):
                assert current.is_primitive(), \
                    "operation in a flow query must be primitive"
                yield f"?{name} ta:transformer <{self.uri(current)}>."
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
            "?workflow ta:result ?output_node.",
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

    def query(self, g: Graph, flow: flow.Flow) -> sparql.QueryResult:
        return g.query(self.sparql_flow(flow))
