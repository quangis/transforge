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

import io
from itertools import count, chain
from rdflib import URIRef, Graph, Namespace, BNode, Literal
from rdflib.term import Node
from rdflib.namespace import RDF, RDFS
from rdflib.tools.rdf2dot import rdf2dot
from rdflib.plugins import sparql

from typing import Dict, Union, Iterator, Optional, Tuple, List

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
            previous_uri = None
            current: Optional[TypeOperator] = t
            while current:
                current_uri = self.uri(current)
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
            if d.description:
                vocab.add((node, RDFS.label, Literal(d.description)))

        return vocab

    def parse_rdf(self, graph: Graph, string: str) -> BNode:
        """
        Convenience function to parse an expression and add it to an RDF graph
        in one go.
        """
        expr = self.parse(string)
        root = BNode()
        if expr:
            output = self.rdf_expr(graph, root, expr.primitive())
            graph.add((root, TA.output, output))

    def bindings(self, g: Graph) -> None:
        """
        Add namespace bindings to RDF.
        """
        g.bind("ta", TA)
        g.bind(self.prefix, self.namespace)

    def rdf_type(self, graph: Graph, value: Type) -> Node:
        """
        Translate the given type to a representation in RDF and add it to the
        given graph. Return the top-level node.
        """
        t = value.instance()
        if isinstance(t, TypeOperation):

            if t.params:
                node = BNode()
                graph.add((node, RDF.type, self.uri(t._operator)))

                if t.params:
                    graph.add((node, RDF.type, RDF.Seq))
                for i, param in enumerate(t.params, start=1):
                    param_node = self.rdf_type(graph, param)
                    graph.add((node, RDF.term(f"_{i}"), param_node))
            else:
                return self.uri(t._operator)
        else:
            # TODO don't make new node if we already encountered this variable
            assert isinstance(t, TypeVar)
            node = BNode()
            graph.add((node, RDF.type, TA.TypeVariable))
            graph.add((node, RDFS.label, Literal(str(t))))
        return node

    def rdf_expr(self, g: Graph, root: Node, expr: Expr,
            inputs: Dict[str, Union[URIRef, Tuple[Node, Expr]]] = {},
            intermediate: Optional[Node] = None) -> Node:
        """
        Translate the given expression to  a representation in RDF and add it
        to the given graph, connecting all intermediary data and operation
        nodes to the given root. Inputs that match the labels in the expression
        are appropriately attached, either as data sources or as input
        expressions.
        """
        assert isinstance(expr.type, TypeInstance)

        # Ensure some basic properties of the graph
        g.bind("ta", TA)
        g.bind(self.prefix, self.namespace)
        g.add((root, RDF.type, TA.Transformation))

        # If no intermediate node was provided, make a fresh one
        intermediate = intermediate or BNode()

        # Add connections to input or output nodes
        if isinstance(expr, Base):
            # assert expr.definition.primitive
            if isinstance(expr.definition, Operation):
                g.add((root, TA.operation, intermediate))
                g.add((intermediate, RDF.type, self.uri(expr.definition)))
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
                        if isinstance(source, URIRef):
                            g.add((intermediate, TA.source, source))
                        else:
                            source_node, source_expr = source
                            assert isinstance(source_node, Node) and \
                                isinstance(source_expr, Expr)
                            try:
                                source_expr.type.unify(expr.type, subtype=True)
                            except error.TATypeError as e:
                                e.while_unifying(source_expr, expr)
                                raise
                            return source_node
                g.add((root, TA.data, intermediate))
                g.add((intermediate, RDF.type, TA.Data))

        elif isinstance(expr, Application):
            f = self.rdf_expr(g, root, expr.f, inputs, intermediate)
            x = self.rdf_expr(g, root, expr.x, inputs)
            g.add((f, TA.input, x))

            # If the output of this application is data (that is, no more
            # arguments to present), make a new intermediate/output node
            if expr.type.operator != Function:
                intermediate = BNode()
                g.add((root, TA.data, intermediate))
                g.add((f, TA.output, intermediate))
                g.add((intermediate, RDF.type, TA.Data))

        elif isinstance(expr, Abstraction):
            assert isinstance(expr.body, Expr) and expr.type and \
                expr.type.operator == Function
            f = self.rdf_expr(g, root, expr.body, inputs)
            g.add((intermediate, TA.input, f))
            g.add((intermediate, RDF.type, TA.Operation))

        else:
            assert isinstance(expr, Variable)
            g.add((intermediate, RDF.type, TA.Variable))

        # Add information on the type of node, but only for data nodes
        if expr.type.operator != Function:
            g.add((intermediate, TA.type, self.rdf_type(g, expr.type)))

        return intermediate

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
                op = next(name_generator)
                yield f"?{name} {pred} ?{op}."
                yield f"?{op} rdfs:subClassOf* <{self.uri(t._operator)}>."

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

    def sparql_chain(self, chain: flow.Chain) -> sparql.Query:
        """
        Convert this chain to a SPARQL query.
        """

        query = [
            "SELECT ?workflow ?description WHERE {",
            "?workflow rdf:type ta:Transformation.",
            "?workflow rdfs:comment ?description.",
            "?workflow ta:target ?output_node.",
            # "?workflow ta:data ?output_node.",
            # "FILTER NOT EXISTS {?next_step ta:input ?output_node}."
        ]
        query.extend(self.trace("output_node", chain))
        query.append("}")

        print()
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
                return f"(^ta:output/ta:input){repeat}"
            else:
                assert isinstance(b, Operation)
                return "^ta:output/((ta:input/^ta:output)*)" if repeat else "^ta:output"
        else:
            assert isinstance(a, Operation)
            if isinstance(b, Type):
                return "ta:input/((^ta:output/ta:input)*)" if repeat else "ta:input"
            else:
                assert isinstance(b, Operation)
                return f"(ta:input/^ta:output){repeat}"

    def query(self, g: Graph, flow: flow.Flow) -> sparql.QueryResult:
        return g.query(self.sparql_chain(flow))


def dot(g: Graph) -> str:
    """
    Return a string of GraphViz dot syntax. You can pass the output to such
    programs as `xdot` to visualize the RDF graph.
    """
    # rdf2dot uses deprecated cgi.escape function; this hack solves that
    import cgi
    import html
    stream = io.StringIO()
    cgi.escape = html.escape  # type: ignore
    rdf2dot(g, stream)
    return stream.getvalue()
