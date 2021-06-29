"""
This module extends transformation algebras such that its expressions can be
parsed as RDF graphs.
"""

from __future__ import annotations

from transformation_algebra import error
from transformation_algebra.type import Type, TypeOperation, TypeVar, \
    Function, TypeOperator, TypeInstance
from transformation_algebra.expr import \
    TransformationAlgebra, Expr, Base, Application, Abstraction, Data, \
    Operation, Variable, Definition

import io
from itertools import count
from rdflib import URIRef, Graph, Namespace, BNode, Literal
from rdflib.term import Node
from rdflib.namespace import RDF, RDFS
from rdflib.tools.rdf2dot import rdf2dot
from rdflib.plugins import sparql

from typing import Dict, Union, List, Iterator, Optional, Tuple

TA = Namespace(
    "https://github.com/quangis/transformation-algebra/"
    "TransformationAlgebra.rdf#"
)


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
                    vocab.add((previous_uri, RDF.type, current_uri))
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
            return self.rdf_expr(graph, root, expr.primitive())

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
        node = BNode()
        if isinstance(t, TypeOperation):
            graph.add((node, RDF.type, self.uri(t._operator)))

            if t.params:
                graph.add((node, RDF.type, RDF.Seq))
            for i, param in enumerate(t.params, start=1):
                param_node = self.rdf_type(graph, param)
                graph.add((node, RDF.term(f"_{i}"), param_node))
        else:
            # TODO don't make new node if we already encountered this variable
            assert isinstance(t, TypeVar)
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

        g.add((root, RDF.type, TA.Transformation))

        # If no intermediate node was provided, make a fresh one
        intermediate = intermediate or BNode()

        # Determine and attach the kind of this node
        if isinstance(expr, Variable):
            node_kind = TA.Variable
        elif expr.type.operator == Function:
            node_kind = TA.Operation
        else:
            node_kind = TA.Data

        # Add connections to input or output nodes
        if isinstance(expr, Base):
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
                            source_expr.type.unify(expr.type, subtype=True)
                            return source_node
                g.add((root, TA.data, intermediate))

        elif isinstance(expr, Abstraction):
            assert isinstance(expr.body, Expr)
            f = self.rdf_expr(g, root, expr.body, inputs)
            g.add((intermediate, TA.input, f))
        elif isinstance(expr, Application):
            f = self.rdf_expr(g, root, expr.f, inputs, intermediate)
            x = self.rdf_expr(g, root, expr.x, inputs)
            g.add((f, TA.input, x))

            # If the output of this application is data (that is, no more
            # arguments to present), make a new intermediate/output node
            if expr.type.operator != Function:
                intermediate = BNode()
                g.add((f, TA.output, intermediate))
                g.add((root, TA.data, intermediate))

        # Add semantic information on the type of node
        if not isinstance(expr, Application) or expr.type.operator != Function:
            node_type = self.rdf_type(g, expr.type)
            g.add((intermediate, RDF.type, node_kind))
            g.add((intermediate, TA.type, node_type))

        return intermediate

    def sparql_type(self, type: Type, name: str,
            name_generator: Iterator[str]) -> Iterator[str]:
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
            yield (
                f"?{name} "
                f"rdf:type "
                f"<{self.uri(t._operator)}>."
            )
            for i, param in enumerate(t.params, start=1):
                next_name = next(name_generator)
                yield (
                    f"?{name} rdf:_{i} ?{next_name}."
                )
                yield from self.sparql_type(param, next_name, name_generator)


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


class Chain(object):
    """
    A chain captures some relevant aspects of the conceptual process described
    by an algebra expression, in terms of the sequence of types and operations
    that must occur in it.

    The same can also be described in terms of semantic linear time logic
    formulae (SLTL), but since we will be using SPARQL to search through
    workflows, the approach chosen here makes for a straightforward
    translation.
    """

    def __init__(self, *chain: Union[None, Type, Operation, List[Chain]]):
        """
        Starting from the output node, describe which data types and operations
        must occur, in what order. A `None` value indicates that we may skip
        zero or more steps, and a list indicates that every one of its
        subchains must at some point occur in one of the paths.
        """
        self.chain = list(chain)
        # Lists may only occur in the final place of a chain
        assert not any(isinstance(p, list) for p in self.chain[:-1])

    def to_sparql(self, algebra: TransformationAlgebraRDF) -> sparql.Query:
        """
        Convert this chain to a SPARQL query.
        """

        query = [
            "SELECT ?workflow WHERE {",
            "?workflow ta:transformation ?transformation.",
            "?tranformation rdf:type ta:Transformation.",
            "?transformation ta:step ?step.",
            "?step ta:output ?node0.",
            "FILTER NOT EXISTS {?next_step ta:input ?node0}."
        ]
        query.extend(self.trace(algebra=algebra))
        query.append("}")

        print()
        print("\n".join(query))
        print()

        return sparql.prepareQuery("\n".join(query),
            initNs={'ta': TA, 'rdf': RDF}
        )

    def trace(self,
            algebra: TransformationAlgebraRDF,
            name_generator: Optional[Iterator[str]] = None,
            previous: Optional[Tuple[str, Union[Type, Operation]]] = None,
            skip: bool = False) -> Iterator[str]:
        """
        Trace the paths between each node in this chain to produce a set of
        SPARQL constraints.
        """
        name_generator = name_generator or iter(f"node{i}" for i in count())

        for current in self.chain:
            if current is None:
                skip = True
            elif isinstance(current, list):
                for subchain in current:
                    yield from subchain.trace(
                        algebra, name_generator, previous, skip)
            else:
                assert isinstance(current, (Operation, Type))
                name = next(name_generator)
                if previous:
                    yield (
                        f"?{previous[0]} "
                        f"{self.path(previous[1], current, skip)} "
                        f"?{name}.")

                if isinstance(current, Operation):
                    yield f"?{name} rdf:type {algebra.uri(current)}."
                else:
                    assert isinstance(current, Type)
                    next_name = next(name_generator)
                    yield f"?{name} ta:type ?{next_name}."
                    yield from algebra.sparql_type(current, next_name,
                            name_generator)

                skip = False
                previous = name, current

    def path(self, before: Union[Type, Operation], after: Union[Type, Operation],
            skip: bool) -> str:
        """
        Produce a SPARQL property path describing the connection between two
        nodes that represent either a data type or an operation. `skip`
        indicates that multiple intermediaries may lie between.
        """
        # See also rdflib.paths

        repeat = "+" if skip else ""
        if isinstance(before, Type):
            if isinstance(after, Type):
                return f"(^ta:output/ta:input){repeat}"
            else:
                assert isinstance(after, Operation)
                return f"(^ta:output){repeat}"
        else:
            assert isinstance(before, Operation)
            if isinstance(after, Type):
                return f"(ta:input){repeat}"
            else:
                assert isinstance(after, Type)
                return f"(ta.input/^ta:output){repeat}"
