"""
This module intends to provide a way to express transformation trees (i.e.
chains of types and operations) in such a way that a query might use it to
check if a given transformation expression contains types and operations in the
specified order.
"""

from __future__ import annotations

import rdflib
from rdflib import Graph
from rdflib.paths import Path, ZeroOrMore, OneOrMore
from rdflib.term import Node
from rdflib.namespace import Namespace, NamespaceManager, RDFS, RDF
from abc import ABC, abstractmethod
from itertools import count
from collections import defaultdict
from transformation_algebra.type import Type, TypeOperation, \
    Function, TypeVariable
from transformation_algebra.expr import Operator
from transformation_algebra.graph import TA
from typing import TYPE_CHECKING, Protocol, Iterator, Any, Union, \
    overload, Optional, TypeVar

# We use '...' to indicate that steps may be skipped. This workaround allows us
# to refer to the ellipsis' type. See github.com/python/typing/issues/684
if TYPE_CHECKING:
    from builtins import ellipsis
else:
    ellipsis = type(Ellipsis)

# For convenience, we allow nested sequences in `Flow.serial()`. The following
# temporary solution to the recursive type thus introduced has been lifted from
# github.com/python/mypy/issues/731
_T_co = TypeVar("_T_co")


class Nested(Protocol[_T_co]):
    def __len__(self) -> int:
        ...

    @overload
    def __getitem__(self, __index: int) -> _T_co | Nested[_T_co]:
        ...

    @overload
    def __getitem__(self, __index: slice) -> Nested[_T_co]:
        ...

    def __contains__(self, __x: object) -> bool:
        ...

    def __iter__(self) -> Iterator[_T_co | Nested[_T_co]]:
        ...

    def __reversed__(self) -> Iterator[_T_co | Nested[_T_co]]:
        ...

    def count(self, __value: Any) -> int:
        ...

    def index(self, __value: Any, __start: int = ...,
            __stop: int = ...) -> int:
        ...


Element = Union[Type, Operator, ellipsis, 'TransformationFlow']
NestedFlow = Union[Element, Nested[Element]]


class TransformationQuery(object):  # TODO subclass rdflib.Query?
    def __init__(self, items: NestedFlow, namespace: Namespace):
        self.flow = TransformationFlow.shorthand(items)
        self.namespace = namespace
        self.prefix = "n"
        self.nsm = NamespaceManager(Graph())

        self.nsm.bind("ta", TA)
        self.nsm.bind(self.prefix, self.namespace)

        self.generator = iter(rdflib.Variable(f"n{i}") for i in count(start=1))
        self.variable: dict[Unit, rdflib.Variable] = \
            defaultdict(self.generator.__next__)

    def n3(self, *items: Path | Node) -> str:
        result = []
        for item in items:
            try:
                result.append(item.n3(self.nsm))
            except TypeError:
                result.append(item.n3())

        return " ".join(result) + "."
        # assert all(isinstance(item, (Path, Node)) for item in items)
        # return " ".join(item.n3(self.nsm) for item in items) + "."

    def sparql_type(self, variable: rdflib.Variable, type: Type,
            index: Optional[int] = None) -> Iterator[str]:
        """
        Produce SPARQL constraints for the given (non-function) type.
        """

        t = type.instance()

        if isinstance(t, TypeVariable):
            # If a type in a trace query contains variables, it must be a
            # wildcard --- because we don't do anything with it
            assert t.wildcard
        else:
            assert isinstance(t, TypeOperation) and t.operator != Function

            pred = RDF.type if index is None else RDF[f"_{index}"]
            if t.params:
                bnode = next(self.generator)
                yield self.n3(variable, pred, bnode)
                yield self.n3(bnode, RDFS.subClassOf,
                        self.namespace[t._operator.name])
                for i, param in enumerate(t.params, start=1):
                    yield from self.sparql_type(bnode, param, index=i)
            else:
                yield self.n3(
                    variable,
                    pred / (RDFS.subClassOf * ZeroOrMore),
                    self.namespace[t._operator.name])

    def trace(self, current: TransformationFlow,
            after: Optional[Unit] = None) -> Iterator[str]:
        """
        Produce SPARQL constraints by tracing the paths between each unit.
        """
        if isinstance(current, Unit):
            if after:
                if current.skip:
                    if isinstance(after, UnitType) and \
                            isinstance(current, UnitOperator):
                        mod = ZeroOrMore
                    else:
                        mod = OneOrMore
                else:
                    if isinstance(after, UnitType) and \
                            isinstance(current, UnitOperator):
                        self.variable[current] = self.variable[after]
                    mod = None

                if self.variable[current] != self.variable[after]:
                    yield self.n3(
                        self.variable[after],
                        (~TA.feeds) * mod if mod else ~TA.feeds,
                        self.variable[current])
            else:
                yield self.n3(
                    rdflib.Variable("workflow"),
                    TA.result,
                    self.variable[current]
                )

            if isinstance(current, UnitOperator):
                yield self.n3(
                    self.variable[current],
                    TA.via,
                    self.namespace[current.operator.name])
            else:
                assert isinstance(current, UnitType)
                yield from self.sparql_type(
                    self.variable[current],
                    current.type)

        elif isinstance(current, Parallel):
            for item in current.items:
                yield from self.trace(item, after)

        elif isinstance(current, Choice):
            yield " UNION ".join(
                "{" + "\n".join(self.trace(item, after)) + "}"
                for item in current.items
            )

        else:
            assert isinstance(current, Serial)

            for item in current.items[:-1]:
                assert isinstance(item, Unit)  # TODO remove when possible
                yield from self.trace(item, after)
                after = item
            yield from self.trace(current.items[-1], after)

    def sparql(self) -> str:
        """
        Convert this Flow to a SPARQL query.
        """

        query = [
            "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>",
            "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>",
            "PREFIX ta: <https://github.com/quangis/transformation-algebra#>",
            f"PREFIX {self.prefix}: <{self.namespace}>",
            "SELECT ?workflow ?description WHERE {",
            "?workflow rdf:type ta:Transformation.",
            "OPTIONAL {?workflow rdfs:comment ?description}",
        ]
        query.extend(self.trace(self.flow))
        query.append("} GROUP BY ?workflow ?description")
        return "\n".join(query)

        # TODO return sparql.Query instead, to avoid parsing at all. This does
        # not avoid parsing:
        # return sparql.prepareQuery("\n".join(query),
        #         initNs={'ta': TA, 'rdf': RDF, 'rdfs': RDFS}
        # )


class TransformationFlow(ABC):
    """
    A flow captures some relevant aspects of a conceptual process, in terms of
    the sequence of elements that must occur in it. For example, the following
    flow holds that there must be datatypes `A` and `B` that are fed to an
    operation f that eventually results in a datatype `C`:

    Serial(C, ..., f, [A, B])

    Note that the flow is 'reversed' (from output to input). This allows for a
    convenient tree-like notation, but it may trip you up.

    Furthermore, for succinct notation, nested tuples are interpreted as
    `Serial` and lists as `Parallel` transformation flows. The ellipsis
    indicates we may skip any number of steps.
    """
    # The same can also be described in terms of semantic linear time logic
    # formulae (SLTL), but since we will be using SPARQL to search through
    # workflows, the approach chosen here makes for a straightforward
    # translation.

    @abstractmethod
    def set_skip(self) -> None:
        raise NotImplementedError

    @staticmethod
    def shorthand(value: NestedFlow) -> TransformationFlow:
        """
        Translate shorthand data structures (ellipsis for skips, tuples for
        serials, lists for parallels) to real flows.
        """
        assert value != ..., "ellipses may only occur in serials"
        if isinstance(value, Operator):
            return UnitOperator(value)
        elif isinstance(value, Type):
            return UnitType(value)
        elif isinstance(value, tuple):
            return Serial(*value)
        elif isinstance(value, list):
            return Parallel(*value)
        elif isinstance(value, TransformationFlow):
            return value
        else:
            raise ValueError(
                f"{value} cannot be interpreted as a TransformationFlow")


class Unit(TransformationFlow):
    """
    A unit represents a single data instance in the flow.
    """

    def __init__(self):
        # Does a skip occur after this unit? (i.e. is `...` specified *before*
        # this unit in the reversed sequence)
        self.skip = False

    def set_skip(self) -> None:
        self.skip = True


class UnitType(Unit):
    def __init__(self, type: Type):
        self.type = type
        super().__init__()


class UnitOperator(Unit):
    def __init__(self, operator: Operator):
        if operator.definition:
            raise ValueError("any operation in a flow query must be primitive")

        self.operator = operator
        super().__init__()


class Serial(TransformationFlow):
    """
    Indicate the order in which transformation elements must occur.
    """

    def __init__(self, *items: NestedFlow):
        self.items = []
        skip: bool = False
        for current in items:
            if current == ...:
                skip = True
            else:
                item = TransformationFlow.shorthand(current)
                if skip:
                    item.set_skip()
                self.items.append(item)
                skip = False

    def set_skip(self) -> None:
        self.items[0].set_skip()


class Parallel(TransformationFlow):
    """
    Indicate which transformation paths must occur conjunctively. That is,
    every path must occur somewhere --- possibly on distinct branches, possibly
    on the same branch.
    """

    def __init__(self, *items: NestedFlow):
        self.items = [TransformationFlow.shorthand(x) for x in items]

    def set_skip(self) -> None:
        for x in self.items:
            x.set_skip()


class Choice(TransformationFlow):
    """
    Indicate which transformation paths can occur disjunctively. That is, at
    least one path must occur somewhere.
    """

    def __init__(self, *items: NestedFlow):
        self.items = [TransformationFlow.shorthand(x) for x in items]

    def set_skip(self) -> None:
        for x in self.items:
            x.set_skip()

