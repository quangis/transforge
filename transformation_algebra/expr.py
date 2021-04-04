"""
Classes to define generic transformation algebras.
"""

from __future__ import annotations

from abc import ABC
import pyparsing as pp
from functools import reduce, partial
from inspect import signature
from typing import Optional, Any, Dict, Callable, Union

from transformation_algebra import error
from transformation_algebra.type import Type, TypeSchema


class PartialExpr(ABC):
    """
    An expression that may contain abstractions.
    """

    def __call__(self, *args: Union[PartialExpr, Definition]) -> PartialExpr:
        args2 = (e.instance() if isinstance(e, Definition) else e for e in args)
        return reduce(PartialExpr.partial_apply, args2, self).complete()

    def partial_apply(self, x: PartialExpr) -> PartialExpr:
        f = self.complete()
        if isinstance(f, Abstraction):
            f.composition = partial(f.composition, x)
            return f.complete()
        elif isinstance(x, Abstraction):
            raise RuntimeError(
                "cannot apply abstraction to primitive expression")
        else:
            assert isinstance(f, Expr) and isinstance(x, Expr)
            return Application(f, x)

    def complete(self) -> PartialExpr:
        """
        Once all parameters of an abstraction have been provided, turn into
        full expression.
        """
        if isinstance(self, Abstraction):
            n = len(signature(self.composition).parameters)
            if n == 0:
                return self.composition()
        return self


class Expr(PartialExpr):
    def __repr__(self) -> str:
        return self.tree()

    def tree(self, lvl: str = "") -> str:
        """
        Obtain a tree representation using Unicode block characters.
        """
        if isinstance(self, Base):
            return f"╼ {self} : {self.definition.type}"
        elif isinstance(self, Application):
            return (
                f"{self.type}\n"
                f"{lvl} ├─{self.f.tree(lvl + ' │ ')}\n"
                f"{lvl} └─{self.x.tree(lvl + '   ')}"
            )
        raise ValueError

    def substitute(self, label: str, expr: Expr) -> Expr:
        """
        Replace the given expression for all expressions with the given label.
        """
        if isinstance(self, Base):
            if self.label == label:
                self.type.plain().unify(expr.type.plain())
                return expr
            else:
                return self
        elif isinstance(self, Application):
            self.f = self.f.substitute(label, expr)
            self.x = self.x.substitute(label, expr)
            return self
        raise ValueError

    def primitive(self) -> Expr:
        """
        Expand this expression into its simplest form.
        """
        f = self.partial_primitive()
        if isinstance(f, Abstraction):
            raise RuntimeError("cannot express partial primitive")
        elif isinstance(f, Expr):
            return f
        raise ValueError

    def partial_primitive(self) -> PartialExpr:
        """
        Expand this expression into its simplest form. May contain
        abstractions.
        """
        if isinstance(self, Base):
            if self.definition.composition:
                return Abstraction(self.definition.composition).complete()
            else:
                return self
        elif isinstance(self, Application):
            f = self.f.partial_primitive()
            x = self.x.partial_primitive()
            return f.partial_apply(x)
        raise ValueError


class Base(Expr):
    """
    A base expression represents either a single transformation or a data
    input. Base expressions may be unfolded into multiple applications of
    primitive transformations. Data input can be seen as a typed variable in an
    expression: in this case, it should be labelled and substituted.
    """

    def __init__(self, definition: Definition, label: Optional[str] = None):
        self.definition = definition
        self.type = definition.type.instance()
        self.label: Optional[str] = label

    def __str__(self) -> str:
        if self.label:
            return (f"{self.definition.name} {self.label}")
        else:
            return f"{self.definition.name}"


class Application(Expr):
    """
    A comlex expression, representing an application of the transformation in
    its first argument to the expression in its second argument.
    """

    def __init__(self, f: Expr, x: Expr):
        self.f: Expr = f
        self.x: Expr = x
        try:
            self.type = f.type.apply(x.type)
        except error.AlgebraTypeError as e:
            e.add_expression(f, x)
            raise e

    def __str__(self) -> str:
        return f"({self.f} {self.x})"


class Abstraction(PartialExpr):
    """
    An incomplete expression that needs to be supplied with arguments. Not
    normally part of an expression tree --- only used for expanding primitives.
    """

    def __init__(self, composition: Callable[..., PartialExpr]):
        self.composition: Callable[..., PartialExpr] = composition


class Definition(ABC):
    """
    A definition represents a non-instantiated data input or transformation.
    """

    def __init__(
            self,
            type: Union[Type, Callable[..., Type]],
            term: Optional[Callable[..., PartialExpr]] = None,
            name: Optional[str] = None,
            descr: Optional[str] = None,
            label: bool = False):
        self.name = name
        self.type = type if isinstance(type, Type) else TypeSchema(type)
        self.labelled = label  # are instances identified or anonymous?
        self.description = descr  # human-readable
        self.composition = term  # non-primitive transformations may be
        # composed of other transformations
        self.is_input = not self.type.is_function()

    def __str__(self) -> str:
        return f"{self.name or 'anonymous'} : {self.type}"

    def __call__(self, *args: Union[Definition, PartialExpr]) -> PartialExpr:
        return self.instance().__call__(*args)

    def instance(self, identifier: Optional[str] = None) -> Expr:
        return Base(self, label=identifier)


class Operation(Definition):
    pass


class Data(Definition):
    pass


class TransformationAlgebra(object):
    def __init__(self, *definitions: Definition):
        self.parser: Optional[pp.Parser] = None
        self.definitions: Dict[str, Definition] = {}

        for d in definitions:
            self.definitions[d.name] = d

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return "\n".join(str(d) for d in self.definitions.values()) + "\n"

    def generate_parser(self) -> pp.Parser:

        label = pp.Word(pp.alphanums + ':_').setName('identifier')

        expr = pp.MatchFirst(
            pp.CaselessKeyword(d.name) + pp.Optional(label)
            if d.is_input else
            pp.CaselessKeyword(d.name)
            for d in self.definitions.values()
        ).setParseAction(
            lambda s, l, t: self.definitions[t[0]].instance(
                t[1] if len(t) > 1 else None)
        )

        return pp.infixNotation(expr, [(
            None, 2, pp.opAssoc.LEFT, lambda s, l, t: reduce(Application, t[0])
        )])

    def parse(self, string: str) -> Expr:
        if not self.parser:
            self.parser = self.generate_parser()
        expr = self.parser.parseString(string, parseAll=True)[0]
        return expr

    def tree(self, string: str) -> None:
        """
        Print a tree corresponding to the given algebra expression.
        """
        print(self.parse(string).tree())

    @staticmethod
    def from_dict(obj: Dict[str, Any]) -> TransformationAlgebra:
        """
        Create transformation algebra from an object, filtering out relevant
        definitions.
        """
        definitions = []
        for k, v in obj.items():
            if isinstance(v, Definition):
                v.name = k.rstrip("_")
                definitions.append(v)
        return TransformationAlgebra(*definitions)
