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
from transformation_algebra.type import Type, Schema


def typed(
        τ: Union[Type, Callable[..., Type]]) -> Callable[..., Definition]:
    """
    A decorator for defining transformations in terms of other transformations.
    Despite appearances, the provided function is *not* an implementation of
    the transformation: it merely represents a decomposition into more
    primitive conceptual building blocks.
    """
    τ2: Type = τ if isinstance(τ, Type) else Schema(τ)

    def wrapper(func: Callable[..., Expr]) -> Definition:
        return Definition(
            name=func.__name__,
            type=τ2,
            description=func.__doc__,
            composition=func
        )
    return wrapper


class Expr(ABC):
    def __init__(self):
        self.type = None

    def __repr__(self) -> str:
        return self.tree()

    def __call__(self, *args: Expr) -> Expr:
        return reduce(Compound, args, self)

    def tree(self, lvl: str = "") -> str:
        """
        Obtain a tree representation of this expression, using Unicode block
        characters to draw the tree.
        """
        if isinstance(self, Simple):
            return f"╼ {self} : {self.definition.type}"
        elif isinstance(self, Compound):
            return (
                f"{self.type}\n"
                f"{lvl} ├─{self.f.tree(lvl+' │ ')}\n"
                f"{lvl} └─{self.x.tree(lvl+'   ')}"
            )
        raise ValueError

    def substitute(self, label: str, expr: Expr) -> Expr:
        """
        Replace the given expression for all expressions with the given label.
        """
        if isinstance(self, Simple):
            if self.label == label:
                self.type.plain().unify(expr.type.plain())
                return expr
            else:
                return self
        elif isinstance(self, Compound):
            self.f = self.f.substitute(label, expr)
            self.x = self.x.substitute(label, expr)
            return self
        raise ValueError

    def primitive(self) -> Expr:
        """
        Expand this expression into its simplest form.
        """
        f = self.partial_primitive()
        if isinstance(f, Expr):
            return f
        raise RuntimeError("cannot express partial primitive")

    def partial_primitive(self) -> Union[Expr, PartialExpr]:
        """
        Return an expression if the resulting expression is a primitive, or a
        function returning an expression if the expression is incomplete.
        """
        if isinstance(self, Simple):
            if self.definition.composition:
                return PartialExpr(self.definition.composition).close()
            else:
                return self
        elif isinstance(self, Compound):
            f = self.f.partial_primitive()
            x = self.x.partial_primitive()
            return f.partial_apply(x)
        raise ValueError

    def partial_apply(
            self,
            x: Union[Expr, PartialExpr]) -> Union[Expr, PartialExpr]:
        if isinstance(x, Expr):
            return Compound(self, x)
        elif isinstance(x, PartialExpr):
            raise RuntimeError(
                "cannot apply partial expression to primitive expression")


class PartialExpr(object):
    """
    An incomplete expression that needs to be supplied with arguments to become
    a full expression.
    """

    def __init__(self, composition: Callable[..., Expr]):
        self.composition = composition

    def close(self) -> Union[Expr, PartialExpr]:
        n = len(signature(self.composition).parameters)
        if n > 0:
            return self
        else:
            return self.composition()

    def partial_apply(
            self,
            x: Union[Expr, PartialExpr]) -> Union[Expr, PartialExpr]:
        if isinstance(x, Expr):
            self.composition = partial(self.composition, x)
        elif isinstance(x, PartialExpr):
            pass
        raise ValueError


class Simple(Expr):
    """
    Represents either a transformation or input data in an algebra expression.
    Represents input data for a transformation algebra expression. This can
    also be seen as a *typed variable* in an expression.
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


class Compound(Expr):
    """
    Represents an *application* of a transformation, capturing its output.
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


class Definition(object):
    """
    A definition represents a non-instantiated data input or transformation.
    """

    def __init__(
            self,
            name: str,
            type: Type,
            labelled: bool = False,
            description: Optional[str] = None,
            composition: Optional[Callable[..., Expr]] = None):
        self.name = name.rstrip("_")
        self.type = type
        self.labelled = labelled  # are instances identified or anonymous?
        self.description = description  # human-readable
        self.composition = composition  # non-primitive transformations may be
        # composed of other transformations
        self.is_input = not self.type.is_function()

    def __str__(self) -> str:
        return f"{self.name or 'anonymous'} : {self.type}"

    def instance(self, identifier: Optional[str] = None) -> Expr:
        return Simple(self, label=identifier)


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
            None, 2, pp.opAssoc.LEFT, lambda s, l, t: reduce(Compound, t[0])
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
                if not v.name:
                    v.name = k.rstrip("_")
                definitions.append(v)
        return TransformationAlgebra(*definitions)
