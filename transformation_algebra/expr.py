"""
Classes to define generic transformation algebras.
"""

from __future__ import annotations

from abc import ABC
import pyparsing as pp
from functools import reduce
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
        return str(self)

    def tree(self, lvl: str = "") -> str:
        if isinstance(self, Simple):
            return f"═ {self} : {self.definition.type}"
        elif isinstance(self, Compound):
            return (
                f"{self.type}\n"
                f"{lvl} ├─{self.f.tree(lvl+' │ ')}\n"
                f"{lvl} └─{self.x.tree(lvl+'   ')}"
            )
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
        self.label = label

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
        self.f = f
        self.x = x
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
        self.name = name
        self.type = type
        self.labelled = labelled  # are instances identified or anonymous?
        self.description = description  # human-readable
        self.composition = composition  # non-primitive transformations may be
        # composed of other transformations
        self.is_input = not self.type.is_function()

    def instance(self, identifier: Optional[str] = None) -> Expr:
        return Simple(self, label=identifier)


class TransformationAlgebra(object):
    def __init__(self, **signatures: Type):
        self.parser: Optional[pp.Parser] = None
        self.definitions: Dict[str, Definition] = {}

        for k, v in signatures.items():
            self.definitions[k] = Definition(name=k, type=v)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return "\n".join(
            f"{k}: {v}" for k, v in self.definitions.items()) + "\n"

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
        Create transformation algebra from an object, filtering out the
        relevant parts: those Type values whose keys start with lowercase.
        """
        return TransformationAlgebra(**{
            k.rstrip("_"): v for k, v in obj.items()
            if k[0].islower() and isinstance(v, Type)
        })
