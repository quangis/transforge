"""
Classes to define generic transformation algebras.
"""

from __future__ import annotations

from abc import ABC
import pyparsing as pp
from functools import reduce
from typing import Optional, Any, Dict

from transformation_algebra import error
from transformation_algebra.type import Type


class Expr(ABC):
    def __init__(self):
        self.type = None

    def __repr__(self) -> str:
        return str(self)


class Input(Expr):
    """
    Represents input data for a transformation algebra expression. This can
    also be seen as a *typed variable* in an expression.
    """

    def __init__(
            self,
            token: str,
            type: Type,
            identifier: Optional[str] = None):
        self.token = token
        self.identifier = identifier
        self.type = type

        if self.type.is_function():
            raise RuntimeError("Must not be a function type")
        if any(self.type.plain().variables()):
            raise RuntimeError("Input types must be fully qualified")

    def __str__(self) -> str:
        if self.identifier:
            return f"{self.token} {self.identifier} : {self.type}"
        return self.token


class Transformed(Expr):
    """
    Represents an *application* of a transformation.
    """

    def __init__(self, f: Expr, x: Expr):
        self.f = f
        self.x = x
        try:
            self.type = f.type.instance().apply(x.type.instance())
        except error.AlgebraTypeError as e:
            e.add_expression(f, x)
            raise e

    def __str__(self) -> str:
        return f"({self.f} {self.x} : {self.type})"


class Transformation(Expr):
    def __init__(self, token: str, type: Type):
        self.token = token
        self.type = type

        if not self.type.is_function():
            raise RuntimeError("Must be a function type")

    def __str__(self) -> str:
        return self.token


class TransformationAlgebra(object):
    def __init__(self, **signatures: Type):
        self.parser: Optional[pp.Parser] = None
        self.transformations: Dict[str, Type] = {}
        self.inputs: Dict[str, Type] = {}

        for k, v in signatures.items():
            if v.is_function():
                self.transformations[k] = v
            else:
                self.inputs[k] = v

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return "\n".join(
            f"{k}: {v}"
            for k, v in dict(**self.transformations, **self.inputs).items()
        ) + "\n"

    def generate_parser(self) -> pp.Parser:

        ident = pp.Word(pp.alphanums + ':_').setName('identifier')

        transformation = pp.MatchFirst(
            pp.CaselessKeyword(kw) for kw in self.transformations
        ).setParseAction(
            lambda s, l, t: Transformation(
                token=t[0],
                type=self.transformations[t[0]],
            )
        )

        data = pp.MatchFirst(
            pp.CaselessKeyword(kw) + pp.Optional(ident) for kw in self.inputs
        ).setParseAction(
            lambda s, l, t: Input(
                token=t[0],
                type=self.inputs[t[0]],
                identifier=t[1] if len(t) > 1 else None
            )
        )

        return pp.infixNotation(transformation | data, [(
            None, 2, pp.opAssoc.LEFT, lambda s, l, t: reduce(Transformed, t[0])
        )])

    def parse(self, string: str) -> Expr:
        if not self.parser:
            self.parser = self.generate_parser()
        expr = self.parser.parseString(string, parseAll=True)[0]
        return expr

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
