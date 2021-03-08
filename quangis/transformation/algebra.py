"""
Classes to define generic transformation algebras.
"""

from __future__ import annotations

import pyparsing as pp
from functools import reduce
from typing import List, Union, Optional, Tuple, Any, Dict

from quangis import error
from quangis.transformation.type import Term, Type


class Expr(object):
    """
    An expression of a transformation algebra.
    """

    def __init__(self, tokens: List[Union[str, Expr]], type: Term):
        self.tokens = tokens
        self.type = type

    def __str__(self, top_level=True) -> str:
        expr = ' '.join(
            t if isinstance(t, str) else t.__str__(False)
            for t in self.tokens
        )
        if top_level:
            return f"{expr} : {self.type}"
        elif len(self.tokens) == 1:
            return f"{expr}"
        else:
            return f"({expr})"

    def resolve(self) -> None:
        self.type = self.type.resolve()
        for token in self.tokens:
            if isinstance(token, Expr):
                token.resolve()

    def apply(self: Expr, arg: Expr) -> Expr:
        try:
            return Expr([self, arg], self.type.apply(arg.type))
        except error.AlgebraTypeError as e:
            e.add_expression(self, arg)
            raise e


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
            lambda s, l, t: Expr(t, self.transformations[t[0]].instance())
        )

        data = pp.MatchFirst(
            pp.CaselessKeyword(kw) + pp.Optional(ident) for kw in self.inputs
        ).setParseAction(
            lambda s, l, t: Expr(t, self.inputs[t[0]].instance())
        )

        return pp.infixNotation(transformation | data, [(
            None, 2, pp.opAssoc.LEFT, lambda s, l, t: reduce(Expr.apply, t[0])
        )])

    def parse(self, string: str) -> Expr:
        if not self.parser:
            self.parser = self.generate_parser()
        expr = self.parser.parseString(string, parseAll=True)[0]
        expr.resolve()
        return expr

    @staticmethod
    def from_dict(obj: Dict[str, Any]) -> TransformationAlgebra:
        """
        Create transformation algebra from an object, filtering out the
        relevant keys.
        """
        return TransformationAlgebra(**{
            k.rstrip("_"): v for k, v in obj.items()
            if k[0].islower() and isinstance(v, Type)
        })
