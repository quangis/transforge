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
    def __init__(self, **functions: Tuple[Type, int]):
        self.functions = functions
        self.parser: Optional[pp.Parser] = None

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return "\n".join([
            f"{k}: {signature}" for k, (signature, _) in self.functions.items()
        ]) + "\n"

    def generate_parser(self) -> pp.Parser:

        identifier = pp.Word(pp.alphanums + ':_').setName('identifier')

        function = pp.MatchFirst(
            pp.CaselessKeyword(keyword) + data_args * identifier
            if data_args else
            pp.CaselessKeyword(keyword)
            for keyword, (signature, data_args) in self.functions.items()
        ).setParseAction(
            lambda s, l, t: Expr(t, self.functions[t[0]][0].instance())
        )

        return pp.infixNotation(function, [(
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
        Create an transformation algebra from any dictionary, filtering out the
        relevant keys. Relevant keys are those that start with a lowercase
        symbol and are either a Type or a tuple starting with a Type.
        """
        algebra = TransformationAlgebra()
        for k, v in obj.items():
            if k[0].islower():
                k = k.rstrip("_")
                if isinstance(v, Type):
                    algebra.functions[k] = v, 0
                elif isinstance(v, tuple) and len(v) == 2 \
                        and isinstance(v[0], Type) and isinstance(v[1], int):
                    algebra.functions[k] = v[0], v[1]
        return algebra
