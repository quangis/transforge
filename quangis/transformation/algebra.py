"""
Classes to define generic transformation algebras.
"""

from __future__ import annotations

import pyparsing as pp
from functools import reduce
from typing import List, Union, Optional

from quangis import error
from quangis.transformation.type import Term, Schema


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

    def apply(self: Expr, arg: Expr) -> Expr:
        try:
            return Expr([self, arg], self.type.apply(arg.type))
        except error.AlgebraTypeError as e:
            e.add_expression(self, arg)
            raise e


class TransformationAlgebra(object):
    def __init__(self, **functions: Schema):
        self.functions = functions
        self.parser: Optional[pp.Parser] = None

    def generate_parser(self) -> pp.Parser:
        sigs = self.functions
        identifier = pp.Word(pp.alphanums + ':_').setName('identifier')

        function = pp.MatchFirst(
            pp.CaselessKeyword(k) + d.data_args * identifier
            if d.data_args else
            pp.CaselessKeyword(k)
            for k, d in sigs.items()
        ).setParseAction(lambda s, l, t: Expr(t, sigs[t[0]].instance()))

        return pp.infixNotation(function, [(
            None, 2, pp.opAssoc.LEFT, lambda s, l, t: reduce(Expr.apply, t[0])
        )])

    def parse(self, string: str) -> Expr:
        if not self.parser:
            self.parser = self.generate_parser()
        return self.parser.parseString(string, parseAll=True)[0]
