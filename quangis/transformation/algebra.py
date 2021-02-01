"""
Classes to define generic transformation algebras.
"""

from __future__ import annotations

import pyparsing as pp
from functools import reduce
from typing import List, Iterable, Union, Optional

from quangis import error
from quangis.transformation.type import AlgebraType, Definition, TypeOperator


class Expr(object):
    """
    An expression of a transformation algebra.
    """

    def __init__(self, tokens: List[Union[str, Expr]], type: AlgebraType):
        self.tokens = tokens
        self.type = type

    def __str__(self) -> str:
        if isinstance(self.type, TypeOperator) and self.type.name == 'function':
            return f"({' '.join(str(t) for t in self.tokens)})"
        else:
            return f"({' '.join(str(t) for t in self.tokens)} : {self.type})"

    def apply(self: Expr, arg: Expr) -> Expr:
        try:
            return Expr([self, arg], self.type.apply(arg.type))
        except error.TypeMismatch as e:
            e.t1 = self
            e.t2 = arg
            raise e
        except error.NonFunctionApplication as e:
            e.fn = self
            e.arg = arg
            raise e


class TransformationAlgebra(object):
    """
    To define a transformation algebra, make an instance of this class. Any
    properties (starting with lowercase) that are `Definition`s or can be
    translated to such via its constructor will be considered functions of the
    transformation algebra.
    """

    def __init__(self):
        self.parser: Optional[pp.Parser] = None

    def __setattr__(self, prop, value):
        """
        Makes sure definitions do not have to be tediously written out: any
        lowercase attribute containing a type, or a tuple beginning with a
        type, will also be converted into a `Definition`.
        """
        if prop[0].islower() and (isinstance(value, AlgebraType) or (
                    isinstance(value, tuple) and
                    isinstance(value[0], AlgebraType))
                ):
            # Invalidate parser
            self.parser = None
            name = "in" if prop == "in_" else prop
            args = value if isinstance(value, Iterable) else (value,)
            super().__setattr__(prop, Definition(name, *args))
        else:
            super().__setattr__(prop, value)

    def definitions(self) -> Iterable[Definition]:
        """
        Obtain the definitions of this algebra.
        """
        for attr in dir(self):
            val = getattr(self, attr)
            if isinstance(val, Definition):
                yield val

    def make_parser(self) -> pp.Parser:

        defs = {d.name: d for d in self.definitions()}

        identifier = pp.Word(pp.alphanums + ':_').setName('identifier')

        function = pp.MatchFirst(
            pp.CaselessKeyword(d.name) + d.data * identifier
            if d.data else
            pp.CaselessKeyword(d.name)
            for k, d in defs.items()
        ).setParseAction(lambda s, l, t: Expr(t, defs[t[0]].instance()))

        return pp.infixNotation(function, [(
            None, 2, pp.opAssoc.LEFT, lambda s, l, t: reduce(Expr.apply, t[0])
        )])

    def parse(self, string: str) -> Expr:
        if not self.parser:
            self.parser = self.make_parser()
        return self.parser.parseString(string, parseAll=True)[0]
