"""
This module allows you to define a transformation algebra as a collection of
types and operations. It also handles parsing expressions of the algebra.
"""

from __future__ import annotations

from enum import Enum, auto
from itertools import groupby, chain
from typing import Optional, Union

from transformation_algebra import error
from transformation_algebra.type import \
    TypeOperator, TypeInstance, Function
from transformation_algebra.expr import \
    Operator, Expr, Application, Source


class TransformationAlgebra(object):
    def __init__(self):
        self.operators: dict[str, Operator] = dict()
        self.types: dict[str, TypeOperator] = dict()

    def insert(self, x: Operator | TypeOperator, name: Optional[str] = None):

        # The operation must already have a name or be named here
        if name:
            name = name.rstrip("_")
            if not x.name:
                x.name = name
            elif x.name != name:
                raise ValueError("operation has conflicting names")
        elif not x.name:
            raise ValueError("operation must have a name")
        else:
            name = x.name

        if isinstance(x, Operator):
            self.operators[name] = x
        else:
            self.types[name] = x

    def add(self, *nargs: Operator, **kwargs: Operator) -> None:
        """
        Add data and operation definitions to this transformation algebra. Note
        that names ending with an underscore will be stripped of that symbol.
        """

        for k, v in chain(kwargs.items(), ((v.name, v) for v in nargs)):
            if isinstance(v, Operator):
                self.insert(v, k)
            elif isinstance(v, TypeOperator):
                self.insert(v, k)
            else:
                # Only if we were fed globals() will we automatically filter
                # out irrelevant types without complaining
                assert '__builtins__' in kwargs

    def add_types(self, *nargs, **kwargs):
        self.add(*nargs, **kwargs)

    def __getattr__(self, name: str) -> Operator | TypeOperator:
        result = self.operators.get(name) or self.types.get(name)
        if result:
            return result
        else:
            raise AttributeError

    def __setattr__(self, name: str, value: Operator | TypeOperator) -> None:
        if isinstance(value, (Operator, TypeOperator)):
            self.insert(value, name)
        else:
            super().__setattr__(name, value)

    def __contains__(self, key: Union[str, Operator, TypeOperator]) -> bool:
        if isinstance(key, str):
            return key in self.operators or key in self.types
        elif isinstance(key, TypeOperator):
            return self.types.get(key.name) is key
        elif isinstance(key, Operator):
            assert isinstance(key.name, str)
            return self.operators.get(key.name) is key
        else:
            return False

    def validate(self) -> None:
        # Validation can only happen once all operations have been defined. If
        # we did it at define-time, it would lead to issues --- see issue #3

        for op in self.operators.values():
            # Check declared types of operations against their inferred type
            op.validate_type()

            # Check that every type is known to the algebra
            for t in op.type.instance().operators():
                if t not in self:
                    raise ValueError

    def parse(self, string: str) -> Expr:
        # This used to be done via pyparsing, but the structure is so simple
        # that I opted to remove the dependency --- this is *much* faster

        labels: dict[str, TypeInstance] = dict()
        stack: list[Optional[Expr]] = [None]

        for token_group, chars in groupby(string, Token.ize):
            if token_group is Token.RPAREN:
                for rparen in chars:
                    try:
                        y = stack.pop()
                        if y:
                            x = stack.pop()
                            stack.append(Application(x, y) if x else y)
                    except IndexError as e:
                        raise error.LBracketMismatch from e
            elif token_group is Token.LPAREN:
                for lparen in chars:
                    stack.append(None)
            elif token_group is Token.COMMA:
                try:
                    y = stack.pop()
                    if y:
                        x = stack.pop()
                        stack.append(Application(x, y) if x else y)
                    stack.append(None)
                except IndexError as e:
                    raise error.LBracketMismatch from e
            elif token_group is Token.IDENT:
                token = "".join(chars)
                previous = stack.pop()
                if previous and isinstance(previous, Source):
                    previous.label = token
                    if token in labels:
                        labels[token].unify(previous.type)
                    else:
                        labels[token] = previous.type
                    stack.append(previous)
                else:
                    try:
                        current = self.operators[token].instance()
                    except KeyError as e:
                        raise error.Undefined(token) from e
                    else:
                        if previous:
                            current = Application(previous, current)
                        stack.append(current)

        if len(stack) == 1:
            result = stack[0]
            if not result:
                raise error.Empty
            else:
                assert result.type.operator != Function
                result.rename()
                return result
        else:
            raise error.RBracketMismatch


class Token(Enum):
    LPAREN = auto()
    RPAREN = auto()
    SPACE = auto()
    IDENT = auto()
    COMMA = auto()

    @staticmethod
    def ize(char: str) -> Token:
        if ord(char) == 40:
            return Token.LPAREN
        elif ord(char) == 41:
            return Token.RPAREN
        elif ord(char) == 44:
            return Token.COMMA
        elif char.isspace():
            return Token.SPACE
        else:
            return Token.IDENT
