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
        """
        Initiate an empty transformation algebra.
        """
        self.definitions: dict[str, Operator] = {}
        self.types: set[TypeOperator] = set()

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return "\n".join(str(d) for d in self.definitions.values()) + "\n"

    def __contains__(self, key: Union[str, Operator, TypeOperator]) -> bool:
        if isinstance(key, TypeOperator):
            return key in self.types
        elif isinstance(key, Operator):
            assert key.name
            return self.definitions.get(key.name) is key
        else:
            assert isinstance(key, str)
            return key in self.definitions

    def __getitem__(self, key: str) -> Operator:
        return self.definitions[key]

    def __setitem__(self, key: str, value: Operator) -> None:
        self.definitions[key] = value

        for op in value.type.instance().operators():
            self.types.add(op)

    def __getattr__(self, name: str) -> Operator:
        result = self.definitions.get(name)
        if result:
            return result
        else:
            raise AttributeError(
                f"The algebra defines no type or operation '{name}'.")

    def __setattr__(self, name: str, value: Operator) -> None:
        if isinstance(value, Operator):
            assert value.name is None or value.name == name
            value.name = name
            self.definitions[name] = value
        super().__setattr__(name, value)

    def validate(self) -> None:
        # Validation only happens all operations have been defined. If we did
        # it at define-time, it would lead to issues --- see issue #3
        for d in self.definitions:
            if isinstance(d, Operator):
                d.validate_type()

    def add(self, *nargs: Operator, **kwargs: Operator) -> None:
        """
        Add data and operation definitions to this transformation algebra. Note
        that names ending with an underscore will be stripped of that symbol.
        """

        for k, v in chain(kwargs.items(),
                ((v.name, v) for v in nargs)):
            assert k is not None, f"unknown name for a {type(v)}"
            if isinstance(v, Operator):
                k = k.rstrip("_")
                assert v.name is None or k == v.name
                v.name = k
                self[k] = v
            else:
                # Only if we were fed globals() will we automatically filter
                # out irrelevant types without complaining
                assert '__builtins__' in kwargs

    def add_types(self, *nargs: TypeOperator, **kwargs: TypeOperator) -> None:
        """
        Explicitly notify the algebra of type operators that may be used.
        """
        for t in chain(nargs, kwargs.values()):
            self.types.add(t)

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
                        current = self[token].instance()
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
