"""
This module allows you to define a transformation algebra as a collection of
types and operations. It also handles parsing expressions of the algebra.
"""

from __future__ import annotations

from enum import Enum, auto
from itertools import groupby, chain
from typing import Optional, Union, Iterator

from transformation_algebra import error
from transformation_algebra.type import \
    TypeOperator, TypeInstance, Function
from transformation_algebra.expr import \
    Operator, Expr, Application, Source


class TransformationAlgebra(object):
    def __init__(self, **kwargs):
        self.operators: dict[str, Operator] = dict()
        self.types: dict[str, TypeOperator] = dict()

        # For convenience, the user may specify the operations and types on
        # initialization. When fed `globals()`, irrelevant items are filtered
        # out without complaint
        filter_irrelevant_args = '__builtins__' in kwargs
        for k, v in kwargs.items():
            if isinstance(v, (Operator, TypeOperator)):
                self.add(item=v, name=k)
            elif not filter_irrelevant_args:
                raise RuntimeError

    def add(self, item: Operator | TypeOperator, name: Optional[str] = None):

        # The item must already have a name or be named here
        if name:
            name = name.rstrip("_")
            if not item.name:
                item.name = name
            elif item.name != name:
                raise ValueError("has conflicting names")
        elif not item.name:
            raise ValueError("must have a name")
        else:
            name = item.name

        if name in self.operators or name in self.types:
            raise ValueError("already exists")

        if isinstance(item, Operator):
            self.operators[name] = item
        else:
            self.types[name] = item

    def __getattr__(self, name: str) -> Operator | TypeOperator:
        result = self.operators.get(name) or self.types.get(name)
        if result:
            return result
        else:
            raise AttributeError

    def __setattr__(self, name: str, value: Operator | TypeOperator) -> None:
        if isinstance(value, (Operator, TypeOperator)):
            self.add(value, name)
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

        for token in tokenize(string, "(,)"):
            assert not token.isspace()
            if token == ')':
                try:
                    y = stack.pop()
                    if y:
                        x = stack.pop()
                        stack.append(Application(x, y) if x else y)
                except IndexError as e:
                    raise error.LBracketMismatch from e
            elif token == '(':
                stack.append(None)
            elif token == ',':
                try:
                    y = stack.pop()
                    if y:
                        x = stack.pop()
                        stack.append(Application(x, y) if x else y)
                    stack.append(None)
                except IndexError as e:
                    raise error.LBracketMismatch from e
            else:
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


def tokenize(text: str, chars: str = "") -> Iterator[str]:
    """
    Break up a string into special characters and identifiers (that is, any
    sequence of non-special characters). Skip spaces.
    """
    for group, tokens in groupby(text,
            lambda x: -2 if x.isspace() else chars.find(x)):
        if group == -1:
            yield "".join(tokens)
        elif group >= 0:
            yield from tokens
