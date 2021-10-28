"""
This module allows you to define a transformation language as a collection of
types and operators. It also handles parsing expressions of the algebra.
"""

from __future__ import annotations

from enum import Enum, auto
from itertools import groupby, chain
from typing import Optional, Union, Iterator, Any

from transformation_algebra import error
from transformation_algebra.type import \
    TypeOperator, TypeInstance, TypeVariable
from transformation_algebra.expr import \
    Operator, Expr, Application, Source


class Language(object):
    def __init__(self, scope: dict[str, Any] = {}):
        self.operators: dict[str, Operator] = dict()
        self.types: dict[str, TypeOperator] = dict()
        if scope:
            self.add_scope(scope)

    def add_scope(self, scope: dict[str, Any]) -> None:
        """
        For convenience, you may add types and operations in bulk via a
        dictionary. This allows you to simply pass `globals()` or `locals()`.
        Irrelevant items are filtered out without complaint.
        """
        for k, v in dict(scope).items():
            if isinstance(v, (Operator, TypeOperator)):
                self.add(item=v, name=k)

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
            assert isinstance(key.name, str)
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

        sources: dict[str, Source] = dict()
        stack: list[Optional[Expr]] = [None]
        stackT: list[TypeInstance | TypeOperator | list[TypeInstance]] = []
        type_mode: bool = False

        for token in tokenize(string, "(,):;"):
            if type_mode:
                if token == "(":
                    stackT.append([])
                elif token in "),":
                    unit = stackT.pop()
                    assert isinstance(unit, TypeInstance)
                    params = stackT.pop()
                    assert isinstance(params, list)
                    params.append(unit)
                    stackT.append(params)
                    if token == ")":
                        params = stackT.pop()
                        assert isinstance(params, list)
                        op = stackT.pop()
                        assert isinstance(op, TypeOperator)
                        stackT.append(op(*params))
                elif token == "_":
                    stackT.append(TypeVariable())
                else:
                    try:
                        op = self.types[token]
                    except KeyError as e:
                        raise error.Undefined(token) from e
                    stackT.append(op() if op.arity == 0 else op)
                if len(stackT) == 1 and isinstance(stackT[0], TypeInstance):
                    type_mode = False
                    t = stackT.pop()
                    assert isinstance(t, TypeInstance)
                    previous = stack[-1]
                    assert isinstance(previous, Expr)
                    previous.type.unify(t, subtype=True)
                continue

            if token in "(,)":
                if token in "),":
                    try:
                        y = stack.pop()
                        if y:
                            x = stack.pop()
                            stack.append(Application(x, y) if x else y)
                    except IndexError as e:
                        raise error.LBracketMismatch from e
                if token in "(,":
                    stack.append(None)
            elif token == ":":
                type_mode = True
            elif token == ";":
                stack.clear()
                stack.append(None)
            else:
                previous = stack.pop()
                if previous and isinstance(previous, Source):
                    current = None
                    previous.label = token
                    if token in sources:
                        sources[token].type.unify(previous.type)
                    else:
                        sources[token] = previous
                    stack.append(previous)
                    continue
                if token == "-":
                    current = Source()
                elif token.isnumeric():
                    try:
                        current = sources[token]
                    except KeyError:
                        current = sources[token] = Source(label=token)
                else:
                    try:
                        current = self.operators[token].instance()
                    except KeyError as e:
                        raise error.Undefined(token) from e
                if previous and current:
                    current = Application(previous, current)
                stack.append(current)

        if len(stack) == 1:
            result = stack[0]
            if not result:
                raise error.Empty
            else:
                return result
        else:
            raise error.RBracketMismatch


def tokenize(string: str, specials: str = "") -> Iterator[str]:
    """
    Break up a string into special characters and identifiers (that is, any
    sequence of non-special characters). Skip spaces.
    """
    for group, tokens in groupby(string,
            lambda x: -2 if x.isspace() else specials.find(x)):
        if group == -1:
            yield "".join(tokens)
        elif group >= 0:
            yield from tokens
