"""
Classes to define generic transformation algebras.
"""

from __future__ import annotations

from enum import Enum, auto
from abc import ABC
from functools import reduce, partial
from itertools import groupby
from inspect import signature
from typing import Optional, Dict, Callable, Union, List

from transformation_algebra import error
from transformation_algebra.type import \
    Type, TypeVar, TypeSchema, TypeInstance, Function, _


class Definition(ABC):
    """
    A definition represents a non-instantiated data input or transformation.
    """

    def __init__(
            self,
            type: Union[Type, Callable[..., TypeInstance]],
            doc: Optional[str] = None):
        self.name: Optional[str] = None
        self.type = type if isinstance(type, Type) else TypeSchema(type)
        self.description = doc

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"{self.name or '[?]'} : {self.type}"

    def __call__(self, *args: Union[Definition, PartialExpr]) -> PartialExpr:
        return self.instance().__call__(*args)

    def instance(self, identifier: Optional[str] = None) -> Expr:
        return Base(self, label=identifier)


class Data(Definition):
    """
    The definition of a data input. An instance of such a definition is a base
    expression.
    """

    def __init__(self, *nargs, **kwargs):
        self.composition = None
        super().__init__(*nargs, **kwargs)
        assert not self.type.is_function()


class Operation(Definition):
    """
    The definition of a transformation. An instance of such a definition is a
    base expression.
    """

    def __init__(
            self, *nargs,
            derived: Optional[Callable[..., PartialExpr]] = None, **kwargs):
        self.composition = derived  # some transformations may be non-primitive
        super().__init__(*nargs, **kwargs)
        assert self.type.is_function()

    def validate(self) -> None:
        """
        This method raises an error if the operation is a composite operation,
        but the declared type cannot be reconciled with the type inferred from
        the composition.
        """
        # If the operation is composite, check that its declared type is no
        # more general than the type we can infer from the composition function
        if self.composition:
            mock = Definition(_)
            args = [Base(mock) for a in signature(self.composition).parameters]
            output = self.composition(*args)
            if isinstance(output, Expr):
                type_infer = output.type
                type_decl = self.type.instance()
                vars_decl = list(type_decl.variables())
                for a in reversed(args):
                    type_infer = Function(a.type, type_infer)
                type_decl.unify(type_infer, subtype=True)
                type_decl = type_decl.resolve()

                # All the variables in the declared type must still be
                # variables --- otherwise we were too general
                if not all(isinstance(v.follow(), TypeVar) for v in vars_decl):
                    raise error.DeclaredTypeTooGeneral(self.type, type_infer)

            else:
                # The type could not be derived because the result is not a
                # full expression. Shouldn't happen?
                raise error.PartialPrimitive()


class PartialExpr(ABC):
    """
    An expression that may contain abstractions.
    """

    def __call__(self, *args: Union[PartialExpr, Definition]) -> PartialExpr:
        a = (e.instance() if isinstance(e, Definition) else e for e in args)
        return reduce(PartialExpr.partial_apply, a, self).complete()

    def partial_apply(self, x: PartialExpr) -> PartialExpr:
        f = self.complete()
        if isinstance(f, Abstraction):
            f.composition = partial(f.composition, x)
            return f.complete()
        elif isinstance(x, Abstraction):
            raise error.PartialPrimitive()
        else:
            assert isinstance(f, Expr) and isinstance(x, Expr)
            return Application(f, x)

    def complete(self) -> PartialExpr:
        """
        Once all parameters of an abstraction have been provided, turn into
        full expression.
        """
        if isinstance(self, Abstraction):
            n = len(signature(self.composition).parameters)
            if n == 0:
                return self.composition()
        return self


class Expr(PartialExpr):
    def __init__(self, type: TypeInstance):
        self.type = type

    def __repr__(self) -> str:
        return self.tree()

    def tree(self, lvl: str = "") -> str:
        """
        Obtain a tree representation using Unicode block characters.
        """
        if isinstance(self, Base):
            return f"╼ {self} : {self.definition.type}"
        else:
            assert isinstance(self, Application)
            return (
                f"{self.type.strc()}\n"
                f"{lvl} ├─{self.f.tree(lvl + ' │ ')}\n"
                f"{lvl} └─{self.x.tree(lvl + '   ')}"
            )

    def substitute(self, label: str, expr: Expr) -> Expr:
        """
        Replace the given expression for all expressions with the given label.
        """
        if isinstance(self, Base):
            if self.label == label:
                self.type.unify(expr.type)
                return expr
            else:
                return self
        else:
            assert isinstance(self, Application)
            self.f = self.f.substitute(label, expr)
            self.x = self.x.substitute(label, expr)
            return self

    def primitive(self) -> Expr:
        """
        Expand this expression into its simplest form.
        """
        f = self.partial_primitive()
        if isinstance(f, Abstraction):
            raise error.PartialPrimitive
        else:
            assert isinstance(f, Expr)
            return f

    def partial_primitive(self) -> PartialExpr:
        """
        Expand this expression into its simplest form. May contain
        abstractions.
        """
        if isinstance(self, Base):
            d = self.definition
            if isinstance(d, Operation) and d.composition:
                return Abstraction(d.composition).complete()
            else:
                return self
        else:
            assert isinstance(self, Application)
            f = self.f.partial_primitive()
            x = self.x.partial_primitive()
            return f.partial_apply(x)

    def renamed(self) -> Expr:
        """
        Give readable variable names to any variable left on the top-level of
        the expression. Differentiate them with a prime symbol (') to make sure
        there's no conflict with schematic variables in any definition.
        """
        # TODO this is turned off at the moment. find a better way to
        # differentiate between schematic and instance variables
        variables = list(self.type.variables(distinct=True))
        names = list("stuvwxyzabcde")

        if len(variables) > len(names):
            names = [f"t{i}" for i in range(len(variables))]

        for v, n in zip(variables, names):
            v.name = f"{n}'"
        return self


class Base(Expr):
    """
    A base expression represents either a single transformation or a data
    input. Base expressions may be unfolded into multiple applications of
    primitive transformations. Data input can be seen as a typed variable in an
    expression: in this case, it should be labelled and substituted.
    """

    def __init__(self, definition: Definition, label: Optional[str] = None):
        self.definition = definition
        self.label: Optional[str] = label
        super().__init__(type=definition.type.instance())

    def __str__(self) -> str:
        name = self.definition.name or "[?]"
        return f"{name} {self.label}" if self.label else name


class Application(Expr):
    """
    A comlex expression, representing an application of the transformation in
    its first argument to the expression in its second argument.
    """

    def __init__(self, f: Expr, x: Expr):
        try:
            result = f.type.apply(x.type)
        except error.TATypeError as e:
            e.while_applying(f, x)
            raise
        else:
            self.f: Expr = f
            self.x: Expr = x
            super().__init__(type=result)

    def __str__(self) -> str:
        return f"({self.f} {self.x})"


class Abstraction(PartialExpr):
    """
    An incomplete expression that needs to be supplied with arguments. Not
    normally part of an expression tree --- only used for expanding primitives.
    """

    def __init__(self, composition: Callable[..., PartialExpr]):
        self.composition: Callable[..., PartialExpr] = composition


class TransformationAlgebra(object):
    def __init__(self, *nargs: Definition, **kwargs: Definition):
        """
        Create a transformation algebra with pre-named (positional arguments)
        or to-be-named (keyword arguments) data and operation definitions.
        """
        self.definitions: Dict[str, Definition] = {}
        for v in nargs:
            assert v.name
            self[v.name] = v

        # If we get fed globals(), we will automatically filter out only
        # definitions, without complaining
        got_globals = '__builtins__' in kwargs

        for k, v in kwargs.items():
            if not got_globals or isinstance(v, Definition):
                assert not v.name
                v.name = k.rstrip("_") if got_globals else k
                self[v.name] = v

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return "\n".join(str(d) for d in self.definitions.values()) + "\n"

    def __getitem__(self, key: str) -> Definition:
        return self.definitions[key.lower()]

    def __setitem__(self, key: str, value: Definition) -> None:
        self.definitions[key.lower()] = value

        # Validation only happens once an operation is added to the algebra. If
        # we did it at define-time, it would lead to issues --- see issue #3
        if isinstance(value, Operation):
            value.validate()

    def parse(self, string: str) -> Optional[Expr]:
        # This used to be done via pyparsing, but the structure is so simple
        # that I opted to remove the dependency --- this is *much* faster
        stack: List[Optional[Expr]] = [None]

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
                if previous and isinstance(previous, Base) \
                        and isinstance(previous.definition, Data):
                    previous.label = token
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
            return stack[0]
        else:
            raise error.RBracketMismatch(string)


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
