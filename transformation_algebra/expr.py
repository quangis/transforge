"""
Classes to define generic transformation algebras.
"""

from __future__ import annotations

from enum import Enum, auto
from abc import ABC
from functools import reduce
from itertools import groupby, chain
from inspect import signature, Signature, Parameter
from typing import Optional, Dict, Callable, Union, List, Iterator, Set

from transformation_algebra import error, flow
from transformation_algebra.type import \
    Type, TypeVar, TypeSchema, TypeOperator, TypeInstance, Function


class Definition(ABC):
    """
    A definition represents a non-instantiated data input or transformation.
    """

    def __init__(
            self,
            type: Union[Type, Callable[..., TypeInstance]],
            doc: Optional[str] = None,
            name: Optional[str] = None):
        self.name = name
        self.type = type if isinstance(type, Type) else TypeSchema(type)
        self.description = doc

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"{self.name or '[?]'} : {self.type}"

    def __call__(self, *args: Union[Definition, Expr]) -> Expr:
        """
        Calling a definition instantiates it as an expression.
        """
        return self.instance().__call__(*args)

    def instance(self, identifier: Optional[str] = None) -> Expr:
        return Base(self, label=identifier)

    def primitive(self) -> bool:
        return isinstance(self, Operation) and not self.composition


class Data(Definition):
    """
    The definition of a data input. An instance of such a definition is a base
    expression.
    """

    def __init__(self, *nargs, **kwargs):
        self.composition = None
        super().__init__(*nargs, **kwargs)
        assert self.type.instance().operator != Function


class Operation(Definition, flow.Unit):
    """
    The definition of a transformation. An instance of such a definition is a
    base expression.
    """

    def __init__(
            self, *nargs,
            derived: Optional[Callable[..., Expr]] = None, **kwargs):
        self.composition = derived  # a transformation may be non-primitive
        super().__init__(*nargs, **kwargs)
        assert self.type.instance().operator == Function

    def validate(self) -> None:
        """
        This method raises an error if the operation is a composite operation,
        but the declared type cannot be reconciled with the type inferred from
        the composition.
        """
        # If the operation is composite, check that its declared type is no
        # more general than the type we can infer from the composition function
        try:
            if self.composition:
                type_decl = self.type.instance()
                vars_decl = list(type_decl.variables())
                type_infer = self.instance().primitive().type

                type_decl.unify(type_infer, subtype=True)
                type_decl = type_decl.resolve()

                # All the variables in the declared type must still be
                # variables --- otherwise we were too general
                if not all(isinstance(v.follow(), TypeVar) for v in vars_decl):
                    raise error.DeclaredTypeTooGeneral(
                        self.type, self.instance().primitive().type)

        except error.TAError as e:
            e.definition = self
            raise


###############################################################################

class Expr(ABC):
    def __init__(self, type: TypeInstance):
        self.type = type

    def __repr__(self) -> str:
        return self.tree()

    def __call__(self, *args: Union[Expr, Definition]) -> Expr:
        return reduce(Expr.apply,
            (e if isinstance(e, Expr) else e.instance() for e in args),
            self)

    def tree(self, lvl: str = "") -> str:
        """
        Obtain a tree representation using Unicode block characters.
        """
        if isinstance(self, Application):
            return (
                f"{self.type.str_with_constraints()}\n"
                f"{lvl} ├─{self.f.tree(lvl + ' │ ')}\n"
                f"{lvl} └─{self.x.tree(lvl + '   ')}"
            )
        elif isinstance(self, Abstraction):
            return (
                f"λ{' '.join(str(p) for p in self.params)}. ... : "
                f"{self.type.str_with_constraints()}\n"
                f"{lvl} └─{self.body.tree(lvl + '   ')}"
            )
        else:
            return f"╼ {self} : {self.type.str_with_constraints()}"

    def apply(self, arg: Expr) -> Expr:
        try:
            if isinstance(self, Abstraction):
                param = self.params.pop(0)
                param.bind(arg)
                return self if self.params else self.body
            else:
                return Application(self, arg)
        except error.TATypeError as e:
            e.while_applying(self, arg)
            raise

    def normalize(self, recurse: bool = False) -> Expr:
        """
        Follow bound variables to their bindings, and replace nested
        abstractions λx.λy.… with a single abstraction λx y.….
        """
        if isinstance(self, Variable) and self.bound:
            return self.bound.normalize(recurse)
        elif isinstance(self, Abstraction):
            if not self.params:
                return self.body.normalize(recurse)
            elif isinstance(self.body, Abstraction):
                self.params += self.body.params
                self.body = self.body.body.normalize(recurse)
            elif recurse:
                self.body = self.body.normalize(True)
        elif isinstance(self, Application) and recurse:
            self.f = self.f.normalize(True)
            self.x = self.x.normalize(True)
        return self

    def primitive(self, rename: bool = True) -> Expr:
        """
        Expand this expression into its simplest form.
        """
        result = self
        if isinstance(self, Base):
            d = self.definition
            if isinstance(d, Operation) and d.composition:
                result = Abstraction(d.composition).primitive(False)
        elif isinstance(self, Application):
            result = self.f.primitive(False).apply(self.x.primitive(False))
        elif isinstance(self, Abstraction):
            self.body = self.body.primitive(False)
        if rename:
            result.rename()
        return result.normalize(False)

    def replace(self, label: str, new: Expr) -> Expr:
        """
        Substitute all labelled expressions with the new expression.
        """
        if isinstance(self, Base) and self.label == label:
            new.type.unify(self.type, subtype=True)
            return new
        elif isinstance(self, Abstraction):
            self.body = self.body.replace(label, new)
        elif isinstance(self, Application):
            self.f = self.f.replace(label, new)
            self.x = self.x.replace(label, new)
        return self

    def leaves(self) -> Iterator[Expr]:
        """
        Obtain leaf expressions.
        """
        a = self.normalize(recurse=False)
        if isinstance(a, (Base, Variable)):
            yield a
        elif isinstance(a, Abstraction):
            yield from a.body.leaves()
        else:
            assert isinstance(self, Application)
            yield from chain(a.f.leaves(), a.x.leaves())

    def labels(self) -> List[str]:
        """
        Obtain all labels in this expression in alphabetical order.
        """
        labels: Set[str] = set()
        for expr in self.leaves():
            if isinstance(expr, Base) and expr.label:
                labels.add(expr.label)
        return sorted(labels)

    def supply(self, *nargs: Expr, **kwargs: Expr) -> Expr:
        """
        Treat labelled inputs as parameters and supply them. Note that
        arguments are in alphabetical order.
        """
        sig = Signature([Parameter(v, Parameter.POSITIONAL_OR_KEYWORD)
            for v in self.labels()])
        binding = sig.bind(*nargs, **kwargs)
        result = self
        for param, value in binding.arguments.items():
            result = result.replace(param, value)
        return result

    def rename(self) -> None:
        """
        Give readable variable names to any expression variable and type
        variable in the expression.
        """
        expr_vars: Set[Variable] = set()
        type_vars: Set[TypeVar] = set()

        for expr in self.leaves():
            type_vars.update(expr.type.variables())
            if isinstance(expr, Variable):
                expr_vars.add(expr)

        for i, expr in enumerate(expr_vars):
            expr.name = f"x{subscript(i)}"

        for i, var in enumerate(type_vars):
            var.name = f"τ{subscript(i)}"


class Base(Expr):
    """
    A base expression represents either a single transformation or a data
    input. Base expressions may be unfolded into multiple applications of
    primitive transformations.
    """

    def __init__(self, definition: Definition, label: Optional[str] = None):
        self.definition = definition
        self.label: Optional[str] = label
        super().__init__(type=definition.type.instance())

    def __str__(self) -> str:
        name = self.definition.name or "[?]"
        return f"({name} {self.label})" if self.label else name


class Application(Expr):
    """
    A complex expression, representing an application of the transformation in
    its first argument to the expression in its second argument.
    """

    def __init__(self, f: Expr, x: Expr):
        self.f: Expr = f
        self.x: Expr = x
        super().__init__(type=f.type.apply(x.type))

    def __str__(self) -> str:
        return f"({self.f} {self.x})"


class Abstraction(Expr):
    """
    An incomplete expression that needs to be supplied with arguments. Not
    normally part of an expression tree --- except after expanding to
    primitives and then not fully applying the derived function.
    """

    def __init__(self, composition: Callable[..., Expr]):
        self.params: List[Variable] = [
            Variable() for _ in signature(composition).parameters]
        self.body: Expr = composition(*self.params)
        self.type = reduce(lambda t, p: p.type ** t,
            reversed(self.params), self.body.type)

    def __str__(self) -> str:
        return f"λ{' '.join(str(p) for p in self.params)}. {self.body}"


class Variable(Expr):
    """
    An expression variable. See `Abstraction`.
    """

    def __init__(self, name: Optional[str] = None):
        self._name = name
        self.bound: Optional[Expr] = None
        super().__init__(type=TypeVar())

    @property
    def name(self) -> str:
        return self._name or f"var{hash(self)}"

    @name.setter
    def name(self, value: str) -> None:
        assert not self.bound
        self._name = value

    def __str__(self) -> str:
        return self.name

    def bind(self, expr: Expr) -> None:
        a = self.normalize(False)
        b = expr.normalize(False)
        assert not a.bound
        a.bound = b
        a.type.unify(b.type)


###############################################################################

class TransformationAlgebra(object):
    def __init__(self):
        """
        Initiate an empty transformation algebra.
        """
        self.definitions: Dict[str, Definition] = {}
        self.types: Set[TypeOperator] = set()

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return "\n".join(str(d) for d in self.definitions.values()) + "\n"

    def __contains__(self, key: Union[str, Definition, TypeOperator]) -> bool:
        if isinstance(key, TypeOperator):
            return key in self.types
        elif isinstance(key, Definition):
            assert key.name
            return self.definitions.get(key.name.lower()) is key
        else:
            assert isinstance(key, str)
            return key.lower() in self.definitions

    def __getitem__(self, key: str) -> Definition:
        return self.definitions[key.lower()]

    def __setitem__(self, key: str, value: Definition) -> None:
        self.definitions[key.lower()] = value

        # Validation only happens once an operation is added to the algebra. If
        # we did it at define-time, it would lead to issues --- see issue #3
        if isinstance(value, Operation):
            value.validate()

        for op in value.type.instance().operators():
            self.types.add(op)

    def add(self, *nargs: Definition, **kwargs: Definition) -> None:
        """
        Add data and operation definitions to this transformation algebra. Note
        that names ending with an underscore will be stripped of that symbol.
        """

        for k, v in chain(kwargs.items(),
                ((v.name, v) for v in nargs)):
            assert k is not None
            if isinstance(v, Definition):
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

        labels: Dict[str, TypeInstance] = dict()
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


def subscript(i: int) -> str:
    """
    Convert a number into a subscripted string.
    """
    return "".join(chr(ord("₀") - ord("0") + ord(d)) for d in str(i))
