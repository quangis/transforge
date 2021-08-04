"""
A generic transformation algebra consists of *definitions* (of operations and
of data constructors). Those definitions can be instantiated and combined into
transformation algebra *expressions*, which describe applications of operations
to data.
"""

from __future__ import annotations

from enum import Enum, auto
from abc import ABC
from functools import reduce
from itertools import groupby, chain
from inspect import signature
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
                type_infer = self.instance().primitive(unify=False).type

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
            return Application(self, arg).normalize(recursive=False)
        except error.TATypeError as e:
            e.while_applying(self, arg)
            raise

    def normalize(self, recursive: bool = True) -> Expr:
        """
        -   Follow bound variables to their bindings.
        -   Apply arguments to abstractions where possible.
        -   Replace nested abstractions (λx.λy. …) with a single (λx y. …).
        -   Collapse abstractions without parameters.

        Set recursive to False to only guarantee that the top level expression
        is in normal form.
        """

        if isinstance(self, Variable) and self.bound:
            return self.bound.normalize(recursive)

        elif isinstance(self, Abstraction):
            if recursive:
                self.body = self.body.normalize(recursive)

            if not self.params:
                return self.body.normalize(recursive)
            elif isinstance(self.body, Abstraction):
                self.params += self.body.params
                self.body = self.body.body
                self.type = self.calculate_type()
                return self.normalize(recursive)

        elif isinstance(self, Application):
            if recursive:
                self.f = self.f.normalize(recursive)
                self.x = self.x.normalize(recursive)

            if isinstance(self.f, Abstraction):
                if self.f.params:
                    self.f.params.pop(0).bind(self.x)
                    self.f.type = self.f.calculate_type()
                    return self.f.normalize(recursive)
                else:
                    assert not recursive
                    return self.f.normalize(recursive)

        return self

    def primitive(self, normalize: bool = True, unify: bool = True) -> Expr:
        """
        Expand this expression into its simplest form.
        """
        expr = self.normalize(recursive=False)

        if isinstance(expr, Base):
            d = expr.definition
            if isinstance(d, Operation) and d.composition:
                expr_primitive = Abstraction(d.composition)
                # The type of the original expression may be less general than
                # that of the primitive expression, but not more general.
                if unify:
                    expr.type.unify(expr_primitive.type, subtype=True)
                    expr_primitive.type = expr_primitive.type.resolve()
                expr = expr_primitive.primitive(normalize=False)

        elif isinstance(expr, Application):
            expr.f = expr.f.primitive(normalize=False)
            expr.x = expr.x.primitive(normalize=False)

        elif isinstance(expr, Abstraction):
            expr.body = expr.body.primitive(normalize=False)

        return expr.normalize(recursive=normalize)

    def match(self, other: Expr) -> bool:
        """
        Check that the normalized expressions are the same.
        """
        a = self.normalize(recursive=False)
        b = other.normalize(recursive=False)
        if isinstance(a, Base) and isinstance(b, Base):
            return a.definition == b.definition and a.label == b.label
        elif isinstance(a, Application) and isinstance(b, Application):
            return a.f.match(b.f) and a.x.match(b.x)
        elif isinstance(a, Abstraction) and isinstance(b, Abstraction):
            return all(x.match(y) for x, y in zip(a.params, b.params)) and \
                a.body.match(b.body)
        return a == b

    def leaves(self) -> Iterator[Expr]:
        """
        Obtain all base expressions and variables in an expression.
        """
        a = self.normalize(recursive=False)
        if isinstance(a, (Base, Variable)):
            yield a
        elif isinstance(a, Abstraction):
            yield from a.params
            yield from a.body.leaves()
        else:
            assert isinstance(a, Application)
            yield from chain(a.f.leaves(), a.x.leaves())

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
        super().__init__(type=self.calculate_type())

    def calculate_type(self) -> TypeInstance:
        """
        (Re)calculate the type of an abstraction expression.
        """
        return reduce(lambda x, y: y.type ** x,
            reversed(self.params), self.body.type)

    def __str__(self) -> str:
        return f"(λ{' '.join(str(p) for p in self.params)}. {self.body})"


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
        assert not self.bound, "cannot bind variable twice"
        self.bound = expr


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
