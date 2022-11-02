"""
A generic transformation expression consists of operators that are applied to
source data or other operators. This module defines how they combine.
"""

from __future__ import annotations

from abc import ABC
from functools import reduce
from itertools import chain
from inspect import signature
from typing import Optional, Callable, Iterator

from transforge.label import Labels
from transforge.type import (Type, TypeVariable, TypeSchema,
    TypeInstance, Function, _, TypingError, TypeOperation)


class Operator(object):
    """
    An operator can be instantiated into an operation, which is an atomary step
    in a transformation. It may be primitive, or it may be defined in terms of
    other operations. In the latter case, it can be unfolded to reveal its
    inner structure.
    """

    def __init__(
            self,
            doc: Optional[str] = None,
            type: Type | Callable[..., TypeInstance] = _,
            body: Optional[Callable[..., Expr]] = None,
            name: Optional[str] = None):
        self._name = name
        self.description = doc
        self.type = type if isinstance(type, Type) else TypeSchema(type)
        self.body = body  # a transformation may be non-primitive

        t = self.type.instance()
        self.is_function = (isinstance(t, TypeOperation) and
            t.operator == Function)

        assert not self.description or isinstance(self.description, str)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return self._name or object.__repr__(self)

    @property
    def name(self) -> str:
        if self._name is None:
            raise RuntimeError("Unnamed operator.")
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        if value is None or (self._name is not None and value != self._name):
            raise RuntimeError(
                f"Cannot name operator {value}; already named {self._name}.")
        self._name = value

    def __call__(self, *args: Operator | Expr) -> Expr:
        """
        Calling an operation instantiates it as an expression.
        """
        return self.instance().__call__(*args)

    def instance(self) -> Expr:
        if self.is_function:
            return Operation(operator=self)
        else:
            return Source(type=self.type)

    def validate(self) -> None:
        """
        If this method is called and completes without error, we obtain some
        guarantees about the operator, namely:

        -   The definition typechecks.
        -   The declared type can be reconciled with the type inferred from the
            definition, if any.
        -   If the type of the operator contains type variables, they must be
            either wildcards or schematic variables. If they are not, that
            indicates that we are manipulating a "global" type variable, which
            is probably not what we want!
        -   If the type is constrained in any way, then the constrained
            variables must be drawn from the type.

        This validation step is not performed during initialization, but
        deferred instead to the point at which all operators in the primitive
        expansion are defined. This avoids `NameError`s.
        """
        try:

            # Check that all variables are schematic or wildcard
            t = self.type
            if (isinstance(t, TypeSchema) and not t.only_schematic()) or \
                    isinstance(t, TypeInstance) and any(t.variables()):
                raise NonSchematicVariablesError(self.type)

            # Check that constraints contain no free variables
            if isinstance(t, TypeSchema):
                t.validate_no_free_variables()

            # If the operation is composite, check that its declared type is no
            # more general than the type we can infer from the definition
            if self.body:
                type_decl = self.type.instance()
                vars_decl = list(type_decl.variables())
                type_infer = self.instance().primitive(unify=False).type

                type_decl.unify(type_infer, subtype=True)
                type_decl = type_decl.fix()

                # All the variables in the declared type must still be
                # variables --- otherwise we were too general
                if not all(isinstance(v.follow(), TypeVariable)
                        for v in vars_decl):
                    raise DeclaredTypeTooGeneralError(
                        self.type, self.instance().primitive().type)

        except (NonSchematicVariablesError, DeclaredTypeTooGeneralError,
                ApplicationError, TypingError) as e:
            raise DeclarationError(self) from e


###############################################################################

class Expr(ABC):
    def __init__(self, type: TypeInstance):
        self.type = type

    def __call__(self, *args: Expr | Operator) -> Expr:
        return reduce(Expr.apply, (Expr.shorthand(e) for e in args), self)

    @staticmethod
    def shorthand(value: Expr | Operator) -> Expr:
        if isinstance(value, Expr):
            return value
        else:
            assert isinstance(value, Operator)
            return value.instance()

    def __repr__(self) -> str:
        return self.tree()

    def __str__(self) -> str:
        return self.text(with_type=True, with_parentheses=False)

    def text(self,
            labels: dict[Variable, str] = Labels("x", subscript=True),
            with_type: bool = False,
            with_parentheses: bool = True):
        if isinstance(self, Operation):
            with_parentheses = False
            result = str(self.operator)
        elif isinstance(self, Source):
            with_type = False
            with_parentheses = False
            result = f"(- : {self.type.text(with_constraints=True)})"
        elif isinstance(self, Application):
            parens = not isinstance(self.f, Application)
            result = f"{self.f.text(labels, with_parentheses=parens)} "\
                f"{self.x.text(labels)}"
        elif isinstance(self, Abstraction):
            params = " ".join(p.text(labels) for p in self.params)
            result = f"λ{params}. {self.body.text(labels)}"
        else:
            assert isinstance(self, Variable)
            with_parentheses = False
            result = labels[self]
        if with_type:
            result = f"{result} : {self.type.text(with_constraints=True)}"
        if with_parentheses:
            result = f"({result})"
        return result

    def tree(self, lvl: str = "") -> str:
        """
        Obtain a tree representation using Unicode block characters.
        """
        if isinstance(self, Application):
            return (
                f"{self.type.text(with_constraints=True)}\n"
                f"{lvl} ├─{self.f.tree(lvl + ' │ ')}\n"
                f"{lvl} └─{self.x.tree(lvl + '   ')}"
            )
        elif isinstance(self, Abstraction):
            return (
                f"λ{' '.join(p.text() for p in self.params)}. ... : "
                f"{self.type.text(with_constraints=True)}\n"
                f"{lvl} └─{self.body.tree(lvl + '   ')}"
            )
        else:
            return f"╼ {self.text(with_type=True)}"

    def apply(self, arg: Expr) -> Expr:
        return Application(self, arg).normalize(recursive=False)

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

        if isinstance(expr, Operation):
            if expr.operator.body:
                expr_primitive = Abstraction(expr.operator.body)
                # The type of the original expression may be less general than
                # that of the primitive expression, but not more general.
                if unify:
                    expr.type.unify(expr_primitive.type, subtype=True)
                    expr_primitive.type = expr_primitive.type.fix()
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
        if isinstance(a, Source) and isinstance(b, Source):
            return bool(a.type.match(b.type))
        elif isinstance(a, Operation) and isinstance(b,
                Operation):
            return a.operator == b.operator
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
        if isinstance(a, (Source, Operation, Variable)):
            yield a
        elif isinstance(a, Abstraction):
            yield from a.params
            yield from a.body.leaves()
        else:
            assert isinstance(a, Application)
            yield from chain(a.f.leaves(), a.x.leaves())


class Operation(Expr):
    "An instance of an operator."

    def __init__(self, operator: Operator):
        self.operator = operator
        if isinstance(operator.type, TypeSchema):
            super().__init__(type=operator.type.instance(origin=self.operator))
        else:
            super().__init__(type=operator.type.instance())


class Source(Expr):
    "A source data input."

    def __init__(self, type: Type = _):
        super().__init__(type=type.instance())


class Application(Expr):
    "A complex expression, applying one expression to another."

    def __init__(self, f: Expr, x: Expr):
        self.f: Expr = f
        self.x: Expr = x

        try:
            t = f.type.apply(x.type)
        except TypingError as e:
            raise ApplicationError(f, x) from e

        super().__init__(type=t)


class Abstraction(Expr):
    """
    An incomplete expression that needs to be supplied with arguments. Not
    normally part of an expression tree --- except after expanding to
    primitives and then not fully applying the derived function.
    """

    def __init__(self, composition: Callable[..., Expr]):
        self.params: list[Variable] = [
            Variable() for _ in signature(composition).parameters]
        self.body: Expr = composition(*self.params)
        super().__init__(type=self.calculate_type())

    def calculate_type(self) -> TypeInstance:
        """
        (Re)calculate the type of an abstraction expression.
        """
        return reduce(lambda x, y: y.type ** x,
            reversed(self.params), self.body.type)


class Variable(Expr):
    """
    An expression variable. See `Abstraction`.
    """

    def __init__(self):
        self.bound: Optional[Expr] = None
        super().__init__(type=TypeVariable())

    def bind(self, expr: Expr) -> None:
        assert not self.bound, "cannot bind variable twice"
        self.bound = expr


# Errors #####################################################################

class DeclarationError(Exception):
    """
    Raised when an operator that has been declared is invalid for some reason.
    """

    def __init__(self, operator: Operator):
        self.operator = operator

    def __str__(self) -> str:
        assert self.__cause__, "must be caused by another error"
        return (
            f"While validating the declared operator '{self.operator}': "
            f"\t{self.__cause__}"
        )


class ApplicationError(Exception):
    """
    Raised when an operator could not be applied to an argument.
    """

    def __init__(self, operation: Expr, argument: Expr):
        self.operation = operation
        self.argument = argument

    def __str__(self) -> str:
        assert self.__cause__, "must be caused by another error"
        return (
            f"Could not apply `{self.operation}` to `{self.argument}`: "
            f"\t{self.__cause__}"
        )


class DeclaredTypeTooGeneralError(DeclarationError):
    """
    Raised when the declared type of a composite transformation is unifiable
    with the type inferred from its derivation, but it is too general.
    """

    def __init__(self, declared: Type, inferred: TypeInstance):
        self.declared = declared
        self.inferred = inferred

    def __str__(self) -> str:
        return \
            f"Declared type {self.declared} is more general than " \
            f"inferred type {self.inferred}."


class NonSchematicVariablesError(DeclarationError):
    """
    Raised when an operator contains a type instance variable rather than
    a schematic variable.
    """

    def __init__(self, type: Type):
        self.type = type

    def __str__(self) -> str:
        return f"Type {self.type} contains a non-schematic variable."
