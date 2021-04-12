"""
This module holds all the errors that might be raised.
"""

from abc import abstractmethod
from typing import Optional

import transformation_algebra as ta


class TAError(RuntimeError):
    """
    Any error raised by this library.
    """
    pass


# Parsing errors #############################################################

class TAParseError(TAError):
    pass


class BracketMismatch(TAParseError):
    def __str__(self) -> str:
        return "Mismatched bracket."


class LBracketMismatch(BracketMismatch):
    pass


class RBracketMismatch(BracketMismatch):
    pass


class Undefined(TAParseError):
    def __init__(self, token: str):
        self.token = token

    def __str__(self) -> str:
        return f"Transformation or data input '{self.token}' is undefined."


# Type errors ################################################################

class TATypeError(TAError):
    """
    This error occurs when an expression does not typecheck.
    """

    def __init__(self, t1: 'ta.type.Type', t2: 'ta.type.Type'):
        self.t1 = t1
        self.t2 = t2
        self.fn = None
        self.arg = None

    def while_applying(self, fn: 'ta.expr.Expr', arg: 'ta.expr.Expr'):
        self.fn = fn
        self.arg = arg

    @abstractmethod
    def specify(self) -> str:
        return NotImplemented

    def __str__(self) -> str:
        if self.fn and self.arg:
            return (
                f"During the application of the following expressions:\n"
                f"\t\033[1m{self.fn}\033[0m to\n"
                f"\t\033[1m{self.arg}\033[0m\n"
                f"{self.specify()}"
            )
        else:
            return f"A type error occurred: {self.specify()}"


class RecursiveType(TATypeError):
    """
    Raised for infinite types.
    """

    def specify(self) -> str:
        return f"Encountered the recursive type {self.t1}~{self.t2}."


class TypeMismatch(TATypeError):
    """
    Raised when compound types mismatch.
    """

    def specify(self) -> str:
        return "Could not unify type {self.t1} with {self.t2}."


class SubtypeMismatch(TypeMismatch):
    """
    Raised when base types are not subtypes.
    """

    def specify(self) -> str:
        return "Could not satisfy subtype {self.t1} <= {self.t2}"


class FunctionApplicationError(TATypeError):
    """
    Raised when an argument is passed to a non-function type.
    """

    def specify(self) -> str:
        return f"Could not apply non-function {self.t1} to {self.t2}."


# Constraint errors ##########################################################

class TAConstraintError(TAError):
    """
    Raised when there is an issue with a typeclass constraint.
    """

    def __init__(self, constraint: 'ta.type.Constraint'):
        self.constraint = constraint


class ConstraintViolation(TAConstraintError):
    """
    Raised when there can be no situation in which a constraint is satisfied.
    """

    def __str__(self) -> str:
        return f"Violated type constraint:\n\t{self.constraint.description}"


class ConstrainFreeVariable(TAConstraintError):
    """
    Raised when a constraint refers to a variable that does not occur in the
    context that it is constraining.
    """

    def __str__(self) -> str:
        return (
            f"Free variable occurs in constraint:\n"
            f"\t{self.constraint.description}")


# Definition errors ##########################################################

class TADefinitionError(TAError):
    """
    An error that occurs in the definition of an operation or data input.
    """
    pass

    def __init__(
            self,
            definition: 'ta.expr.Definition',
            e: 'Optional[TAError]' = None):
        self.definition = definition
        self.e = e


class TypeAnnotationError(TADefinitionError):
    """
    Raised when the declared type of a composite transformation is not
    unifiable with the type inferred from its derivation, or when the declared
    type is more general than the inferred type.
    """

    def __init__(
            self,
            declared: 'ta.type.Type',
            inferred: 'ta.type.Type',
            *nargs, **kwargs):
        self.declared = declared
        self.inferred = inferred
        super().__init__(*nargs, **kwargs)

    def __str__(self) -> str:
        return (
            f"Declared type {self.declared} could not be reconciled with "
            f"inferred type {self.inferred}. {self.e if self.e else ''}"
        )


class PartialPrimitiveError(TAError):
    """
    A composite expression must be fully applied before for its primitive to be
    derivable. Otherwise, an expression tree would contain abstractions. This
    error is raised when the primitive of a partially applied composite
    expression is taken.
    """

    def __init__(self):
        pass

    def __str__(self) -> str:
        return (
            f"Cannot express partially applied composite "
            f"expression as primitive."
        )
