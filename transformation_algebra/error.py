"""
This module holds all the errors that might be raised.
"""

from abc import abstractmethod

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
        clause = f" while applying {self.fn} to {self.arg}" \
            if self.fn and self.arg else ""
        return f"A type error occurred{clause}: {self.specify()}"


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
        return f"Could not unify type {self.t1} with {self.t2}."


class SubtypeMismatch(TypeMismatch):
    """
    Raised when base types are not subtypes.
    """

    def specify(self) -> str:
        return f"Could not satisfy subtype {self.t1} <= {self.t2}."


class FunctionApplicationError(TATypeError):
    """
    Raised when an argument is passed to a non-function type.
    """

    def specify(self) -> str:
        return f"Could not apply non-function {self.t1} to {self.t2}."


class DeclaredTypeTooGeneral(TATypeError):
    """
    Raised when the declared type of a composite transformation is unifiable
    with the type inferred from its derivation, but it is too general.
    """

    def specify(self) -> str:
        return (
            f"Declared type {self.t1} is more general than "
            f"inferred type {self.t2}."
        )


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
        return f"Violated typeclass constraint {self.constraint.description}."


class ConstrainFreeVariable(TAConstraintError):
    """
    Raised when a constraint refers to a variable that does not occur in the
    context that it is constraining.
    """

    def __str__(self) -> str:
        return (
            f"A free variable occurs in constraint "
            f"{self.constraint.description}")


# Other errors ###############################################################

class PartialPrimitive(TAError):
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
            "Cannot express partially applied composite "
            "expression as a primitive expression."
        )
