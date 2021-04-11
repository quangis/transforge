"""
This module holds all the errors that might be raised.
"""

import transformation_algebra as ta


class TAError(RuntimeError):
    """
    Any error raised by this library.
    """
    pass


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


class TATypeError(TAError):
    """
    This error occurs when an expression does not typecheck.
    """

    def add_expression(self, fn, arg):
        self.fn = fn
        self.arg = arg

    def add_definition(self, definition):
        self.definition = definition

    def __str__(self) -> str:
        result = []
        if self.fn and self.arg:
            result.extend([
                "Error while applying:",
                f"\t\033[1m{self.arg}\033[0m\n\tto",
                f"\t\033[1m{self.fn}\033[0m",
            ])
        if self.definition:
            result.append(f"in the definition of {self.definition.name}")
        return "\n".join(result or "Typing error.") + "\n"


class DefinitionTypeMismatch(TATypeError):
    def __init__(self, definition, declared, inferred):
        self.declared = declared
        self.inferred = inferred
        self.definition = definition

    def __str__(self) -> str:
        return (
            f"Declared type {self.declared} cannot be reconciled with "
            f"inferred type {self.inferred} "
            f"in {self.definition.name or 'an anonymous operation'}"
        )


class RecursiveType(TATypeError):
    def __init__(self, t1, t2):
        self.t1 = t1
        self.t2 = t2

    def __str__(self) -> str:
        return (
            super().__str__() +
            f"Recursive type: {self.t1} and {self.t2}"
        )


class TypeMismatch(TATypeError):
    def __init__(self, t1, t2):
        self.t1 = t1
        self.t2 = t2
        self.fn = None
        self.arg = None

    def __str__(self) -> str:
        return (
            super().__str__() +
            "Type mismatch. Could not unify:\n"
            f"\t{self.t1} with {self.t2}"
        )


class SubtypeMismatch(TATypeError):
    def __init__(self, c1, c2):
        self.c1 = c1
        self.c2 = c2
        self.fn = None
        self.arg = None

    def __str__(self) -> str:
        return (
            super().__str__() +
            "Subtype mismatch. Could not satisfy:\n"
            f"\t{self.c1} <= {self.c2}"
        )


class ConstraintViolation(TATypeError):
    def __init__(self, constraint: 'ta.type.Constraint'):
        self.constraint = constraint

    def __str__(self) -> str:
        return (
            super().__str__() +
            f"Violated type constraint:\n\t{self.constraint.description}"
        )


class ConstrainFreeVariable(TATypeError):
    pass


class NonFunctionApplication(TATypeError):
    def __init__(self, fn, arg):
        self.fn = fn
        self.arg = arg

    def __str__(self) -> str:
        return (
            super().__str__() +
            f"Cannot apply {self.arg} to non-function {self.fn}"
        )
