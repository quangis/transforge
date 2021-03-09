"""
This module holds all the errors that might be raised.
"""


class AlgebraTypeError(RuntimeError):
    """
    This error occurs when an expression does not typecheck.
    """

    def add_expression(self, fn, arg):
        self.fn = fn
        self.arg = arg

    def __str__(self) -> str:
        if self.fn and self.arg:
            return (
                f"Error while applying:\n"
                f"\t\033[1m{self.arg}\033[0m\n\tto\n"
                f"\t\033[1m{self.fn}\033[0m\n"
            )
        else:
            return "Typing error.\n"


class RecursiveType(AlgebraTypeError):
    def __init__(self, t1, t2):
        self.t1 = t1
        self.t2 = t2

    def __str__(self) -> str:
        return (
            super().__str__() +
            f"Recursive type: {self.t1} and {self.t2}"
        )


class TypeMismatch(AlgebraTypeError):
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


class SubtypeMismatch(AlgebraTypeError):
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


class ViolatedConstraint(AlgebraTypeError):
    def __init__(self, c):
        self.c = c

    def __str__(self) -> str:
        return (
            super().__str__() +
            f"Violated type constraint:\n\t{self.c}"
        )


class NonFunctionApplication(AlgebraTypeError):
    def __init__(self, fn, arg):
        self.fn = fn
        self.arg = arg

    def __str__(self) -> str:
        return (
            super().__str__() +
            f"Cannot apply {self.arg} to non-function {self.fn}"
        )
