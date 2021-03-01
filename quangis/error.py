"""
This module holds all the errors that might be raised. It should be exhaustive.
"""

from typing import Any


class NonUniqueParents(RuntimeError):
    """
    This error occurs when a concept is in a taxonomy is subsumed by multiple
    superconcepts, neither of which is subsumed by the other. This cannot be
    chalked up to transitivity, so a true taxonomical tree cannot be
    constructed.
    """

    def __init__(self, child: Any, old: Any, new: Any):
        self.old = old
        self.new = new
        self.child = child


class Cycle(RuntimeError):
    """
    This error occurs when a cyclic graph is defined that should not be cyclic.
    """
    pass


class DisconnectedTree(RuntimeError):
    """
    This error occurs when some node in a tree floats unconnected to the root.
    """
    pass


class Key(KeyError):
    pass


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
            f"\t\033[1m{self.t1}\033[0m with \033[1m{self.t2}\033[0m"
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
            f"\t\033[1m{self.c1}\033[0m <= \033[1m{self.c2}\033[0m"
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
