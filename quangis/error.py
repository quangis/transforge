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
    pass


class RecursiveType(AlgebraTypeError):
    def __init__(self, t1, t2):
        self.t1 = t1
        self.t2 = t2

    def __str__(self) -> str:
        return f"Recursive type: {self.t1} and {self.t2}"


class TypeMismatch(AlgebraTypeError):
    def __init__(self, t1, t2):
        self.t1 = t1
        self.t2 = t2

    def __str__(self) -> str:
        return f"Type mismatch: {self.t1} and {self.t2}"


class ViolatedConstraint(AlgebraTypeError):
    def __init__(self, c):
        self.c = c

    def __str__(self) -> str:
        return f"Violated type constraint: {self.c}"


class NonFunctionApplication(AlgebraTypeError):
    def __init__(self, fn, arg):
        self.fn = fn
        self.arg = arg

    def __str__(self) -> str:
        return f"Cannot apply {self.arg} to non-function {self.fn}"


class AlreadyBound(AlgebraTypeError):
    def __init__(self, var):
        self.var = var

    def __str__(self) -> str:
        return f"Variable {self.var} was already bound"
