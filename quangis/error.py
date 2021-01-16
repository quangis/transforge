"""
This module holds all the errors that might be raised. It should be exhaustive.
"""

from typing import Any


class NonUniqueParents(RuntimeError):
    """
    This error occurs when a concept is subsumed by multiple superconcepts,
    neither of which is subsumed by the other. This cannot be chalked up to
    transitivity, so a tree cannot be constructed.
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
