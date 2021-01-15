"""
This module holds all the errors that might be raised. It should be exhaustive.
"""


class NonUniqueParents(RuntimeError):
    """
    This error occurs when a concept in a taxonomy has multiple subconcepts in
    such a way that it cannot be chalked up to transitivity.
    """
    pass


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
