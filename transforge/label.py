"""
Provides the `Labels` class for robust human-readable variable identifiers.
"""

from itertools import count
from collections import defaultdict


class Labels(defaultdict):
    """
    Provides a mapping from hashable objects to unique string identifiers.
    Anything it accesses gets assigned a fresh, human-readable name. It's one
    of the few mutable objects for which it makes sense to set it as a default
    argument: it will then assign 'global' identifiers.
    """

    def __init__(self, prefix: str, subscript: bool = False, start: int = 1):
        self.iterator = (f"{prefix}{i}" for i in
            map(subscripted if subscript else int.__str__, count(start))
        )
        super().__init__(lambda: next(self.iterator))


def subscripted(i: int) -> str:
    """
    Convert a number into a subscripted string.
    """
    return "".join(chr(ord("₀") - ord("0") + ord(d)) for d in str(i))
