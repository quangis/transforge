from __future__ import annotations

from transforge.type import Type
from typing import Iterator, Iterable
from collections.abc import MutableSet

class TypeUnion(MutableSet[Type]):

    def __init__(self, xs: Iterable[Type] = ()) -> None:
        self.data: set[Type] = set()
        for x in xs:
            self.add(x)

    def __contains__(self, value: object) -> bool:
        return value in self.data

    def __iter__(self) -> Iterator[Type]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return repr(self.data)

    def is_subtype(self, other: Type | "TypeUnion") -> bool:
        if isinstance(other, Type):
            return bool(self) and all(x.is_subtype(other) for x in self.data)
        else:
            assert isinstance(other, TypeUnion)
            return bool(self) and all(all(x.is_subtype(y) for x in self.data) 
                for y in other.data)

    def add(self, new: Type) -> None:
        to_remove = set()
        for t in self.data:
            if new.is_subtype(t):
                to_remove.add(t)
            elif t.is_subtype(new):
                return
        self.data -= to_remove
        self.data.add(new)

    def discard(self, item: Type) -> None:
        self.data.discard(item)


# TODO: sort on type depth, so that the most specific types are checked first
class Bag(object):
    """A bag of types is a conjunction of disjunctions (cq intersection of 
    unions) of concrete types. A bag is unordered and contains no duplicates, 
    nor even supertypes of types that are already in the bag. That is, if you 
    have a type A with subtypes B and C, then adding A, and either B or C will 
    result in a bag containing only either B or C."""

    def __init__(self) -> None:
        self.content: list[TypeUnion] = []

    def add(self, *new_types: Type):
        """Add a type to the bag. If multiple types are given, the union of 
        these types is added."""

        # Any disjuncts that are covered already by other types in the bag can 
        # be dropped
        new = TypeUnion(nt for nt in new_types
            if not any(t.is_subtype(nt) for t in self.content))

        if not new:
            return

        # Conversely, any types in the bag that are obsoleted by the new type 
        # can also be removed
        self.content = [c for c in self.content
            if not new.is_subtype(c)]

        self.content.append(new)
