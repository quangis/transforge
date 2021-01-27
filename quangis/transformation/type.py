"""
Generic type system. Inspired loosely by Hindley-Milner type inference in
functional programming languages.

Be warned: This module abuses overloading of Python's standard operators.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import chain
from typing import Dict, Optional, Tuple, Iterable


class AlgebraType(ABC):
    """
    Abstract base class for type operators and type variables. Note that basic
    types are just 0-ary type operators and functions are just particular 2-ary
    type operators.
    """

    #def __repr__(self):
    #    return self.__str__()

    def to_str(self) -> str:
        """
        To string including constraints on the top level
        """
        return "{} | {{{}}}".format(
            str(self),
            ", ".join(
                "{}: {}".format(var, tc)
                for var in set(self.variables())
                for tc in var.constraints
            )
        )

    #@abstractmethod
    #def equiv(self, other: AlgebraType) -> bool:
    #    return NotImplemented

    @abstractmethod
    def __contains__(self, value: AlgebraType) -> bool:
        return NotImplemented

    @abstractmethod
    def instantiate(self) -> AlgebraType:
        return NotImplemented

    def __pow__(self, other: AlgebraType) -> TypeOperator:
        """
        This is an overloaded (ab)use of Python's exponentiation operator. It
        allows us to use the infix operator ** for the arrow in function
        signatures.

        Note that this operator is one of the few that is right-to-left
        associative, matching the conventional behaviour of the function arrow.
        The right-bitshift operator >> (for __rshift__) would have been more
        intuitive visually, but does not have this property.
        """
        return TypeOperator('function', self, other)

    def __or__(self, constraints: Iterable[Constraint]) -> AlgebraType:
        """
        Abuse of Python's binary OR operator, for a pleasing notation of
        typeclass constraints.
        """

        # In principle, this function could return None, were it not for the
        # fact that we need a fresh version of self: since constraints will be
        # directly added to the variables, if we didn't do this, we'd need to
        # bend over backwards for generic variables
        ctx: Dict[TypeVar, TypeVar] = {}
        new = self.fresh(ctx)

        for constraint in constraints:
            new_constraint = constraint.fresh(ctx)
            for var in new_constraint.subject.variables():
                var.constraints.add(new_constraint)
        return new

    def fresh(self, ctx: Optional[Dict[TypeVar, TypeVar]] = None) -> AlgebraType:
        """
        Create a fresh copy of this type, with unique new variables.
        """

        ctx = {} if ctx is None else ctx
        if isinstance(self, TypeOperator):
            return TypeOperator(self.name, *(t.fresh(ctx) for t in self.types))
        elif isinstance(self, TypeVar):
            if self.bound:
                raise RuntimeError("cannot refresh bound variable")
            elif self in ctx:
                return ctx[self]
            else:
                new = TypeVar()
                for tc in self.constraints:
                    new.constraints.add(tc)
                ctx[self] = new
                return new
        raise RuntimeError("inexhaustive pattern")

    def unify(self: AlgebraType, other: AlgebraType) -> None:
        """
        Bind variables such that both types become the same. Note that subtypes
        on the "self" side are tolerated: that is, if self is a subtype of
        other, then they are considered the same, but not vice versa.
        """
        if isinstance(self, TypeOperator) and isinstance(other, TypeOperator):

            if self.arity == 0 and other.arity == 0 and self.subtype(other):
                pass
            elif self.signature != other.signature:
                raise RuntimeError("type mismatch")
            else:
                for x, y in zip(self.types, other.types):
                    x.unify(y)
        else:
            if isinstance(self, TypeVar):
                if self != other and self in other:
                    raise RuntimeError("recursive type")
                self.bind(other)
            elif isinstance(other, TypeVar):
                other.unify(self)

    def is_function(self) -> bool:
        if isinstance(self, TypeOperator):
            return self.name == 'function'
        return False

    def variables(self) -> Iterable[TypeVar]:
        """
        Obtain any variables left in the type expression.
        """
        if isinstance(self, TypeVar):
            yield self
        elif isinstance(self, TypeOperator):
            for v in chain(*(t.variables() for t in self.types)):
                yield v

    def __lshift__(self, other: Iterable[AlgebraType]) -> Constraint:
        """
        Abuse the left-shift operator to create a constraint like this:

            x, y = TypeVar(), TypeVar()
            Int, Str = TypeOperator("int"), TypeOperator("str")
            x << {Int ** y, Str ** y}
        """

        return Constraint(self, other)


class TypeOperator(AlgebraType):
    """
    n-ary type constructor.
    """

    def __init__(
            self,
            name: str,
            *types: AlgebraType,
            supertype: Optional[TypeOperator] = None):
        self.name = name
        self.types = list(types)
        self.supertype = supertype

        if self.name == 'function' and self.arity != 2:
            raise RuntimeError("functions must have 2 argument types")
        if self.types and self.supertype:
            raise RuntimeError("only nullary types may have supertypes")

    def subtype(self, other: TypeOperator) -> bool:
        """
        Is this type a subtype of another?
        """
        if self.arity == 0:
            up = self.supertype
            return self == other or bool(up and up.subtype(other))
        else:
            raise RuntimeError("direct subtypes can only be determined for nullary types")
            #return self.signature == other.signature and \
            #    all(s.subtype(t) for s, t in zip(self.types, other.types))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TypeOperator):
            return self.signature == other.signature and \
               all(s == t for s, t in zip(self.types, other.types))
        else:
            return False

    def __contains__(self, value: AlgebraType) -> bool:
        return value == self or any(value in t for t in self.types)

    def __str__(self) -> str:
        if self.is_function():
            return "({0} -> {1})".format(*self.types)
        elif self.types:
            return "{}({})".format(
                self.name, ", ".join(map(str, self.types))
            )
        else:
            return self.name

    def instantiate(self) -> AlgebraType:
        self.types = [t.instantiate() for t in self.types]
        return self

    def apply(self, arg: AlgebraType) -> AlgebraType:
        """
        Apply an argument to a function type to get its output type.
        """
        # TODO note that we cannot apply an argument to a type variable, even
        # though that sounds like it could be possible
        if self.is_function():
            input_type, output_type = self.types
            arg.instantiate().unify(input_type.instantiate())
            return output_type.instantiate()
        else:
            raise RuntimeError("Cannot apply an argument to non-function type")

    @property
    def arity(self) -> int:
        return len(self.types)

    @property
    def signature(self) -> Tuple[str, int]:
        return self.name, self.arity


class TypeVar(AlgebraType):

    counter = 0

    def __init__(self):
        cls = type(self)
        self.id = cls.counter
        self.bound = None
        self.constraints = set()
        cls.counter += 1

    def __str__(self) -> str:
        return "x" + str(self.id)

    def __contains__(self, value: AlgebraType) -> bool:
        return self == value

    def bind(self, binding: AlgebraType):
        assert (not self.bound or binding == self.bound), \
            "binding variable multiple times"
        self.bound = binding

        for constraint in self.constraints:
            constraint.enforce()

    def instantiate(self) -> AlgebraType:
        #for c in self.constraints:
        #    tc.instantiate()

        if self.bound:
            return self.bound
        else:
            return self


class Constraint(object):
    """
    A constraint is a ...
    To avoid recursive types, the typeclass may not contain any variable from
    the type.
    """
    # The way it works is: we assign the constraint to all variables occurring
    # in said constraint. We don't do this immediately, but only after the
    # user has refreshed the variables.

    def __init__(self, t: AlgebraType, typeclass: Iterable[AlgebraType]):
        self.subject = t
        self.typeclass = list(typeclass)

        # TODO check that t has at least one variable and that none of them
        # occur in the typeclasses

    def fresh(self, ctx: Dict[TypeVar, TypeVar]) -> Constraint:
        return Constraint(
            self.subject.fresh(ctx), (t.fresh(ctx) for t in self.typeclass)
        )

    def __str__(self) -> str:
        return "{} << [{}]".format(
            self.subject,
            ", ".join(map(str, self.typeclass))
        )

    def enforce(self):

        # Check that at least one of the typeclasses matches
        subject = self.subject.instantiate()
        if not any(
                same_structure(t.instantiate(), subject)
                for t in self.typeclass
                ):
            raise RuntimeError("Violated typeclass constraint")


def same_structure(self: AlgebraType, other: AlgebraType) -> bool:
    """
    Test if a type is structurally equivalent to another, that is, if it is the
    same if variables are disregarded.
    """
    if isinstance(self, TypeOperator) and isinstance(other, TypeOperator):
        return self.signature == other.signature and \
            all(same_structure(s, t) for s, t in zip(self.types, other.types))
    return True

