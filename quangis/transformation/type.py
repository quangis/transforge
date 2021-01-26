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

    def __repr__(self):
        return self.__str__()

    def to_str(self) -> str:
        """
        To string including constraints on the top level
        """
        return "{} | {{{}}}".format(
            str(self),
            ", ".join(
                "{}: {}".format(var, tc)
                for var in set(self.variables())
                for tc in var.typeclasses
            )
        )

    @abstractmethod
    def __contains__(self, value: AlgebraType) -> bool:
        return NotImplemented

    @abstractmethod
    def _fresh(self, ctx: Dict[TypeVar, TypeVar]) -> AlgebraType:
        return NotImplemented

    @abstractmethod
    def constrain(self, var: TypeVar, typeclass: TypeClass):
        return NotImplemented

    def fresh(self) -> AlgebraType:
        """
        Create a fresh copy of this type, with unique new variables.
        """

        return self._fresh({})

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

    def __or__(self, constraints: Dict[TypeVar, Typeclass]) -> AlgebraType:
        """
        Abuse of Python's binary OR operator, for a pleasing notation of
        typeclass constraints.
        """

        # Needs a new context because constraints will be directly added to the
        # variables
        ctx: Dict[TypeVar, TypeVar] = {}
        new = self._fresh(ctx)
        for old_variable, old_typeclass in constraints.items():
            new_variable = ctx[old_variable]
            new_typeclass = old_typeclass._fresh(ctx)
            new.constrain(new_variable, new_typeclass)
        return new

    def substitute(self, subst: Dict[TypeVar, AlgebraType]) -> AlgebraType:
        if isinstance(self, TypeOperator):
            self.types = [t.substitute(subst) for t in self.types]
        elif isinstance(self, TypeVar):
            if self in subst:
                return subst[self]
        else:
            raise RuntimeError("non-exhaustive pattern")
        return self

    def unify(
            self: AlgebraType,
            other: AlgebraType,
            ctx: Dict[TypeVar, AlgebraType]) -> Dict[TypeVar, AlgebraType]:
        """
        Obtain a substitution that would make these two types the same, if
        possible. Note that subtypes on the "self" side are tolerated: that is,
        if self is a subtype of other, then they are considered the same, but
        not vice versa.
        """
        if isinstance(self, TypeOperator) and isinstance(other, TypeOperator):

            if self.arity == 0 and other.arity == 0 and self.subtype(other):
                pass
            elif self.signature != other.signature:
                raise RuntimeError("type mismatch")
            else:
                for x, y in zip(self.types, other.types):
                    x.unify(y, ctx)
        elif isinstance(self, TypeVar):
            if self != other and self in other:
                raise RuntimeError("recursive type")
            else:
                ctx[self] = other
        elif isinstance(other, TypeVar):
            other.unify(self, ctx)
        return ctx

    def is_function(self) -> bool:
        if isinstance(self, TypeOperator):
            return self.name == 'function'
        return False

    def apply(self, arg: AlgebraType) -> AlgebraType:
        """
        Apply an argument to a function type to get its output type.
        """
        if self.is_function():
            input_type, output_type = self.types
            env = arg.unify(input_type, {})
            return output_type.substitute(env)
        else:
            raise RuntimeError("Cannot apply an argument to non-function type")

    def variables(self) -> Iterable[TypeVar]:
        """
        Obtain unbound variables left in the type expression.
        """
        if isinstance(self, TypeVar):
            yield self
        elif isinstance(self, TypeOperator):
            for v in chain(*(t.variables() for t in self.types)):
                yield v


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
            return self.signature == other.signature and \
                all(s.subtype(t) for s, t in zip(self.types, other.types))

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

    def _fresh(self, ctx: Dict[TypeVar, TypeVar]) -> TypeOperator:
        return TypeOperator(self.name, *(t._fresh(ctx) for t in self.types))

    def constrain(self, var: TypeVar, typeclass: TypeClass) -> None:
        for t in self.types:
            t.constrain(var, typeclass)

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
        self.typeclasses = set()
        cls.counter += 1

    def __str__(self) -> str:
        return "x" + str(self.id)

    def __contains__(self, value: AlgebraType) -> bool:
        return self == value

    def constrain(self, var: TypeVar, tc: TypeClass) -> None:
        if var == self:
            self.typeclasses.add(tc)

    def _fresh(self, ctx: Dict[TypeVar, TypeVar]) -> TypeVar:
        if self in ctx:
            return ctx[self]
        else:
            new = TypeVar()
            for tc in self.typeclasses:
                new.typeclasses.add(tc)
            ctx[self] = new
            return new


class TypeClass(object):

    def __init__(self, name: str, *type_parameters: AlgebraType, **kwargs):
        self.name = name
        self.types = type_parameters
        self.additional = kwargs

    def _fresh(self, ctx: Dict[TypeVar, TypeVar]) -> TypeClass:
        return TypeClass(self.name, *(t._fresh(ctx) for t in self.types))

    def enforce(self):
        """
        Check if a particular binding might satisfy membership of this
        typeclass. If such a thing would entail constraints on other variables,
        they are added. Raises an error if typeclass constraints cannot be
        satisfied.
        """
        return NotImplemented

    def __str__(self) -> str:
        return "{}({})".format(self.name, ", ".join(map(str, self.types)))

    def subtype(self):
        """
        Enforce types that are subsumed by the given superclass.
        """
        t1, t2 = self.types

        if isinstance(t2, TypeOperator) and isinstance(t1, TypeOperator):
            if t2.arity == 0:
                if not t2.subtype(t1):
                    raise RuntimeError("does not satisfy typeclass constraint")
            else:
                if t1.signature != t2.signature:
                    raise RuntimeError("does not satisfy typeclass constraint")
                for s, t in zip(t1.types, t2.types):
                    tc = TypeClass(self.name, s, t)
                    tc.enforce()
        elif isinstance(t1, TypeVar):
            pass

    def contains(self):
        """
        Enforce parameterized types that contain the given type in one of their
        columns.
        """
        t1, t2 = self.types

        if isinstance(t1, TypeOperator) and isinstance(t2, TypeOperator):
            tc = TypeClass("subtype", t1, t2.types[0])
            tc.enforce()
