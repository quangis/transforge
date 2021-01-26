"""
Generic type system. Inspired loosely by Hindley-Milner type inference in
functional programming languages.

Be warned: This module abuses overloading of Python's standard operators. It
also deviates from Python's convention of using capitalized names for classes
and lowercase for values. These decisions were made to get an interface that is
as close as possible to its formal type system counterpart.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Callable, Optional, Tuple


class AlgebraType(ABC):
    """
    Abstract base class for type operators and type variables. Note that basic
    types are just 0-ary type operators.
    """

    def __pow__(self: AlgebraType, other: AlgebraType) -> Transformation:
        """
        This is an overloaded (ab)use of Python's exponentiation operator. It
        allows us to use the infix operator ** for the arrow in function
        signatures.

        Note that this operator is one of the few that is right-to-left
        associative, matching the conventional behaviour of the function arrow.
        The right-bitshift operator >> (for __rshift__) would have been more
        intuitive visually, but does not have this property.
        """
        return Transformation(self, other)

    def __or__(a: AlgebraType, b: Dict[TypeVar, TypeClass]) -> AlgebraType:
        """
        """
        #for constraint in b:
        #    return a.constrain(b)
        return a

    #def constrain(self, constraint: TypeConstraint):
    #    self.

    def __repr__(self):
        return self.__str__()

    @abstractmethod
    def __contains__(self, value: AlgebraType) -> bool:
        return NotImplemented

    @abstractmethod
    def _fresh(self, ctx: Dict[TypeVar, TypeVar]) -> AlgebraType:
        return NotImplemented

    @abstractmethod
    def map(self, fn: Callable[[AlgebraType], AlgebraType]) -> AlgebraType:
        return NotImplemented

    def fresh(self) -> AlgebraType:
        """
        Create a fresh copy of this type, with unique new variables.
        """

        return self._fresh({})
        #ctx: Dict[TypeVar, TypeVar] = {}
#
#        def f(t):
#            if isinstance(t, TypeVar):
#                if self in ctx:
#                    return ctx[self]
#                else:
#                    new = TypeVar.new()
#                    ctx[self] = new
#                    return new
#            return t

#        return self.map(f)

    def apply(self, arg: AlgebraType) -> AlgebraType:
        raise RuntimeError("Cannot apply an argument to non-function type")

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
        possible. Note that subtypes on the "self" side are tolerated.
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

    def __eq__(self, other: object):
        if isinstance(other, TypeOperator):
            return self.signature == other.signature and \
               all(s == t for s, t in zip(self.types, other.types))
        else:
            return False

    def __contains__(self, value: AlgebraType) -> bool:
        return value == self or any(value in t for t in self.types)

    def __str__(self) -> str:
        if self.types:
            return "{}({})".format(self.name, ", ".join(map(str, self.types)))
        else:
            return self.name

    def _fresh(self, ctx: Dict[TypeVar, TypeVar]) -> TypeOperator:
        return TypeOperator(self.name, *(t._fresh(ctx) for t in self.types))

    def map(self, fn: Callable[[AlgebraType], AlgebraType]) -> AlgebraType:
        return TypeOperator(self.name, *map(fn, self.types))

    @property
    def arity(self) -> int:
        return len(self.types)

    @property
    def signature(self) -> Tuple[str, int]:
        return self.name, self.arity


class Transformation(TypeOperator):

    def __init__(self, input_type: AlgebraType, output_type: AlgebraType):
        super().__init__("transformation", input_type, output_type)

    def map(self, fn: Callable[[AlgebraType], AlgebraType]) -> AlgebraType:
        return Transformation(*map(fn, self.types))

    def _fresh(self, ctx: Dict[TypeVar, TypeVar]) -> Transformation:
        return Transformation(*(t._fresh(ctx) for t in self.types))

    def __str__(self) -> str:
        return "({0} -> {1})".format(*self.types)

    def apply(self, arg: AlgebraType) -> AlgebraType:
        """
        Apply an argument to a function type to get its output type.
        """
        input_type, output_type = self.types
        env = arg.unify(input_type, {})
        return output_type.substitute(env)


class TypeVar(AlgebraType):

    counter = 0

    def __init__(self):
        cls = type(self)
        self.id = cls.counter
        cls.counter += 1

    def __str__(self) -> str:
        return "x" + str(self.id)

    def __contains__(self, value: AlgebraType) -> bool:
        return self == value

    def map(self, fn: Callable[[AlgebraType], AlgebraType]) -> AlgebraType:
        return fn(self)

    def _fresh(self, ctx: Dict[TypeVar, TypeVar]) -> TypeVar:
        if self in ctx:
            return ctx[self]
        else:
            new = TypeVar()
            ctx[self] = new
            return new


class TypeClass(object):
    pass


class Contains(TypeClass):
    """
    Typeclass for relation types that contain the given value type in one of
    their columns.
    """

    def __init__(self, domain: AlgebraType, at: Optional[int] = None):
        self.domain = domain


class Sub(TypeClass):
    """
    Typeclass for value types that are subsumed by the given superclass.
    """

    def __init__(self, supertype: TypeOperator):
        pass

