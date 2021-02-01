"""
Generic type system. Inspired loosely by Hindley-Milner type inference in
functional programming languages.

Be warned: This module abuses overloading of Python's standard operators.
"""
from __future__ import annotations

from abc import ABC, ABCMeta
from functools import partial
from itertools import chain
from collections import defaultdict
from typing import Dict, Optional, Iterable, Union, List, Callable, Set

from quangis import error


class Variables(defaultdict):
    """
    For convenient notation, we provide a dispenser for type variables. Instead
    of writing x, y = TypeVar(), TypeVar() to use type variables x and y, we
    can just instantiate a var = Variables() object and get var.x, var.y on the
    fly.
    """

    def __init__(self):
        super().__init__(TypeVar)

    def __getattr__(self, key):
        return self[key]


class Definition(object):
    """
    This class defines a function: it knows its general type and constraints,
    plus additional information that may be used by some parser, and can
    generate fresh instances of the function.
    """

    def __init__(
            self,
            name: str,
            t: AlgebraType,
            *args: Union[AlgebraType, Constraint, int]):
        """
        Define a function type. Additional arguments are distinguished by their
        type. This helps simplify notation: we won't have to painstakingly
        write out every definition, and instead just provide a tuple of
        relevant information.
        """

        types = [t]
        constraints = []
        number_of_data_arguments = 0

        for arg in args:
            if isinstance(arg, Constraint):
                constraints.append(arg)
            elif isinstance(arg, AlgebraType):
                types.append(arg)
            elif isinstance(arg, int):
                number_of_data_arguments = arg
            else:
                raise ValueError(f"cannot use type {type(arg)} in Definition")

        # if more than one type is given, consider the function overloaded
        if len(types) > 1:
            t = TypeVar()
            constraints.append(t.limit(*types))

        self.name = name
        self.type = t
        self.constraints = constraints
        self.data = number_of_data_arguments

    def instance(self) -> AlgebraType:
        ctx: Dict[TypeVar, TypeVar] = {}
        t = self.type.fresh(ctx)
        for constraint in self.constraints:
            new_constraint = constraint.fresh(ctx)
            for var in new_constraint.variables():
                var.constraints.add(new_constraint)
        return t

    def __str__(self) -> str:
        return (
            f"{self.name} : {self.type}{', ' if self.constraints else ''}"
            f"{', '.join(str(c) for c in self.constraints)}"
        )


class TypeDefiner(ABCMeta):
    """
    Allowing us to write type definitions in an intuitive way, such as
    TypeOperator.Int() for basic types or TypeOperator.Tuple for parameterized
    types.
    """
    # TODO add a parameter that fixes the arity of the operator or perhaps even
    # constrains it arguments

    def __getattr__(self, key: str) -> Callable[..., TypeOperator]:
        return partial(TypeOperator, key)


class AlgebraType(ABC, metaclass=TypeDefiner):
    """
    Abstract base class for type operators and type variables. Note that basic
    types are just 0-ary type operators and functions are just particular 2-ary
    type operators.
    """

    def __repr__(self):
        return self.__str__()

    def __contains__(self, value: AlgebraType) -> bool:
        return value == self or (
            isinstance(self, TypeOperator) and
            any(value in t for t in self.types))

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

    def variables(self) -> Iterable[TypeVar]:
        """
        Obtain all type variables left in the type expression.
        """
        if isinstance(self, TypeVar):
            yield self
        elif isinstance(self, TypeOperator):
            for v in chain(*(t.variables() for t in self.types)):
                yield v

    def all_constraints(self) -> Set[Constraint]:
        """
        Obtain the constraints relevant on the variables.
        """
        return set.union(*(var.constraints for var in self.variables()))

    def fresh(self, ctx: Dict[TypeVar, TypeVar]) -> AlgebraType:
        """
        Create a fresh copy of this type, with unique new variables.
        """

        if isinstance(self, TypeOperator):
            new = TypeOperator(self.name, *(t.fresh(ctx) for t in self.types))
            new.supertype = self.supertype
            return new
        elif isinstance(self, TypeVar):
            if self.bound:
                raise error.AlreadyBound(self)
            elif self in ctx:
                return ctx[self]
            else:
                new2 = TypeVar()
                for tc in self.constraints:
                    new2.constraints.add(tc)
                ctx[self] = new2
                return new2
        raise ValueError(f"{self} is neither a type nor a type variable")

    def unify(self, other: AlgebraType) -> None:
        """
        Bind variables such that both types become the same. Note that subtypes
        on the "self" side are tolerated: that is, if self is a subtype of
        other, then they are considered the same, but not vice versa.
        """
        if isinstance(self, TypeOperator) and isinstance(other, TypeOperator):
            if self.match(other):
                for x, y in zip(self.types, other.types):
                    x.unify(y)
            else:
                raise error.TypeMismatch(self, other)
        else:
            a = self.binding(False)
            b = other.binding(False)
            if isinstance(a, TypeVar):
                if a != b and a in b:
                    raise error.RecursiveType(a, b)
                a.bind(b)
            elif isinstance(b, TypeVar):
                b.unify(a)

    def compatible(self, other: AlgebraType) -> bool:
        """
        Test if a type is structurally consistent with another, that is, if
        they fit together modulo variables. Subtypes allowed on the self side.
        """
        if isinstance(self, TypeOperator) and isinstance(other, TypeOperator):
            return self.match(other) and \
                all(s.compatible(t) for s, t in zip(self.types, other.types))
        return True

    def apply(self, arg: AlgebraType) -> AlgebraType:
        """
        Apply an argument to a function type to get its output type.
        """
        if isinstance(self, TypeOperator) and self.name == 'function':
            input_type, output_type = self.types
            arg.binding().unify(input_type.binding())
            return output_type.binding()
        elif isinstance(self, TypeVar):
            input_type = TypeVar()
            input_type.constraints.union(self.constraints)
            output_type = TypeVar()
            output_type.constraints.union(self.constraints)
            fn = input_type ** output_type
            fn.unify(self)
            return fn.apply(arg)
        else:
            raise error.NonFunctionApplication(self, arg)

    def binding(self, full: bool = True) -> AlgebraType:
        if isinstance(self, TypeVar) and self.bound:
            return self.bound.binding(full)
        elif full and isinstance(self, TypeOperator):
            self.types: List[AlgebraType] = [
                t.binding(full) for t in self.types]
        return self

    def limit(self, *options: AlgebraType) -> Constraint:
        """
        Produce a constraint ensuring that the subject must be one of several
        options.
        """
        return Constraint(self, *options)

    def has_param(
            self,
            op: Callable[..., TypeOperator],
            target: AlgebraType,
            at: Optional[int] = None,
            min: int = 1,
            max: int = 3) -> Constraint:
        """
        Produce a constraint ensuring that the subject must be a parameterized
        type operator `op` that contains the target at some point in its `min`
        to `max` parameters.
        """
        options: List[AlgebraType] = []
        positions = list(range(min, max+1)) if at is None else [at]

        for n in range(min, max+1):
            for p in positions:
                if n >= p:
                    options.append(op(*(
                        target if i == p else TypeVar()
                        for i in range(min, n+1))
                    ))
        return Constraint(self, *options)


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
            raise ValueError("functions must have 2 argument types")
        if self.supertype and (self.types or self.supertype.types):
            raise ValueError("only nullary types may have supertypes")

    def __str__(self) -> str:
        if self.name == 'function':
            return f"({self.types[0]} -> {self.types[1]})"
        elif self.types:
            return f'{self.name}({", ".join(str(t) for t in self.types)})'
        else:
            return self.name

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TypeOperator):
            return self.match(other, allow_subtype=False) and \
               all(s == t for s, t in zip(self.types, other.types))
        else:
            return False

    def match(self, other: TypeOperator, allow_subtype: bool = True) -> bool:
        """
        Check if the top-level type operator matches another (modulo subtypes).
        """
        return (
            (self.name == other.name and self.arity == other.arity) or
            (allow_subtype and bool(
                self.supertype and self.supertype.match(other)
            ))
        )

    @property
    def arity(self) -> int:
        return len(self.types)


class TypeVar(AlgebraType):

    counter = 0

    def __init__(self):
        cls = type(self)
        self.id = cls.counter
        self.bound = None
        self.constraints = set()
        cls.counter += 1

    def __str__(self) -> str:
        return f"x{self.id}"

    def bind(self, binding: AlgebraType):
        assert (not self.bound or binding == self.bound), \
            "variable cannot be bound twice"

        self.bound = binding

        for constraint in self.constraints:
            constraint.enforce()


class Constraint(object):
    """
    A constraint is a set of types that must remain consistent with a subject
    type.
    """

    def __init__(self, t: AlgebraType, *options: AlgebraType):
        self.subject = t
        self.initial_options = list(options)
        self.options = list(options)

        for t in self.options:
            for v in self.subject.variables():
                if v in t:
                    raise error.RecursiveType(v, t)

    def __str__(self) -> str:
        return (
            f"{self.subject.binding()} must be one of "
            f"[{', '.join(str(t) for t in self.initial_options)}]"
        )

    def fresh(self, ctx: Dict[TypeVar, TypeVar]) -> Constraint:
        return Constraint(
            self.subject.fresh(ctx),
            *(t.fresh(ctx) for t in self.options))

    def variables(self) -> Iterable[TypeVar]:
        return chain(
            self.subject.variables(),
            *(t.variables() for t in self.options))

    def enforce(self):
        self.options = [
            t for t in self.options
            if self.subject.binding().compatible(t.binding())
        ]

        if len(self.options) == 0:
            raise error.ViolatedConstraint(self)
        elif len(self.options) == 1:
            self.subject.binding().unify(self.options[0].binding())


