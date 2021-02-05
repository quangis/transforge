"""
Generic type system. Inspired loosely by Hindley-Milner type inference in
functional programming languages.
"""
# This module abuses overloading of Python's standard operators. A more urgent
# warning: objects are intricately linked, meaning that changes may have
# unforeseen side-effects. However, this should be limited to this particular
# module, as long as it is used in the intended way.
#
# A primer: Applying an argument A to a function B -> C works by trying to bind
# type variables in such a way that A becomes equal to B. Type variables are
# created once, and binding them happens on the objects themselves. That is why
# we copy fresh instances of generic type expressions before using them or
# adding constraints to them. Type operators encompass all other type
# components: basic types, parameterized types and functions. For them,
# pointers are not as meaningful. Constraints are added to relevant variables
# and checked whenever that variable is bound. To understand the module, start
# by reading the methods of the AlgebraType class.
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
    of writing x = TypeVar() to introduce type variable x every time, we can
    just instantiate a var = Variables() object and get var.x, var.y on the
    fly. To get a wildcard variable, use 'var._'; the assumption is that a
    wildcard will never be used anywhere else, so it will return a new type
    variable every time.
    """

    def __init__(self):
        super().__init__(TypeVar)

    def __getattr__(self, key):
        if key == '_':
            return TypeVar(wildcard=True)
        return self[key]


class Definition(object):
    """
    This class defines a function: it knows its general type and constraints,
    and can generate fresh instances.
    """

    def __init__(
            self,
            name: str,
            t: AlgebraType,
            *args: Union[Constraint, int]):
        """
        Define a function type. Additional arguments are distinguished by their
        type. This helps simplify notation: we won't have to painstakingly
        write out every definition, and instead just provide a tuple of
        relevant information.
        """

        constraints = []
        number_of_data_arguments = 0

        for arg in args:
            if isinstance(arg, Constraint):
                constraints.append(arg)
            elif isinstance(arg, int):
                number_of_data_arguments = arg
            else:
                raise ValueError(f"cannot use extra {type(arg)} in Definition")

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

    def is_resolved(self) -> bool:
        """
        Test if every variable in this type is resolved.
        """
        return all(var.bound is None for var in self.variables())

    def variables(self) -> Iterable[TypeVar]:
        """
        Obtain all type variables currently in the type expression.
        """
        if isinstance(self, TypeOperator):
            for v in chain(*(t.variables() for t in self.types)):
                yield v
        else:
            a = self.resolve(full=False)
            if isinstance(a, TypeVar):
                yield a
            else:
                for v in a.variables():
                    yield v

    def fresh(self, ctx: Dict[TypeVar, TypeVar]) -> AlgebraType:
        """
        Create a fresh copy of this type, with unique new variables.
        """

        if isinstance(self, TypeOperator):
            return TypeOperator(
                self.name,
                *(t.fresh(ctx) for t in self.types),
                supertype=self.supertype)
        elif isinstance(self, TypeVar):
            assert self.is_resolved(), \
                "Cannot create a copy of a type with bound variables"
            if self in ctx:
                return ctx[self]
            else:
                new = TypeVar(
                    (c.fresh(ctx) for c in self.constraints),
                    wildcard=self.wildcard)
                ctx[self] = new
                return new
        raise ValueError(f"{self} is neither a type nor a type variable")

    def unify(self, other: AlgebraType) -> Set[Constraint]:
        """
        Bind variables such that both types become the same. Binding is a
        side-effect; use resolve() to consolidate the bindings. Returns a set
        of constraints whose enforcement should be triggered as a result.
        """
        result: Set[Constraint] = set()
        a = self.resolve(full=False)
        b = other.resolve(full=False)
        if a is not b:
            if isinstance(a, TypeOperator) and isinstance(b, TypeOperator):
                if a.match(b):
                    for x, y in zip(a.types, b.types):
                        result = result.union(x.unify(y))
                        x.unify(y)
                else:
                    raise error.TypeMismatch(a, b)
            else:
                if (isinstance(a, TypeOperator) and b in a) or \
                        (isinstance(b, TypeOperator) and a in b):
                    raise error.RecursiveType(a, b)
                elif isinstance(a, TypeVar) and a != b:
                    a.bind(b)
                    result = result.union(a.constraints)
                elif isinstance(b, TypeVar) and a != b:
                    b.bind(a)
                    result = result.union(b.constraints)
        return result

    def apply(self, arg: AlgebraType) -> AlgebraType:
        """
        Apply an argument to a function type to get its resolved output type.
        """
        if isinstance(self, TypeOperator) and self.name == 'function':
            input_type, output_type = self.types
            for constraint in arg.unify(input_type):
                constraint.resolve()
                constraint.enforce()
            return output_type.resolve()
        else:
            raise error.NonFunctionApplication(self, arg)

    def resolve(self, full: bool = True) -> AlgebraType:
        """
        A `full` resolve obtains a version of this type with all bound
        variables replaced with their bindings. Otherwise, just resolve the
        current variable.
        """
        if isinstance(self, TypeVar) and self.bound:
            return self.bound.resolve(full)
        elif full and isinstance(self, TypeOperator):
            return TypeOperator(
                self.name,
                *(t.resolve(full) for t in self.types),
                supertype=self.supertype)
        return self

    def compatible(
            self,
            other: AlgebraType,
            allow_subtype: bool = False) -> bool:
        """
        Is the type structurally consistent with another, that is, do they
        'fit', modulo variables. Subtypes may be allowed on the self side.
        """
        if isinstance(self, TypeOperator) and isinstance(other, TypeOperator):
            return self.match(other, allow_subtype) and all(
                s.compatible(t, allow_subtype)
                for s, t in zip(self.types, other.types))
        return True

    # constraints #############################################################

    def subtype(self, *patterns: AlgebraType) -> Constraint:
        """
        Ensure that the subject is a subtype of something else.
        """
        return self.is_(*patterns, subtype=True)

    def is_(self, *patterns: AlgebraType, subtype: bool = False) -> Constraint:
        """
        Ensure that the subject is one of several options. Subtypes may be
        allowed.
        """
        return Constraint(self, *patterns, allow_subtype=subtype)

    def has(
            self,
            op: Callable[..., TypeOperator],
            target: AlgebraType,
            subtype: bool = False,
            at: Optional[int] = None,
            min: int = 1,
            max: int = 3) -> Constraint:
        """
        Produce a constraint ensuring that the subject must be a parameterized
        type operator `op` that contains the target at some point in its `min`
        to `max` parameters.
        """
        patterns: List[AlgebraType] = []
        positions = list(range(min, max+1)) if at is None else [at]

        for n in range(min, max+1):
            for p in positions:
                if n >= p:
                    patterns.append(op(*(
                        target if i == p else TypeVar(wildcard=True)
                        for i in range(min, n+1))
                    ))
        return Constraint(self, *patterns, allow_subtype=subtype)


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
        self.supertype = supertype
        self.types: List[AlgebraType] = list(types)

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

    def match(self, other: TypeOperator, allow_subtype: bool = False) -> bool:
        """
        Check if the top-level type operator matches another.
        """
        return (
            (self.name == other.name and self.arity == other.arity) or
            (allow_subtype and bool(
                self.supertype and self.supertype.match(other, allow_subtype)
            ))
        )

    @property
    def arity(self) -> int:
        return len(self.types)


class TypeVar(AlgebraType):
    """
    Type variable. Note that any bindings and constraints are bound to the
    actual object instance, so make a fresh copy before applying them if the
    variable is not supposed to be a concrete instance.
    """

    counter = 0

    def __init__(
            self,
            constraints: Iterable[Constraint] = (),
            wildcard: bool = False):
        cls = type(self)
        self.id = cls.counter
        self.bound: Optional[AlgebraType] = None
        self.constraints = set(constraints)
        self.wildcard = wildcard
        cls.counter += 1

    def __str__(self) -> str:
        return "_" if self.wildcard else f"x{self.id}"

    def bind(self, binding: AlgebraType) -> None:
        assert (not self.bound or binding == self.bound), \
            "variable cannot be bound twice"

        # Once a variable has been bound, its constraints must carry over to
        # the variables in its binding. Consider a variable x that is
        # constrained to T(A) or T(B); and it has now been bound to T(z). This
        # can work, but only if binding z will still trigger the check that the
        # initial constraint still holds.
        for var in binding.variables():
            var.constraints = set.union(var.constraints, self.constraints)

        self.bound = binding


class Constraint(object):
    """
    A constraint is a set of types, at least one of which must always remain
    consistent with its subject type.
    """

    def __init__(
            self,
            t: AlgebraType,
            *patterns: AlgebraType,
            allow_subtype: bool = False):
        self.subject = t
        self.patterns = list(patterns)
        self.allow_subtype = allow_subtype

        # In general, a recursivity check would need to inspect a web of
        # interdependencies --- and anyway, it isn't needed because we will not
        # unify, only check

    def __str__(self) -> str:
        return (
            f"{self.subject} must be "
            f"{'a subtype of' if self.allow_subtype else ''} "
            f"{' or '.join(str(t) for t in self.patterns)}"
        )

    def fresh(self, ctx: Dict[TypeVar, TypeVar]) -> Constraint:
        return Constraint(
            self.subject.fresh(ctx),
            *(t.fresh(ctx) for t in self.patterns),
            allow_subtype=self.allow_subtype)

    def variables(self) -> Iterable[TypeVar]:
        return chain(
            self.subject.variables(),
            *(t.variables() for t in self.patterns))

    def resolve(self) -> None:
        self.subject = self.subject.resolve(full=True)
        self.patterns = [t.resolve(full=True) for t in self.patterns]

    def enforce(self) -> None:
        if not any(
                self.subject.compatible(t, self.allow_subtype)
                for t in self.patterns):
            raise error.ViolatedConstraint(self)
