"""
Generic type system. Inspired loosely by Hindley-Milner type inference in
functional programming languages.
"""
# A primer: A type consists of type operators and type variables. Type
# operators encompass basic types, parameterized types and functions. When
# applying an argument of type A to a function of type B ** C, the algorithm
# tries to bind variables in such a way that A becomes equal to B. Constraints
# can be added to variables to make place further conditions on them;
# otherwise, variables are universally quantified. Constraints are enforced
# whenever a relevant variable is bound.
# When we bind a type to a type variable, binding happens on the type variable
# object itself. That is why we make fresh copies of generic type
# expressions before using them or adding constraints to them. This means that
# pointers are somewhat interwoven --- keep this in mind.
# To understand the module, I recommend you start by reading the methods of the
# PlainType class.
from __future__ import annotations

from enum import Enum
from abc import ABC, abstractmethod
from functools import reduce
from itertools import chain
from inspect import signature
from collections import defaultdict
from typing import Dict, Optional, Iterable, Union, List, Callable

from quangis import error


class Variance(Enum):
    """
    The variance of a type parameter indicates how subtype relations of
    compound types relate to their constituent types. For example, a function
    type α₁ → β₁ is contravariant in its input parameter (consider that a
    subtype α₂ → β₂ ≤ α₁ → β₁ must be just as liberal or more in what input it
    accepts, e.g. α₁ ≤ α₂) and covariant in its output parameter (it must be
    just as conservative or more in what output it produces, e.g. β₂ ≤ β₁).
    """

    COVARIANT = 0
    CONTRAVARIANT = 1


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


class Signature(object):
    """
    This class provides the definition of a *signature* for a function or for
    data: it knows its general type and constraints, and can generate fresh
    instances, possibly containing fresh variables.
    """

    def __init__(
            self,
            t: Typish,
            name: str = "anonymous",
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
                raise ValueError(f"cannot use extra {type(arg)} in Signature")

        self.name = name
        self.type = Type.coerce(t)
        self.type.constraints.extend(constraints)
        self.data = number_of_data_arguments

    def instance(self) -> Type:
        return self.type.fresh({})

    def __call__(self, *args: Typish) -> Type:
        return self.instance().__call__(*args)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"{self.name} : {self.type}"


class Type(object):
    """
    A top-level type term decorated with constraints.
    """

    def __init__(self, t: PlainType, constraints: Iterable[Constraint] = ()):
        self.plain = t
        self.constraints = list(constraints)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        res = [str(self.plain)]

        if self.constraints:
            res.append(', '.join(str(c) for c in self.constraints))

        for v in self.plain.variables():
            if v.lower or v.upper:
                res.append(f"{v.lower or '?'} <= {v} <= {v.upper or '?'}")

        return ', '.join(res)

    @staticmethod
    def coerce(t: Typish) -> Type:
        if isinstance(t, PlainType):
            return Type(t)
        elif isinstance(t, Signature):
            return t.instance()
        elif isinstance(t, Type):
            return t
        elif callable(t):
            n = len(signature(t).parameters)
            return Type.coerce(t(*(TypeVar() for _ in range(n))))  # bind fresh variables
        else:
            raise ValueError(f"Cannot convert a {type(t)} to a Type")

    def __call__(self, *args: Typish) -> Type:
        return reduce(Type.apply, (Type.coerce(a) for a in args), self)

    def apply(self, arg: Type) -> Type:
        """
        Apply an argument to a function type to get its resolved output type.
        """

        if isinstance(self.plain, TypeOperator) and self.plain.constructor == Function:
            input_type, output_type = self.plain.params
            arg_type = arg.plain

            arg_type.unify(input_type)

            return Type(
                output_type,
                (constraint
                    for constraint in chain(self.constraints, arg.constraints)
                    if not constraint.fulfilled())
            )
        else:
            raise error.NonFunctionApplication(self.plain, arg)

    def fresh(self, ctx: Dict[TypeVar, TypeVar]) -> Type:
        return Type(
            self.plain.fresh(ctx),
            (constraint.fresh(ctx) for constraint in self.constraints)
        )


class PlainType(ABC):
    """
    Abstract base class for type operators and type variables. Note that basic
    types are just 0-ary type operators and functions are just particular 2-ary
    type operators.
    """

    def __repr__(self):
        return self.__str__()

    def __contains__(self, value: PlainType) -> bool:
        return value == self or (
            isinstance(self, TypeOperator) and
            any(value in t for t in self.params))

    def __pow__(self, other: PlainType) -> TypeOperator:
        """
        This is an overloaded (ab)use of Python's exponentiation operator. It
        allows us to use the infix operator ** for the arrow in function
        signatures.

        Note that this operator is one of the few that is right-to-left
        associative, matching the conventional behaviour of the function arrow.
        The right-bitshift operator >> (for __rshift__) would have been more
        intuitive visually, but does not have this property.
        """
        return TypeOperator(Function, self, other)

    def variables(self) -> Iterable[TypeVar]:
        """
        Obtain all type variables currently in the type expression.
        """
        if isinstance(self, TypeOperator):
            for v in chain(*(t.variables() for t in self.params)):
                yield v
        else:
            a = self.resolve(full=False)
            if isinstance(a, TypeVar):
                yield a
            else:
                for v in a.variables():
                    yield v

    def fresh(self, ctx: Dict[TypeVar, TypeVar]) -> PlainType:
        """
        Create a fresh copy of this type, with unique new variables.
        """

        if isinstance(self, TypeOperator):
            return TypeOperator(
                self.constructor,
                *(t.fresh(ctx) for t in self.params))
        elif isinstance(self, TypeVar):
            if self in ctx:
                return ctx[self]
            else:
                new = TypeVar(wildcard=self.wildcard)
                ctx[self] = new
                return new
        raise ValueError(f"{self} is neither a type nor a type variable")

    def resolve(self, full: bool = True) -> PlainType:
        """
        A `full` resolve obtains a version of this type with all bound
        variables replaced with their bindings. Otherwise, just resolve the
        current variable.
        """
        if isinstance(self, TypeVar) and self.bound:
            return self.bound.resolve(full)
        elif full and isinstance(self, TypeOperator):
            return TypeOperator(
                self.constructor,
                *(t.resolve(full) for t in self.params))
        return self

    def compatible(
            self,
            other: PlainType,
            allow_subtype: bool = False) -> bool:
        """
        Is the type structurally consistent with another, that is, do they
        'fit', modulo variables. Subtypes may be allowed on the self side.
        """
        if isinstance(self, TypeOperator) and isinstance(other, TypeOperator):
            return (
                self.constructor <= other.constructor and
                all(s.compatible(t, allow_subtype)
                    for s, t in zip(self.params, other.params))
            )
        return True

    def skeleton(self) -> PlainType:
        """
        Create a copy of this operator, substituting fresh variables for basic
        types.
        """
        if isinstance(self, TypeOperator):
            if self.constructor.basic:
                return TypeVar()
            else:
                return TypeOperator(
                    self.constructor,
                    *(p.skeleton() for p in self.params))
        else:
            return self

    def unify(self, other: PlainType) -> None:
        """
        Make sure that a is equal to, or a subtype of b. Like normal
        unification, but instead of just a substitution of variables to terms,
        also produces lower and upper bounds on subtypes that it must respect.
        """
        a = self.resolve(False)
        b = other.resolve(False)

        if isinstance(a, TypeOperator) and isinstance(b, TypeOperator):
            if a.constructor.basic:
                if not (a.constructor <= b.constructor):
                    raise error.SubtypeMismatch(a, b)
            elif a.constructor == b.constructor:
                for v, x, y in zip(a.constructor.variance, a.params, b.params):
                    if v == Variance.COVARIANT:
                        x.unify(y)
                    elif v == Variance.CONTRAVARIANT:
                        y.unify(x)
            else:
                raise error.TypeMismatch(a, b)

        elif isinstance(a, TypeVar) and isinstance(b, TypeVar):
            a.bind(b)

        elif isinstance(a, TypeVar) and isinstance(b, TypeOperator):
            if a in b:
                raise error.RecursiveType(a, b)
            elif b.constructor.basic:
                a.below(b.constructor)
            else:
                a.bind(b.skeleton())
                a.unify(b)

        elif isinstance(a, TypeOperator) and isinstance(b, TypeVar):
            if b in a:
                raise error.RecursiveType(b, a)
            elif a.constructor.basic:
                b.above(a.constructor)
            else:
                b.bind(a.skeleton())
                b.unify(a)

    def __call__(self, *args: Typish) -> Type:
        return Type(self).__call__(*args)

    # constraints #############################################################

    def subtype(self, *patterns: PlainType) -> Constraint:
        return NoConstraint(self, *patterns, allow_subtype=True)

    def member(self, *patterns: PlainType) -> Constraint:
        return NoConstraint(self, *patterns, allow_subtype=False)

    def param(
            self, target: PlainType,
            subtype: bool = False,
            at: Optional[int] = None) -> Constraint:
        return NoConstraint(
            self, target)  # , position=at, allow_subtype=subtype)


class TypeConstructor(object):
    """
    An n-ary type constructor.
    """

    def __init__(
            self,
            name: str,
            *variance: Variance,
            supertype: Union[None, TypeConstructor, TypeOperator] = None):
        self.name = name
        self.variance = list(variance)

        self.supertype: Optional[TypeConstructor] = None
        if isinstance(supertype, TypeOperator):
            self.supertype = supertype.constructor
        else:
            self.supertype = supertype

        if self.supertype and not self.basic:
            raise ValueError("only nullary types can have direct supertypes")

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, TypeConstructor) and
            self.name == other.name
            and self.variance == other.variance)

    def __le__(self, other: TypeConstructor) -> bool:
        return self == other or self < other

    def __lt__(self, other: TypeConstructor) -> bool:
        return bool(self.supertype and self.supertype <= other)

    def __call__(self, *params) -> TypeOperator:
        return TypeOperator(self, *params)

    @property
    def basic(self) -> bool:
        return self.arity == 0

    @property
    def arity(self) -> int:
        return len(self.variance)


class TypeOperator(PlainType):
    """
    An instance of an n-ary type constructor.
    """

    def __init__(self, constructor: TypeConstructor, *params: PlainType):
        self.constructor = constructor
        self.params: List[PlainType] = list(params)

        if len(self.params) != self.constructor.arity:
            raise ValueError

    def __str__(self) -> str:
        if self.constructor == Function:
            inT, outT = self.params
            if isinstance(inT, TypeOperator) and inT.constructor == Function:
                return f"({inT}) ** {outT}"
            return f"{inT} ** {outT}"
        elif self.params:
            return f'{self.constructor}({", ".join(str(t) for t in self.params)})'
        else:
            return str(self.constructor)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TypeOperator):
            return self.constructor == other.constructor and \
               all(s == t for s, t in zip(self.params, other.params))
        else:
            return False


class TypeVar(PlainType):
    """
    Type variable. Note that any bindings and constraints are bound to the
    actual object instance, so make a fresh copy before applying them if the
    variable is not supposed to be a concrete instance.
    """

    counter = 0

    def __init__(self, wildcard: bool = False):
        cls = type(self)
        self.id = cls.counter
        self.bound: Optional[PlainType] = None
        self.lower: Optional[TypeConstructor] = None
        self.upper: Optional[TypeConstructor] = None
        self.wildcard = wildcard
        cls.counter += 1

    def __str__(self) -> str:
        return "_" if self.wildcard else f"x{self.id}"

    def bind(self, binding: PlainType) -> None:
        assert (not self.bound or binding == self.bound), \
            "variable cannot be bound twice"

        if binding is not self:
            self.bound = binding

            if isinstance(binding, TypeVar):
                if self.lower:
                    binding.above(self.lower)
                if self.upper:
                    binding.below(self.upper)
                if binding.lower:
                    self.above(binding.lower)
                if binding.upper:
                    self.below(binding.upper)

    def above(self, new: TypeConstructor) -> None:
        """
        Constrain this variable to be a basic type with the given type as lower
        bound.
        """
        lower, upper = self.lower or new, self.upper or new

        # lower bound higher than the upper bound fails
        if upper < new:
            raise error.SubtypeMismatch(new, upper)

        # lower bound lower than the current lower bound is ignored
        elif new < lower:
            pass

        # tightening the lower bound
        elif lower <= new:
            self.lower = new

        # new bound from another lineage (neither sub- nor supertype) fails
        else:
            raise error.SubtypeMismatch(lower, new)

    def below(self, new: TypeConstructor) -> None:
        """
        Constrain this variable to be a basic type with the given subtype as
        upper bound.
        """
        # symmetric to subtype
        lower, upper = self.lower or new, self.upper or new
        if new < lower:
            raise error.SubtypeMismatch(lower, new)
        elif upper < new:
            pass
        elif new <= upper:
            self.upper = new
        else:
            raise error.SubtypeMismatch(new, upper)


"The special constructor for function types."
Function = TypeConstructor(
    'Function',
    Variance.CONTRAVARIANT,
    Variance.COVARIANT)

"A shortcut for writing function signatures."
Σ = Signature

"A shortcut for writing type constructors."
τ = TypeConstructor

Typish = Union[Type, PlainType, Signature, Callable[..., Type]]


class Constraint(ABC):
    """
    A constraint enforces that its subject type always remains consistent with
    whatever condition it represents.
    """

    def __init__(
            self,
            type: PlainType,
            *patterns: PlainType,
            allow_subtype: bool = False):
        self.subject = type
        self.patterns = list(patterns)
        self.allow_subtype = allow_subtype

    def variables(self) -> Iterable[TypeVar]:
        return chain(
            self.subject.variables(),
            *(t.variables() for t in self.patterns))

    def resolve(self, full: bool = True) -> Constraint:
        self.subject = self.subject.resolve(full)
        self.patterns = [t.resolve(full) for t in self.patterns]
        self.enforce()
        return self

    @abstractmethod
    def fresh(self, ctx: Dict[TypeVar, TypeVar]):
        return NotImplemented

    def fulfilled(self) -> bool:
        self.resolve(True)
        return not any(not var.wildcard for var in self.variables())

    @abstractmethod
    def enforce(self) -> None:
        """
        Check that the resolved constraint is still satisfied.
        """
        raise NotImplementedError


class NoConstraint(Constraint):
    """
    Temporarily shut off other constraints.
    """

    def fresh(self, ctx) -> NoConstraint:
        return NoConstraint(
            self.subject.fresh(ctx),
            (*(t.fresh(ctx) for t in self.patterns)))

    def enforce(self) -> None:
        pass


class MembershipConstraint(Constraint):
    """
    A membership constraint checks that the subject is one of the given types.
    """

    def __str__(self) -> str:
        return (
            f"{self.subject} must be "
            f"{'a subtype of' if self.allow_subtype else ''} "
            f"{' or '.join(str(t) for t in self.patterns)}"
        )

    def fresh(self, ctx: Dict[TypeVar, TypeVar]) -> MembershipConstraint:
        return MembershipConstraint(
            self.subject.fresh(ctx),
            *(t.fresh(ctx) for t in self.patterns),
            allow_subtype=self.allow_subtype)

    def enforce(self) -> None:
        if not any(
                self.subject.compatible(pattern, self.allow_subtype)
                for pattern in self.patterns):
            raise error.ViolatedConstraint(self)


class ParameterConstraint(Constraint):
    """
    A parameter constraint checks that the subject is a parameterized type with
    one of the given types occurring somewhere in its parameters.
    """

    def __init__(
            self,
            *nargs,
            position: Optional[int] = None,
            **kwargs):
        self.position = position
        super().__init__(*nargs, **kwargs)

    def fresh(self, ctx: Dict[TypeVar, TypeVar]) -> ParameterConstraint:
        return ParameterConstraint(
            self.subject.fresh(ctx),
            *(t.fresh(ctx) for t in self.patterns),
            allow_subtype=self.allow_subtype,
            position=self.position)

    def __str__(self) -> str:
        return (
            f"{self.subject} must have "
            f"{'a subtype of ' if self.allow_subtype else ''}"
            f"{' or '.join(str(t) for t in self.patterns)} as parameter"
            f"{'' if self.position is None else ' at #' + str(self.position)}"
        )

    def compatible(self, pattern: PlainType) -> bool:
        if isinstance(self.subject, TypeOperator):
            if self.position is None:
                return any(
                    param.compatible(pattern, self.allow_subtype)
                    for param in self.subject.params)
            elif self.position-1 < len(self.subject.params):
                return self.subject.params[self.position-1].compatible(
                    pattern, self.allow_subtype)
            return False
        else:
            return True

    def enforce(self) -> None:
        if not any(self.compatible(pattern) for pattern in self.patterns):
            raise error.ViolatedConstraint(self)
