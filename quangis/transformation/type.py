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
# TypeTerm class.
from __future__ import annotations

from enum import Enum
from abc import ABC, abstractmethod
from functools import partial, reduce
from itertools import chain
from collections import defaultdict
from typing import Dict, Optional, Iterable, Union, List, Callable

from quangis import error


class Variance(Enum):
    INVARIANT = 0
    COVARIANT = 1
    CONTRAVARIANT = 2


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
            type: TypeTerm,
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
        self.type = QTypeTerm(type, constraints)
        self.data = number_of_data_arguments

    def instance(self) -> QTypeTerm:
        return self.type.fresh({})

    def __call__(self, *args: Definition) -> QTypeTerm:
        return self.instance().__call__(*map(Definition.instance, args))

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"{self.name} : {self.type}"


class QTypeTerm(object):
    """
    A constrained type term.
    """

    def __init__(self, type: TypeTerm, constraints: Iterable[Constraint] = ()):
        self.plain = type
        self.constraints = list(constraints)

        # bound_variables = set(self.plain.variables())
        # for constraint in self.constraints:
        #    if not all(
        #            var.wildcard or var in bound_variables
        #            for var in constraint.variables()):
        #        raise ValueError(
        #            "all variables in a constraint must be bound by "
        #            "an occurrence in the accompanying type signature")

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        if self.constraints:
            return f"{self.plain} such that {', '.join(str(c) for c in self.constraints)}"
        else:
            return str(self.plain)

    def __call__(self, *args: QTypeTerm) -> QTypeTerm:
        return reduce(QTypeTerm.apply, args, self)

    def fresh(self, ctx: Dict[TypeVar, TypeVar]) -> QTypeTerm:
        return QTypeTerm(
            self.plain.fresh(ctx),
            (constraint.fresh(ctx) for constraint in self.constraints)
        )

    def apply(self, arg: QTypeTerm) -> QTypeTerm:
        """
        Apply an argument to a function type to get its resolved output type.
        """
        if isinstance(self.plain, TypeOperator) and self.plain.name == 'function':
            input_type, output_type = self.plain.params
            arg.plain.unify(input_type)
            return QTypeTerm(
                output_type.resolve(),
                (constraint
                    for constraint in chain(self.constraints, arg.constraints)
                    if not constraint.fulfilled())
            )
        else:
            raise error.NonFunctionApplication(self.plain, arg)


class TypeTerm(ABC):
    """
    Abstract base class for type operators and type variables. Note that basic
    types are just 0-ary type operators and functions are just particular 2-ary
    type operators.
    """

    def __repr__(self):
        return self.__str__()

    def __contains__(self, value: TypeTerm) -> bool:
        return value == self or (
            isinstance(self, TypeOperator) and
            any(value in t for t in self.params))

    def __pow__(self, other: TypeTerm) -> TypeOperator:
        """
        This is an overloaded (ab)use of Python's exponentiation operator. It
        allows us to use the infix operator ** for the arrow in function
        signatures.

        Note that this operator is one of the few that is right-to-left
        associative, matching the conventional behaviour of the function arrow.
        The right-bitshift operator >> (for __rshift__) would have been more
        intuitive visually, but does not have this property.
        """
        return TypeOperator(
            'function',
            (Variance.CONTRAVARIANT, Variance.COVARIANT),
            self,
            other)

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
            for v in chain(*(t.variables() for t in self.params)):
                yield v
        else:
            a = self.resolve(full=False)
            if isinstance(a, TypeVar):
                yield a
            else:
                for v in a.variables():
                    yield v

    def fresh(self, ctx: Dict[TypeVar, TypeVar]) -> TypeTerm:
        """
        Create a fresh copy of this type, with unique new variables.
        """

        if isinstance(self, TypeOperator):
            return TypeOperator(
                self.name,
                self.variance,
                *(t.fresh(ctx) for t in self.params),
                supertype=self.supertype)
        elif isinstance(self, TypeVar):
            assert self.is_resolved(), \
                "Cannot create a copy of a type with bound variables"
            if self in ctx:
                return ctx[self]
            else:
                new = TypeVar(wildcard=self.wildcard)
                ctx[self] = new
                return new
        raise ValueError(f"{self} is neither a type nor a type variable")

    def unify(self, other: TypeTerm) -> None:
        """
        Bind variables such that both types become the same. Binding is a
        side-effect; use resolve() to consolidate the bindings.
        """
        a = self.resolve(full=False)
        b = other.resolve(full=False)
        if a is not b:
            if isinstance(a, TypeOperator) and isinstance(b, TypeOperator):
                if a.match(b):
                    for x, y in zip(a.params, b.params):
                        x.unify(y)
                else:
                    raise error.TypeMismatch(a, b)
            else:
                if (isinstance(a, TypeOperator) and b in a) or \
                        (isinstance(b, TypeOperator) and a in b):
                    raise error.RecursiveType(a, b)
                elif isinstance(a, TypeVar) and a != b:
                    a.bind(b)
                elif isinstance(b, TypeVar) and a != b:
                    b.bind(a)

    def resolve(self, full: bool = True) -> TypeTerm:
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
                self.variance,
                *(t.resolve(full) for t in self.params),
                supertype=self.supertype)
        return self

    def compatible(
            self,
            other: TypeTerm,
            allow_subtype: bool = False) -> bool:
        """
        Is the type structurally consistent with another, that is, do they
        'fit', modulo variables. Subtypes may be allowed on the self side.
        """
        if isinstance(self, TypeOperator) and isinstance(other, TypeOperator):
            return self.match(other, allow_subtype) and all(
                s.compatible(t, allow_subtype)
                for s, t in zip(self.params, other.params))
        return True

    def subtypes(self, other: TypeTerm) -> bool:
        """
        Is this a subtype of another type?
        """
        if isinstance(self, TypeOperator) and isinstance(other, TypeOperator):
            if self.name == 'function' and other.name == 'function':
                return (
                    self.params[0].subtypes(other.params[0]) and
                    other.params[1].subtypes(self.params[1])
                )
            return (
                (
                    (self.name == other.name and self.arity == other.arity) or
                    (bool(self.supertype and self.supertype.subtypes(other)))
                ) and
                all(s.subtypes(t) for s, t in zip(self.params, other.params))
            )
        return False

    # constraints #############################################################

    def subtype(self, *patterns: TypeTerm) -> Constraint:
        return NoConstraint(self, *patterns, allow_subtype=True)

    def member(self, *patterns: TypeTerm) -> Constraint:
        return NoConstraint(self, *patterns, allow_subtype=False)

    def param(
            self, target: TypeTerm,
            subtype: bool = False,
            at: Optional[int] = None) -> Constraint:
        return NoConstraint(
            self, target) #  , position=at, allow_subtype=subtype)


class TypeOperator(TypeTerm):
    """
    n-ary type constructor.
    """

    def __init__(
            self,
            name: str,
            variance: Iterable[Variance] = (),
            *params: TypeTerm,
            supertype: Optional[TypeOperator] = None):
        self.name = name
        self.supertype = supertype
        self.variance = list(variance)
        self.params: List[TypeTerm] = list(params)

        if len(self.params) != len(self.variance):
            raise ValueError("every parameter must have a variance")
        if self.name == 'function' and self.arity != 2:
            raise ValueError("functions must have 2 argument types")
        if self.supertype and (self.params or self.supertype.params):
            raise ValueError("only nullary types may have supertypes")

    def __str__(self) -> str:
        if self.name == 'function':
            inT, outT = self.params
            if isinstance(inT, TypeOperator) and inT.name == 'function':
                return f"({inT}) ** {outT}"
            return f"{inT} ** {outT}"
        elif self.params:
            return f'{self.name}({", ".join(str(t) for t in self.params)})'
        else:
            return self.name

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TypeOperator):
            return self.match(other, allow_subtype=False) and \
               all(s == t for s, t in zip(self.params, other.params))
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
        return len(self.params)

    @staticmethod
    def parameterized(
            name: str,
            *variance: Variance) -> Callable[..., TypeOperator]:
        """
        Allowing us to define parameterized types in an intuitive way, while
        optionally fixing the arity of the operator.
        """
        def f(*params):
            if len(params) != len(variance):
                raise TypeError(
                    f"type operator {name} has arity {len(variance)}, "
                    f"but was given {len(params)} parameter(s)")
            return TypeOperator(name, variance, *params)
        return f


class TypeVar(TypeTerm):
    """
    Type variable. Note that any bindings and constraints are bound to the
    actual object instance, so make a fresh copy before applying them if the
    variable is not supposed to be a concrete instance.
    """

    counter = 0

    def __init__(self, wildcard: bool = False):
        cls = type(self)
        self.id = cls.counter
        self.bound: Optional[TypeTerm] = None
        self.wildcard = wildcard
        cls.counter += 1

    def __str__(self) -> str:
        return "_" if self.wildcard else f"x{self.id}"

    def bind(self, binding: TypeTerm) -> None:
        assert (not self.bound or binding == self.bound), \
            "variable cannot be bound twice"

        self.bound = binding


class Constraint(ABC):
    """
    A constraint enforces that its subject type always remains consistent with
    whatever condition it represents.
    """

    def __init__(
            self,
            type: TypeTerm,
            *patterns: TypeTerm,
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

    def compatible(self, pattern: TypeTerm) -> bool:
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
