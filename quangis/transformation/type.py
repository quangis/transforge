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
# Term class.
from __future__ import annotations

from enum import Enum
from abc import ABC, abstractmethod
from functools import reduce
from itertools import chain
from inspect import signature
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


class Signature(object):
    """
    This class provides the definition of a *signature* for a function or for
    data: it knows its schematic type, and can generate fresh instances.
    """

    def __init__(self, schema: TypeSchema, data_args: int = 0):
        self.type = schema
        self.data_args = data_args

    def instance(self) -> Type:
        """
        Get an instance of the type represented by this function schema.
        """
        return instance(self.type)

    def __call__(self, *args: TypeSchema) -> Type:
        return self.instance().__call__(*args)


class Type(object):
    """
    A top-level type term decorated with constraints.
    """

    def __init__(self, plain: Term, constraints: Iterable[Constraint] = ()):
        self.plain = plain
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

    def __call__(self, *args: TypeSchema) -> Type:
        return reduce(Type.apply, (instance(a) for a in args), self)

    def apply(self, arg: Type) -> Type:
        """
        Apply an argument to a function type to get its resolved output type.
        """

        f: Term = self.plain
        if isinstance(self.plain, VariableTerm):
            f = VariableTerm() ** VariableTerm()
            self.plain.bind(f)

        if isinstance(f, OperatorTerm) and f.operator == Function:
            input_type, output_type = f.params
            arg.plain.unify(input_type)

            return Type(
                output_type.resolve(),
                (constraint
                    for constraint in chain(self.constraints, arg.constraints)
                    if not constraint.fulfilled())
            )
        else:
            raise error.NonFunctionApplication(self.plain, arg)


class Term(ABC):
    """
    Abstract base class for plain type terms (operator terms and type
    variables) without constraints. Note that basic types are just 0-ary type
    operators and functions are just particular 2-ary type operators.
    """

    def __repr__(self):
        return self.__str__()

    def __contains__(self, value: Term) -> bool:
        return value == self or (
            isinstance(self, OperatorTerm) and
            any(value in t for t in self.params))

    def __pow__(self, other: Union[Term, Operator]) -> OperatorTerm:
        """
        This is an overloaded (ab)use of Python's exponentiation operator. It
        allows us to use the infix operator ** for the arrow in function
        signatures.

        Note that this operator is one of the few that is right-to-left
        associative, matching the conventional behaviour of the function arrow.
        The right-bitshift operator >> (for __rshift__) would have been more
        intuitive visually, but does not have this property.
        """
        return Function(
            self,
            other() if isinstance(other, Operator) else other)

    def __or__(self, other: Union[Constraint, Iterable[Constraint]]) -> Type:
        """
        Another abuse of Python's operators, allowing for us to add constraints
        to plain types using the | operator.
        """
        return Type(
            self,
            (other,) if isinstance(other, Constraint) else other)

    def variables(self) -> Iterable[VariableTerm]:
        """
        Obtain all type variables currently in the type expression.
        """
        if isinstance(self, OperatorTerm):
            for v in chain(*(t.variables() for t in self.params)):
                yield v
        else:
            a = self.resolve(full=False)
            if isinstance(a, VariableTerm):
                yield a
            else:
                for v in a.variables():
                    yield v

    def resolve(self, full: bool = True) -> Term:
        """
        A `full` resolve obtains a version of this type with all bound
        variables replaced with their bindings. Otherwise, just resolve the
        current variable.
        """
        if isinstance(self, VariableTerm) and self.bound:
            return self.bound.resolve(full)
        elif full and isinstance(self, OperatorTerm):
            return OperatorTerm(
                self.operator,
                *(t.resolve(full) for t in self.params))
        return self

    def specify(self, prefer_lower: bool = True) -> Term:
        """
        Consolidate all variables with subtype constraints into the most
        specific basic type possible.
        """
        if isinstance(self, OperatorTerm):
            return OperatorTerm(
                self.operator,
                *(p.specify(prefer_lower ^ (v == Variance.CONTRAVARIANT))
                    for v, p in zip(self.operator.variance, self.params))
            )
        elif isinstance(self, VariableTerm):
            if prefer_lower and self.lower:
                return OperatorTerm(self.lower)
            elif not prefer_lower and self.upper:
                return OperatorTerm(self.upper)
            elif self.lower is not None:  # TODO not sure it makes sense to
                return OperatorTerm(self.lower)  # accept lower bounds when
            elif self.upper is not None:  # we want upper bounds. need for
                return OperatorTerm(self.upper)  # lattice subtype structure?
            else:
                return self
        raise ValueError

    def compatible(
            self,
            other: Term,
            allow_subtype: bool = False) -> bool:
        """
        Is the type structurally consistent with another, that is, do they
        'fit', modulo variables. Subtypes may be allowed on the self side.
        """
        if isinstance(self, OperatorTerm) and isinstance(other, OperatorTerm):
            return (
                self.operator <= other.operator and
                all(s.compatible(t, allow_subtype)
                    for s, t in zip(self.params, other.params))
            )
        return True

    def skeleton(self) -> Term:
        """
        Create a copy of this operator, substituting fresh variables for basic
        types.
        """
        if isinstance(self, OperatorTerm):
            if self.operator.basic:
                return VariableTerm()
            else:
                return OperatorTerm(
                    self.operator,
                    *(p.skeleton() for p in self.params))
        else:
            return self

    def unify(self, other: Term) -> None:
        """
        Make sure that a is equal to, or a subtype of b. Like normal
        unification, but instead of just a substitution of variables to terms,
        also produces lower and upper bounds on subtypes that it must respect.
        """
        a = self.resolve(False)
        b = other.resolve(False)

        if isinstance(a, OperatorTerm) and isinstance(b, OperatorTerm):
            if a.operator.basic:
                if not (a.operator <= b.operator):
                    raise error.SubtypeMismatch(a, b)
            elif a.operator == b.operator:
                for v, x, y in zip(a.operator.variance, a.params, b.params):
                    if v == Variance.COVARIANT:
                        x.unify(y)
                    elif v == Variance.CONTRAVARIANT:
                        y.unify(x)
            else:
                raise error.TypeMismatch(a, b)

        elif isinstance(a, VariableTerm) and isinstance(b, VariableTerm):
            a.bind(b)

        elif isinstance(a, VariableTerm) and isinstance(b, OperatorTerm):
            if a in b:
                raise error.RecursiveType(a, b)
            elif b.operator.basic:
                a.below(b.operator)
            else:
                a.bind(b.skeleton())
                a.unify(b)

        elif isinstance(a, OperatorTerm) and isinstance(b, VariableTerm):
            if b in a:
                raise error.RecursiveType(b, a)
            elif a.operator.basic:
                b.above(a.operator)
            else:
                b.bind(a.skeleton())
                b.unify(a)

    def __call__(self, *args: TypeSchema) -> Type:
        return Type(self).__call__(*args)

    # constraints #############################################################

    def subtype(self, *patterns: Term) -> Constraint:
        return NoConstraint(self, *patterns, allow_subtype=True)

    def member(self, *patterns: Term) -> Constraint:
        return NoConstraint(self, *patterns, allow_subtype=False)

    def param(
            self, target: Term,
            subtype: bool = False,
            at: Optional[int] = None) -> Constraint:
        return NoConstraint(
            self, target)  # , position=at, allow_subtype=subtype)


class Operator(object):
    """
    An n-ary type constructor.
    """

    def __init__(
            self,
            name: str,
            params: Union[int, Iterable[Variance]] = 0,
            supertype: Optional[Operator] = None):
        self.name = name
        self.supertype: Optional[Operator] = supertype

        if isinstance(params, int):
            self.variance = list(Variance.COVARIANT for _ in range(params))
        else:
            self.variance = list(params)
        self.arity = len(self.variance)

        if self.supertype and not self.basic:
            raise ValueError("only nullary types can have direct supertypes")

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Operator) and
            self.name == other.name
            and self.variance == other.variance)

    def __le__(self, other: Operator) -> bool:
        return self == other or self < other

    def __lt__(self, other: Operator) -> bool:
        return bool(self.supertype and self.supertype <= other)

    def __call__(self, *params) -> OperatorTerm:
        return OperatorTerm(self, *params)

    def __pow__(self, other: Union[Term, Operator]) -> OperatorTerm:
        return self().__pow__(other)

    @property
    def basic(self) -> bool:
        return self.arity == 0

    @property
    def compound(self) -> bool:
        return not self.basic


class OperatorTerm(Term):
    """
    An instance of an n-ary type constructor.
    """

    def __init__(
            self,
            operator: Operator,
            *params: Union[Term, Operator]):
        self.operator = operator
        self.params: List[Term] = list(
            p() if isinstance(p, Operator) else p for p in params)

        if len(self.params) != self.operator.arity:
            raise ValueError(
                f"{self.operator} takes {self.operator.arity}"
                f"parameters; {len(self.params)} given"
            )

    def __str__(self) -> str:
        if self.operator == Function:
            inT, outT = self.params
            if isinstance(inT, OperatorTerm) and inT.operator == Function:
                return f"({inT}) ** {outT}"
            return f"{inT} ** {outT}"
        elif self.params:
            return f'{self.operator}({", ".join(str(t) for t in self.params)})'
        else:
            return str(self.operator)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, OperatorTerm):
            return self.operator == other.operator and \
               all(s == t for s, t in zip(self.params, other.params))
        else:
            return False


class VariableTerm(Term):
    """
    Type variable.
    """
    counter = 0

    def __init__(self):
        cls = type(self)
        self.id = cls.counter
        self.bound: Optional[Term] = None
        self.lower: Optional[Operator] = None
        self.upper: Optional[Operator] = None
        cls.counter += 1

    def __str__(self) -> str:
        return f"x{self.id}"

    def bind(self, binding: Term) -> None:
        assert (not self.bound or binding == self.bound), \
            "variable cannot be bound twice"

        if binding is not self:
            self.bound = binding

            if isinstance(binding, VariableTerm):
                if self.lower:
                    binding.above(self.lower)
                if self.upper:
                    binding.below(self.upper)
                if binding.lower:
                    self.above(binding.lower)
                if binding.upper:
                    self.below(binding.upper)

    def above(self, new: Operator) -> None:
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

    def below(self, new: Operator) -> None:
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
Function = Operator(
    'Function',
    params=(Variance.CONTRAVARIANT, Variance.COVARIANT)
)

"A shortcut for writing function signatures."
Σ = Signature

"A union type for anything that can act as a schematic type."
TypeSchema = Union[
    Type,
    Term,
    Operator,
    Signature,
    Callable[..., Type],
    Callable[..., Term]]


def instance(schema: TypeSchema) -> Type:
    """
    Turn anything that can act as a type schema into an instance of that type.
    """
    if isinstance(schema, Term):
        return Type(schema)
    elif isinstance(schema, Type):
        return schema
    elif isinstance(schema, Signature):
        return schema.instance()
    elif isinstance(schema, Operator):
        return instance(schema())
    elif callable(schema):
        n = len(signature(schema).parameters)
        return instance(schema(*(VariableTerm() for _ in range(n))))
    else:
        raise ValueError(f"Cannot convert a {type(schema)} to a Type")


class Constraint(ABC):
    """
    A constraint enforces that its subject type always remains consistent with
    whatever condition it represents.
    """

    def __init__(
            self,
            type: Term,
            *patterns: Term,
            allow_subtype: bool = False):
        self.subject = type
        self.patterns = list(patterns)
        self.allow_subtype = allow_subtype

    def variables(self) -> Iterable[VariableTerm]:
        return chain(
            self.subject.variables(),
            *(t.variables() for t in self.patterns))

    def resolve(self, full: bool = True) -> Constraint:
        self.subject = self.subject.resolve(full)
        self.patterns = [t.resolve(full) for t in self.patterns]
        self.enforce()
        return self

    @abstractmethod
    def fresh(self, ctx: Dict[VariableTerm, VariableTerm]):
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

    def fresh(self, ctx: Dict[VariableTerm, VariableTerm]) -> MembershipConstraint:
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

    def fresh(self, ctx: Dict[VariableTerm, VariableTerm]) -> ParameterConstraint:
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

    def compatible(self, pattern: Term) -> bool:
        if isinstance(self.subject, OperatorTerm):
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
