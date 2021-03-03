"""
Generic type system. Inspired loosely by Hindley-Milner type inference in
functional programming languages.
"""
# A primer: A type consists of type operators and type variables. Term
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
# PlainTerm class.
from __future__ import annotations

from enum import Enum
from abc import ABC, abstractmethod
from functools import reduce
from itertools import chain
from inspect import signature
from typing import Optional, Iterable, Union, List, Callable

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


class Type(ABC):
    """
    The base class for anything that can be treated as a (schematic) type.
    """

    @abstractmethod
    def instance(self) -> Term:
        return NotImplemented


class Schema(Type):
    """
    This class provides the definition of a *schema* for function and data
    signatures: it knows its schematic type, and can generate fresh instances.
    """

    def __init__(self, schema: Union[Type, Callable[..., Type]]):
        self.schema = schema
        if callable(schema):
            self.variables = len(signature(schema).parameters)
        else:
            self.variables = 0
            self.schema = schema.instance()

    def instance(self) -> Term:
        if isinstance(self.schema, Term):
            return self.schema
        elif callable(self.schema):
            return self.schema(
                *(VariableTerm() for _ in range(self.variables))
            ).instance()
        raise ValueError

    def __call__(self, *args: Type) -> Term:
        return self.instance().__call__(*args)


class Term(Type):
    """
    A top-level type term decorated with constraints.
    """

    def __init__(self, plain: PlainTerm, constraints: Iterable[Constraint] = ()):
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

    def __call__(self, *args: Type) -> Term:
        return reduce(Term.apply, (a.instance() for a in args), self)

    def instance(self) -> Term:
        return self

    def resolve(self) -> Term:
        return Term(
            self.plain.resolve(),
            (c for c in self.constraints if c.enforce())
        )

    def apply(self, arg: Term) -> Term:
        """
        Apply an argument to a function type to get its output type.
        """

        f = self.plain.follow()
        x = arg.plain.follow()

        if isinstance(f, VariableTerm):
            f.unify(VariableTerm() ** VariableTerm())
            f = f.follow()

        if isinstance(f, OperatorTerm) and f.operator == Function:
            x.unify_subtype(f.params[0])
            return Term(
                f.params[1],
                (c for c in chain(self.constraints, arg.constraints)
                    if c.enforce())
            )
        else:
            raise error.NonFunctionApplication(f, x)


class PlainTerm(Type):
    """
    Abstract base class for plain type terms (operator terms and type
    variables) without constraints. Note that basic types are just 0-ary type
    operators and functions are just particular 2-ary type operators.
    """

    def __repr__(self):
        return self.__str__()

    def __contains__(self, value: PlainTerm) -> bool:
        return value == self or (
            isinstance(self, OperatorTerm) and
            any(value in t for t in self.params))

    def __pow__(self, other: Union[PlainTerm, Operator]) -> OperatorTerm:
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

    def __or__(self, other: Union[Constraint, Iterable[Constraint]]) -> Term:
        """
        Another abuse of Python's operators, allowing us to add constraints
        to plain types using the | operator.
        """
        return Term(
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
            a = self.follow()
            if isinstance(a, VariableTerm):
                yield a
            else:
                for v in a.variables():
                    yield v

    def follow(self) -> PlainTerm:
        """
        Follow a unification until the nearest operator.
        """
        if isinstance(self, VariableTerm) and self.unified:
            return self.unified.follow()
        return self

    def skeleton(self) -> PlainTerm:
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

    def subtype(self, other: PlainTerm) -> Optional[bool]:
        """
        Return true if self is definitely a subtype of other, False if it is
        definitely not, and None if there is not enough information.
        """
        a = self.follow()
        b = other.follow()

        if isinstance(a, OperatorTerm) and isinstance(b, OperatorTerm):
            if a.operator.basic:
                return a.operator <= b.operator
            elif a.operator != b.operator:
                return False
            else:
                result = True
                for v, s, t in zip(a.operator.variance, a.params, b.params):
                    if v == Variance.COVARIANT:
                        r = s.subtype(t)
                    else:
                        r = t.subtype(s)

                    if r is None:
                        return None
                    else:
                        result &= r
                return result
        return None

    def unify_subtype(self, other: PlainTerm) -> None:
        """
        Make sure that a is equal to, or a subtype of b. Like normal
        unification, but instead of just a substitution of variables to terms,
        also produces lower and upper bounds on subtypes that it must respect.

        Resulting constraints are a side-effect; use resolve() to consolidate
        equality.
        """
        a = self.follow()
        b = other.follow()

        if isinstance(a, OperatorTerm) and isinstance(b, OperatorTerm):
            if a.operator.basic:
                if not (a.operator <= b.operator):
                    raise error.SubtypeMismatch(a, b)
            elif a.operator == b.operator:
                for v, x, y in zip(a.operator.variance, a.params, b.params):
                    if v == Variance.COVARIANT:
                        x.unify_subtype(y)
                    elif v == Variance.CONTRAVARIANT:
                        y.unify_subtype(x)
            else:
                raise error.TypeMismatch(a, b)

        elif isinstance(a, VariableTerm) and isinstance(b, VariableTerm):
            a.unify(b)

        elif isinstance(a, VariableTerm) and isinstance(b, OperatorTerm):
            if a in b:
                raise error.RecursiveType(a, b)
            elif b.operator.basic:
                a.below(b.operator)
            else:
                a.unify(b.skeleton())
                a.unify_subtype(b)

        elif isinstance(a, OperatorTerm) and isinstance(b, VariableTerm):
            if b in a:
                raise error.RecursiveType(b, a)
            elif a.operator.basic:
                b.above(a.operator)
            else:
                b.unify(a.skeleton())
                b.unify_subtype(a)

    def resolve(self, prefer_lower: bool = True) -> PlainTerm:
        """
        Obtain a version of this type with all unified variables substituted
        and all subtypes resolved to their most specific type.
        """
        a = self.follow()

        if isinstance(a, OperatorTerm):
            return OperatorTerm(
                a.operator,
                *(p.resolve(prefer_lower ^ (v == Variance.CONTRAVARIANT))
                    for v, p in zip(a.operator.variance, a.params))
            )
        elif isinstance(a, VariableTerm):
            if prefer_lower and a.lower:
                a.unify(a.lower())
            elif not prefer_lower and a.upper:
                a.unify(a.upper())
            elif a.lower:  # TODO not sure it makes sense to accept lower
                a.unify(a.lower())  # bounds when we want upper bounds. need
            elif a.upper:  # for a lattice subtyping structure?
                a.unify(a.upper())
            return a.follow()
        raise ValueError

    def instance(self) -> Term:
        return Term(self)

    def __call__(self, *args: Type) -> Term:
        return Term(self).__call__(*args)


class Operator(Type):
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

    def __pow__(self, other: Union[PlainTerm, Operator]) -> OperatorTerm:
        return self().__pow__(other)

    def instance(self) -> Term:
        return Term(self())

    @property
    def basic(self) -> bool:
        return self.arity == 0

    @property
    def compound(self) -> bool:
        return not self.basic


class OperatorTerm(PlainTerm):
    """
    An instance of an n-ary type constructor.
    """

    def __init__(
            self,
            operator: Operator,
            *params: Union[PlainTerm, Operator]):
        self.operator = operator
        self.params: List[PlainTerm] = list(
            p() if isinstance(p, Operator) else p for p in params)

        if len(self.params) != self.operator.arity:
            raise ValueError(
                f"{self.operator} takes {self.operator.arity} "
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


class VariableTerm(PlainTerm):
    """
    Term variable.
    """
    counter = 0

    def __init__(self):
        cls = type(self)
        self.id = cls.counter
        self.lower: Optional[Operator] = None
        self.unified: Optional[PlainTerm] = None
        self.upper: Optional[Operator] = None
        cls.counter += 1

    def __str__(self) -> str:
        return f"x{self.id}"

    def unify(self, t: PlainTerm) -> None:
        assert (not self.unified or t == self.unified), \
            "variable cannot be unified twice"

        if self is not t:
            self.unified = t

            if isinstance(t, VariableTerm):
                if self.lower:
                    t.above(self.lower)
                if self.upper:
                    t.below(self.upper)
                if t.lower:
                    self.above(t.lower)
                if t.upper:
                    self.below(t.upper)

            elif isinstance(t, OperatorTerm) and t.operator.basic:
                if self.lower is not None and t.operator < self.lower:
                    raise error.SubtypeMismatch(t, self)
                if self.upper is not None and self.upper < t.operator:
                    raise error.SubtypeMismatch(self, t)

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

    @staticmethod
    def names(number: int, ascii: bool = False) -> Iterable[str]:
        """
        Produce some suitable variable names.
        """
        base = "xyzvuw" if ascii else "τσαβγφψ"
        for i in range(number):
            if number < len(base):
                yield base[i]
            else:
                j = str(i + 1)
                if not ascii:
                    j = "".join(chr(ord("₀") - ord("0") + ord(d)) for d in j)
                yield base[0] + j


"The special constructor for function types."
Function = Operator(
    'Function',
    params=(Variance.CONTRAVARIANT, Variance.COVARIANT)
)


class Constraint(ABC):
    """
    A constraint enforces that its subject type always remains consistent with
    whatever condition it represents.
    """

    def __init__(self, *patterns: Union[PlainTerm, Operator], **kwargs):
        self.patterns = list(
            p() if isinstance(p, Operator) else p for p in patterns
        )
        self.kwargs = kwargs

    def __str__(self) -> str:
        return (
            f"{type(self).__name__}"
            f"({', '.join(str(p) for p in self.patterns)}, "
            f"{', '.join(f'{k}={v}' for k, v in self.kwargs.items())})"
        )

    @abstractmethod
    def enforce(self) -> bool:
        """
        Check that the resolved constraint has not been violated. Return False
        if it has also been completely fulfilled and need not be enforced any
        longer.
        """
        raise NotImplementedError


class Subtype(Constraint):
    """
    Check that its first pattern is a subtype of at least one of its other
    patterns.
    """

    def enforce(self) -> bool:
        subject = self.patterns[0]
        for other in self.patterns[1:]:
            status = subject.subtype(other)
            if status is True:
                return False
            elif status is None:
                return True
        raise error.ViolatedConstraint(self)


Member = Subtype


class Param(Constraint):
    """
    Check that its first pattern is a compound type with one of the given types
    occurring somewhere in its parameters.
    """

    def enforce(self) -> bool:
        subject = self.patterns[0]
        position = self.kwargs.get('at')
        if isinstance(subject, OperatorTerm):
            if position is None:
                for p in subject.params:
                    for other in self.patterns[1:]:
                        status = p.subtype(other)
                        if status is True:
                            return False
                        elif status is None:
                            return True
            elif position - 1 < len(subject.params):
                p = subject.params[position - 1]
                for other in self.patterns[1:]:
                    status = p.subtype(other)
                    if status is True:
                        return False
                    elif status is None:
                        return True
            raise error.ViolatedConstraint(self)
        return True



