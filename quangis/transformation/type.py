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

from enum import Enum, auto
from abc import ABC, abstractmethod
from functools import reduce
from itertools import chain, accumulate
from inspect import signature, Signature, Parameter
from typing import Optional, Iterable, Union, Callable, Tuple, List

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

    COVARIANT = auto()
    CONTRAVARIANT = auto()


class Type(ABC):
    """
    The base class for anything that can be treated as a (schematic) type.
    """

    @abstractmethod
    def instance(self, *arg: VariableTerm, **kwargs: VariableTerm) -> Term:
        return NotImplemented

    def __pow__(self, other: Type) -> Type:
        """
        Function abstraction. This is an overloaded (ab)use of Python's
        exponentiation operator. It allows us to use the infix operator ** for
        the arrow in function signatures.

        Note that this operator is one of the few that is right-to-left
        associative, matching the conventional behaviour of the function arrow.
        The right-bitshift operator >> (for __rshift__) would have been more
        intuitive visually, but does not have this property.
        """

        return Type.combine(self, other, by=lambda a, b:
            Term(Function(a.plain, b.plain), *(a.constraints + b.constraints))
        )

    def __call__(self, *args: Union[Tuple[Type, int], Type]) -> Type:
        """
        Function application. This allows us to apply two types to eachother by
        calling the function type with its argument type.
        """
        # This is only to accept tuples, because sometimes we define a
        # signature with a tuple. Now we don't have to think about something
        # like pi1(objectregions[0]). Might become unnecessary.
        args1 = (t[0] if isinstance(t, tuple) else t for t in args)

        return Type.combine(self, *args1, by=lambda x, *xs:
            reduce(Term.apply, xs, x)
        )

    def __or__(self, constraint: Optional[Constraint]) -> Type:
        """
        Another abuse of Python's operators, allowing us to add constraints by
        using the | operator.
        """
        if not constraint:
            return self

        if isinstance(self, Schema):
            def σ(*args, **kwargs):
                t = self.instance(*args, **kwargs)
                return Term(t.plain, constraint, *t.constraints)
            σ.__signature__ = self.signature  # type: ignore
            return Schema(σ)
        else:
            t = self.instance()
            return Term(t.plain, constraint, *t.constraints)

    def __lshift__(self, other: Union[Type, Iterable[Type]]
            ) -> Optional[Constraint]:
        """
        Allows us to write subtype relations and constraints using <<.
        """
        if isinstance(other, Type):
            self.instance().plain.unify_subtype(other.instance().plain)
            return None
        else:
            return Constraint(self.instance(), *(o.instance() for o in other))

    def parameter_of(
            self,
            *ops: Operator,
            at: Optional[int] = None) -> List[PlainTerm]:
        """
        Generate a list of instances operators that contain this type somewhere
        in its parameters.
        """
        target = self.instance().plain
        options: List[PlainTerm] = []
        for op in ops:
            for i in range(op.arity) if at is None else (at - 1,):
                if i < op.arity:
                    options.append(op(*(
                        target if i == j else VariableTerm(wildcard=True)
                        for j in range(op.arity)
                    )))
        return options

    @staticmethod
    def combine(*types: Type, by: Callable[..., Term]) -> Type:
        """
        Combine several types into a single (possibly schematic) type using a
        function that combines instances of those types into a single term.
        """

        if any(isinstance(t, Schema) for t in types):
            # A new schematic variable for every such one needed by arguments
            n_vars = [t.n_vars if isinstance(t, Schema) else 0 for t in types]
            names = list(VariableTerm.names(sum(n_vars)))
            params = [
                Parameter(v, Parameter.POSITIONAL_OR_KEYWORD) for v in names]
            sig = Signature(params)

            # Divvy up the new parameters for all the argument types
            types_with_varnames = [
                (t, names[i:i + δ])
                for t, i, δ in zip(types, accumulate([0] + n_vars), n_vars)
            ]

            # Combine into a new schema
            def σ(*args: VariableTerm, **kwargs: VariableTerm) -> Term:
                binding = sig.bind(*args, **kwargs)
                return by(*(
                    t.instance(*(binding.arguments[v] for v in varnames))
                    for t, varnames in types_with_varnames
                ))
            σ.__signature__ = (  # type: ignore
                signature(σ).replace(parameters=params))

            return Schema(σ)
        else:
            return by(*(t.instance() for t in types))


class Schema(Type):
    """
    Provides a definition of a *schema* for function and data signatures, that
    is, a type containing some schematic type variable.
    """

    def __init__(self, schema: Callable[..., Type]):
        self.schema = schema
        self.signature = signature(schema)
        self.n_vars = len(self.signature.parameters)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return str(self.instance(*(
            VariableTerm(v) for v in self.signature.parameters)).resolve())

    def instance(self, *args: VariableTerm, **kwargs: VariableTerm) -> Term:
        """
        Create an instance of this schema. Optionally bind schematic variables
        to concrete variables; non-bound variables will get automatically
        assigned a concrete variable.
        """
        binding = self.signature.bind_partial(*args, **kwargs)
        for param in self.signature.parameters:
            if param not in binding.arguments:
                binding.arguments[param] = VariableTerm()
        return self.schema(*binding.args, **binding.kwargs).instance()


class Term(Type):
    """
    A top-level type term decorated with constraints.
    """

    def __init__(self, plain: PlainTerm, *constraints: Constraint):
        self.plain = plain
        self.constraints = []

        for c in constraints:
            self.constraints.append(c)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        res = [str(self.plain)]

        for c in self.constraints:
            res.append(str(c))

        for v in set(self.plain.variables()):
            if v.lower:
                res.append(f"{v.lower} << {v}")
            if v.upper:
                res.append(f"{v} << {v.upper}")

        return ' | '.join(res)

    def instance(self, *args, **kwargs) -> Term:
        Signature().bind(*args, **kwargs)
        return self

    def resolve(self, force: bool = False) -> Term:
        constraints = [c for c in self.constraints if not c.fulfilled()]
        t = self.plain.resolve(force=force)
        return Term(t, *constraints)

    def apply(self, arg: Term) -> Term:
        """
        Apply an argument to a function type to get its output type.
        """

        f = self.plain.follow()
        x = arg.plain.follow()

        if isinstance(f, VariableTerm):
            f.bind(Function(VariableTerm(), VariableTerm()))
            f = f.follow()

        if isinstance(f, OperatorTerm) and f.operator == Function:
            x.unify_subtype(f.params[0])
            return Term(
                f.params[1],
                *chain(self.constraints, arg.constraints)
            ).resolve()
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

    def variables(self) -> Iterable[VariableTerm]:
        """
        Obtain all type variables currently in the type expression.
        """
        a = self.follow()
        if isinstance(a, VariableTerm):
            yield a
        elif isinstance(self, OperatorTerm):
            for v in chain(*(t.variables() for t in self.params)):
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
        Return True if self is definitely a subtype of other, False if it is
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
                    elif v == Variance.CONTRAVARIANT:
                        r = t.subtype(s)
                    else:
                        raise ValueError

                    if r is None:
                        return None
                    else:
                        result &= r
                return result
        else:
            if (isinstance(a, VariableTerm) and a.wildcard) or \
                    (isinstance(b, VariableTerm) and b.wildcard):
                return True
            # TODO if the upper limit of the subtype is a subtype of the lower
            # limit of the supertype...
            pass
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
                        raise ValueError
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
                a.unify_subtype(b)

        elif isinstance(a, OperatorTerm) and isinstance(b, VariableTerm):
            if b in a:
                raise error.RecursiveType(b, a)
            elif a.operator.basic:
                b.above(a.operator)
            else:
                b.bind(a.skeleton())
                b.unify_subtype(a)

    def resolve(
            self,
            force: bool = False,
            prefer_lower: bool = True) -> PlainTerm:
        """
        Obtain a version of this type with all unified variables substituted
        and optionally all subtypes resolved to their most specific type.

        If `force`d, then variables that should be unified with their lower
        bounds may also be unified with their upper bounds and vice versa. This
        is not technically sound, but it can help to quickly avoid variables.
        Note that if the subtype structure is a lattice, this can be avoided by
        unifying with the supremum or infimum.
        """
        a = self.follow()

        if isinstance(a, OperatorTerm):
            return OperatorTerm(
                a.operator,
                *(p.resolve(
                    prefer_lower=prefer_lower ^ (v == Variance.CONTRAVARIANT),
                    force=force)
                    for v, p in zip(a.operator.variance, a.params))
            )
        elif isinstance(a, VariableTerm):
            if prefer_lower and a.lower:
                a.bind(a.lower())
            elif not prefer_lower and a.upper:
                a.bind(a.upper())
            elif force:
                if a.upper:
                    a.bind(a.upper())
                elif a.lower:
                    a.bind(a.lower())
            return a.follow()
        raise ValueError

    def instance(self, *args, **kwargs) -> Term:
        Signature().bind(*args, **kwargs)
        return Term(self)


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

    def __call__(self, *params) -> OperatorTerm:  # type: ignore
        return OperatorTerm(self, *params)

    def instance(self, *args, **kwargs) -> Term:
        Signature().bind(*args, **kwargs)
        return Term(OperatorTerm(self))

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

    def __init__(self, op: Operator, *params: Union[PlainTerm, Operator]):
        self.operator = op
        self.params = [p() if isinstance(p, Operator) else p for p in params]

        if len(self.params) != self.operator.arity:
            raise ValueError(
                f"{self.operator} takes {self.operator.arity} "
                f"parameter{'' if self.operator.arity == 1 else 's'}; "
                f"{len(self.params)} given"
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
                all(s.follow() == t.follow()
                    for s, t in zip(self.params, other.params))
        else:
            return False


class VariableTerm(PlainTerm):
    """
    Term variable.
    """
    counter = 0

    def __init__(self, name: Optional[str] = None, wildcard: bool = False):
        cls = type(self)
        self.id = cls.counter
        self.name = name
        self.wildcard = wildcard
        self.lower: Optional[Operator] = None
        self.unified: Optional[PlainTerm] = None
        self.upper: Optional[Operator] = None
        cls.counter += 1

    def __str__(self) -> str:
        return "_" if self.wildcard else self.name or f"_{self.id}"

    def bind(self, t: PlainTerm) -> None:
        assert (not self.unified or t == self.unified), \
            "variable cannot be unified twice"

        self.wildcard = False

        if self is not t:
            self.unified = t

            if isinstance(t, VariableTerm):
                t.wildcard = False

                if self.lower:
                    t.above(self.lower)
                if self.upper:
                    t.below(self.upper)

                if t.lower == t.upper and t.lower is not None:
                    t.bind(t.lower())

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
        # symmetric to `above`
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
    def names(n: int, unicode: bool = False) -> Iterable[str]:
        """
        Produce some suitable variable names.
        """
        base = "τσαβγφψ" if unicode else "xyzuvw"
        for i in range(n):
            yield base[i] if n < len(base) else base[0] + str(i + 1)


"The special constructor for function types."
Function = Operator(
    'Function',
    params=(Variance.CONTRAVARIANT, Variance.COVARIANT)
)

"A wildcard: produces an unrelated variable, to be matched with anything."
_ = Schema(lambda: VariableTerm(wildcard=True))


class Constraint(object):
    """
    A constraint enforces that its subject type always remains consistent with
    whatever condition it represents.
    """

    def __init__(self, subject: Type, *patterns: Type):
        self.subject = subject.instance().plain
        self.patterns = [t.instance().plain for t in patterns]

    def __str__(self) -> str:
        return (
            f"{self.subject.resolve()} << "
            f"{[p.resolve() for p in self.patterns]}"
        )

    def fulfilled(self) -> bool:
        """
        Check that the constraint has not been violated and raise an error
        otherwise. Additionally, return True if it has been completely
        fulfilled and need not be enforced any longer.
        """

        statuses = [(p, self.subject.subtype(p)) for p in self.patterns]
        patterns = [(p, s) for p, s in statuses if s is not False]
        self.patterns = [p[0] for p in patterns]
        if len(patterns) == 0:
            raise error.ViolatedConstraint(self)
        elif len(patterns) == 1:  # and patterns[0][0] is True:
            self.subject.unify_subtype(patterns[0][0])
            return True
        return False

