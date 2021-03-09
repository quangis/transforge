"""
Generic type system. Inspired loosely by Hindley-Milner type inference in
functional programming languages, as well as Traytel et al (2011).
"""
from __future__ import annotations

from enum import Enum, auto
from abc import ABC, abstractmethod
from functools import reduce
from itertools import chain, accumulate
from inspect import signature, Signature, Parameter
from typing import Optional, Iterable, Union, Callable, List

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

    CO = auto()
    CONTRA = auto()


class Type(ABC):
    """
    The base class for anything that can be treated as a (schematic) type.
    """

    @abstractmethod
    def instance(self, *arg: VariableTerm, **kwargs: VariableTerm) -> Term:
        return NotImplemented

    def plain(self) -> PlainTerm:
        if isinstance(self, PlainTerm):
            return self
        elif isinstance(self, Term):
            #if self.constraints:
            #    raise ValueError("No")
            return self._plain
        return self.instance().plain()

    def __repr__(self) -> str:
        return str(self)

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
            Term(Function(a._plain, b._plain), *(a.constraints + b.constraints))
        )

    def __call__(self, *args: Type) -> Type:
        """
        Function application. This allows us to apply two types to eachother by
        calling the function type with its argument type.
        """

        return Type.combine(self, *args, by=lambda x, *xs:
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
                return Term(t._plain, constraint, *t.constraints)
            σ.__signature__ = self.signature  # type: ignore
            return Schema(σ)
        else:
            t = self.instance()
            return Term(t._plain, constraint, *t.constraints)

    def __matmul__(self, other: Union[Type, Iterable[Type]]) -> Constraint:
        """
        Allows us to write typeclass constraints using @.
        """
        if isinstance(other, Type):
            constraints = [other.plain()]
        else:
            constraints = list(o.plain() for o in other)
        return Constraint(self.plain(), *constraints)

    def __lt__(self, other: Type) -> Optional[bool]:
        return self != other and self <= other

    def __gt__(self, other: Type) -> Optional[bool]:
        return self != other and self >= other

    def __le__(self, other: Type) -> Optional[bool]:
        return self.plain().subtype(other.plain())

    def __ge__(self, other: Type) -> Optional[bool]:
        return self.plain().subtype(other.plain())

    def is_function(self) -> bool:
        """
        Does this type represent a function?
        """
        t = self.plain()
        return isinstance(t, OperatorTerm) and t.operator == Function

    @staticmethod
    def combine(*types: Type, by: Callable[..., Term]) -> Type:
        """
        Combine several types into a single (possibly schematic) type using a
        function that combines instances of those types into a single term.
        """

        if any(isinstance(t, Schema) for t in types):
            # A new schematic variable for every such one needed by arguments
            n_vars = [t.n_vars if isinstance(t, Schema) else 0 for t in types]
            names = list(varnames(sum(n_vars)))
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
        self._plain = plain
        self.constraints = []

        for c in constraints:
            self.constraints.append(c)

    def __str__(self) -> str:
        res = [str(self._plain)]

        for c in self.constraints:
            res.append(str(c))

        variables = []
        for v in self._plain.variables():
            if v not in variables:
                if v.lower:
                    res.append(f"{v} >= {v.lower}")
                if v.upper:
                    res.append(f"{v} <= {v.upper}")
            variables.append(v)

        return ' | '.join(res)

    def instance(self, *args, **kwargs) -> Term:
        Signature().bind(*args, **kwargs)
        return self

    def resolve(self) -> Term:
        constraints = [c for c in self.constraints if not c.fulfilled()]
        self._plain.resolve()
        return Term(self._plain, *constraints)

    def apply(self, arg: Term) -> Term:
        """
        Apply an argument to a function type to get its output type.
        """

        f = self._plain.follow()
        x = arg._plain.follow()

        if isinstance(f, VariableTerm):
            f.bind(Function(VariableTerm(), VariableTerm()))
            f = f.follow()

        if isinstance(f, OperatorTerm) and f.operator == Function:
            x.unify_subtype(f.params[0])
            f.resolve()
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

    def __contains__(self, value: PlainTerm) -> bool:
        return value == self or (
            isinstance(self, OperatorTerm) and
            any(value in t for t in self.params))

    def basics(self, follow: bool = True) -> Iterable[TermOperator]:
        """
        Find the basic types (optionally only those non-unified).
        """
        a = self.follow() if follow else self
        if isinstance(a, OperatorTerm):
            if a.operator.basic:
                yield a
            for t in chain(*(t.basics(follow) for t in self.params)):
                yield t

    def variables(self) -> Iterable[VariableTerm]:
        """
        Obtain all type variables currently in the type expression.
        """
        a = self.follow()
        if isinstance(a, VariableTerm):
            yield a
        elif isinstance(a, OperatorTerm):
            for v in chain(*(t.variables() for t in a.params)):
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
                    *(p.follow().skeleton() for p in self.params))
        else:
            return self

    def subtype(self, other: PlainTerm) -> Optional[bool]:
        """
        Return True if self is (or can be) definitely a subtype of other, False
        if it is definitely not, and None if there is not enough information.
        """
        a = self.follow()
        b = other.follow()

        if isinstance(a, OperatorTerm) and isinstance(b, OperatorTerm):
            if a.operator.basic:
                return a.operator.subtype(b.operator)
            elif a.operator != b.operator:
                return False
            else:
                result: Optional[bool] = True
                for v, s, t in zip(a.operator.variance, a.params, b.params):
                    if v == Variance.CO:
                        r = s.subtype(t)
                    elif v == Variance.CONTRA:
                        r = t.subtype(s)
                    if r is False:
                        return False
                    elif r is None:
                        result = None
                return result
        elif isinstance(a, OperatorTerm) and isinstance(b, VariableTerm):
            if b.upper and b.upper.subtype(a.operator, True):
                return False
            if b.wildcard:
                return True
        elif isinstance(a, VariableTerm) and isinstance(b, OperatorTerm):
            if a.lower and b.operator.subtype(a.lower, True):
                return False
            if a.wildcard:
                return True
        elif isinstance(a, VariableTerm) and isinstance(b, VariableTerm):
            if a.lower and b.upper and a.lower.subtype(b.upper, True):
                return False
        return None

    def unify_subtype(self, other: PlainTerm) -> None:
        """
        Make sure that a is equal to, or a subtype of b. Like normal
        unification, but instead of just a substitution of variables to terms,
        also produces lower and upper bounds on subtypes that it must respect.
        """
        a = self.follow()
        b = other.follow()

        if isinstance(a, OperatorTerm) and isinstance(b, OperatorTerm):
            if a.operator.basic:
                if not a.operator.subtype(b.operator):
                    raise error.SubtypeMismatch(a, b)
            elif a.operator == b.operator:
                for v, x, y in zip(a.operator.variance, a.params, b.params):
                    if v == Variance.CO:
                        x.unify_subtype(y)
                    elif v == Variance.CONTRA:
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

    def resolve(self, prefer_lower: bool = True) -> PlainTerm:
        """
        Obtain a version of this type with all eligible variables with subtype
        constraints resolved to their most specific type.
        """
        # For starters, prefer the lower bound on a variable. After all, a
        # value of type T is also a value of type S for T < S; so the lower
        # bound represents the most specific type without loss of generality.
        a = self.follow()

        if isinstance(a, OperatorTerm):
            for v, p in zip(a.operator.variance, a.params):
                p.resolve(prefer_lower ^ (v == Variance.CONTRA))
        elif isinstance(a, VariableTerm):
            if prefer_lower and a.lower:
                a.bind(a.lower())
            elif not prefer_lower and a.upper:
                a.bind(a.upper())
        return a.follow()

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
            self.variance = list(Variance.CO for _ in range(params))
        else:
            self.variance = list(params)
        self.arity = len(self.variance)

        if self.supertype and not self.basic:
            raise ValueError("only nullary types can have direct supertypes")

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Operator) and
            self.name == other.name
            and self.variance == other.variance)

    def __call__(self, *params: Type) -> OperatorTerm:  # type: ignore
        return OperatorTerm(self, *(p.plain() for p in params))

    def subtype(self, other: Operator, strict: bool = False) -> bool:
        return ((not strict and self == other) or
            bool(self.supertype and self.supertype.subtype(other)))

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

    def __init__(self, op: Operator, *params: PlainTerm):
        self.operator = op
        self.params = list(params)

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
            return (self.operator == other.operator and
                all(s == t for s, t in zip(self.params, other.params)))
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
        if self.unified:
            return str(self.unified)
        return "_" if self.wildcard else self.name or f"_{self.id}"

    def __eq__(self, other: object) -> bool:
        if self.unified:
            return self.follow() == other
        elif isinstance(other, VariableTerm) and other.unified:
            return self == other.follow()
        return super().__eq__(other)

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
                if self.lower and t.operator.subtype(self.lower, True):
                    raise error.SubtypeMismatch(t, self)
                if self.upper and self.upper.subtype(t.operator, True):
                    raise error.SubtypeMismatch(self, t)

    def above(self, new: Operator) -> None:
        """
        Constrain this variable to be a basic type with the given type as lower
        bound.
        """
        lower, upper = self.lower or new, self.upper or new

        # lower bound higher than the upper bound fails
        if upper.subtype(new, True):
            raise error.SubtypeMismatch(new, upper)

        # lower bound lower than the current lower bound is ignored
        elif new.subtype(lower, True):
            pass

        # tightening the lower bound
        elif lower.subtype(new):
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
        if new.subtype(lower, True):
            raise error.SubtypeMismatch(lower, new)
        elif upper.subtype(new, True):
            pass
        elif new.subtype(upper):
            self.upper = new
        else:
            raise error.SubtypeMismatch(new, upper)


"The special constructor for function types."
Function = Operator(
    'Function',
    params=(Variance.CONTRA, Variance.CO)
)

"A wildcard: produces an unrelated variable, to be matched with anything."
_ = Schema(lambda: VariableTerm(wildcard=True))


class Constraint(object):
    """
    A constraint enforces that its subject type is a subtype of one of its
    object types.
    """

    def __init__(self, subject: PlainTerm, *objects: PlainTerm):
        self.subject = subject
        self.objects = list(objects)
        self.objects_initial = self.objects
        self.fulfilled()

    def __str__(self) -> str:
        return f"{self.subject} @ {self.objects_initial}"

    def fulfilled(self) -> bool:
        """
        Check that the constraint has not been violated and raise an error
        otherwise. Additionally, return True if it has been completely
        fulfilled and need not be enforced any longer.
        """
        compatibility = [self.subject.subtype(t) for t in self.objects]
        self.objects = [
            t for t, c in zip(self.objects, compatibility) if c is not False
        ]

        if len(self.objects) == 0:
            raise error.ViolatedConstraint(self)

        elif len(self.objects) == 1:
            obj = self.objects[0]

            # This only works if there are no non-unified basic types, because
            # we might be looking for a subtype of that basic type. In that
            # case, we don't want to unify already, because the variable would
            # be resolved against an overly loose bound
            if not any(obj.basics(follow=False)):
                self.subject.unify_subtype(obj)
                return True

        # A constraint is also fulfilled if the subject is fully concrete and
        # there is at least one definitely compatible object
        if not any(self.subject.variables()) and any(compatibility):
            return True

        return False


def operators(
        *ops: Operator,
        param: Optional[Type] = None,
        at: int = None) -> List[PlainTerm]:
    """
    Generate a list of instances of operator terms. Optionally, the generated
    operator terms must contain a certain parameter (at some index, if given).
    """
    options: List[PlainTerm] = []
    for op in ops:
        for i in ([at - 1] if at else range(op.arity)) if param else [-1]:
            if i < op.arity:
                options.append(op(*(
                    param if param and i == j else _ for j in range(op.arity)
                )))
    return options


def varnames(n: int, unicode: bool = False) -> Iterable[str]:
    """
    Produce some suitable variable names.
    """
    base = "τσαβγφψ" if unicode else "xyzuvwabcde"
    for i in range(n):
        yield base[i] if n < len(base) else base[0] + str(i + 1)
