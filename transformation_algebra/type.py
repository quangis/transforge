"""
Generic type system. Inspired loosely by Hindley-Milner type inference in
functional programming languages, as well as Traytel et al (2011).
"""
from __future__ import annotations

from enum import Enum, auto
from abc import ABC, abstractmethod
from itertools import chain
from inspect import signature
from typing import Optional, Iterable, Union, Callable, List, Set

from transformation_algebra import error


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
    def instance(self, *arg: TypeVar, **kwargs: TypeVar) -> TypeInstance:
        return NotImplemented

    def __repr__(self) -> str:
        return str(self)

    def __pow__(self, other: Type) -> TypeInstance:
        """
        Function abstraction. This is an overloaded (ab)use of Python's
        exponentiation operator. It allows us to use the right-to-left
        associative infix operator ** for the arrow in function signatures.
        """
        # Note that this operator is one of the few that is right-to-left
        # associative, matching the conventional behaviour of the function
        # arrow. The right-bitshift operator >> (for __rshift__) might have
        # been more intuitive visually, but would not have this property.
        return Function(self.instance(), other.instance())

    def __or__(self, constraint: Constraint) -> Type:
        """
        Another abuse of Python's operators, allowing us to add constraints by
        using the | operator.
        """
        # Since constraints are fully attached during their instantiation, we
        # don't have to do anything here
        return self

    def __matmul__(self, other: Union[Type, Iterable[Type]]) -> Constraint:
        """
        Allows us to write typeclass constraints using @.
        """
        return Constraint(
            self.instance(), *(
                (other.instance(),)
                if isinstance(other, Type) else
                (o.instance() for o in other)
            ))

    def __lt__(self, other: Type) -> Optional[bool]:
        return self != other and self <= other

    def __gt__(self, other: Type) -> Optional[bool]:
        return self != other and self >= other

    def __le__(self, other: Type) -> Optional[bool]:
        return self.instance().subtype(other.instance())

    def __ge__(self, other: Type) -> Optional[bool]:
        return self.instance().subtype(other.instance())

    def is_function(self) -> bool:
        """
        Does this type represent a function?
        """
        t = self.instance()
        return isinstance(t, TypeOperation) and t.operator == Function


class TypeSchema(Type):
    """
    Provides a definition of a *schema* for function and data signatures, that
    is, a type containing some schematic type variable.
    """

    def __init__(self, schema: Callable[..., Type]):
        self.schema = schema
        self.signature = signature(schema)
        self.nvar = len(self.signature.parameters)

    def __str__(self) -> str:
        return str(self.instance(*(
            TypeVar(v) for v in self.signature.parameters)).resolve())

    def instance(self, *args: TypeVar, **kwargs: TypeVar) -> TypeInstance:
        """
        Create an instance of this schema. Optionally bind schematic variables
        to concrete variables; non-bound variables will get automatically
        assigned a concrete variable.
        """
        binding = self.signature.bind_partial(*args, **kwargs)
        for param in self.signature.parameters:
            if param not in binding.arguments:
                binding.arguments[param] = TypeVar()
        return self.schema(*binding.args, **binding.kwargs).instance()


class TypeInstance(Type):
    """
    Base class for type instances (type operations and -variables). Note that
    base types are just 0-ary type operators and functions are just particular
    2-ary type operators.
    """

    def __str__(self) -> str:
        result = [self.str2()]
        constraints: Set[Constraint] = set()

        for v in self.variables():
            constraints = constraints.union(v.constraints)
            if v.lower:
                result.append(f"{v} >= {v.lower}")
            if v.upper:
                result.append(f"{v} <= {v.upper}")

        result.extend(str(c) for c in constraints)
        return ' | '.join(result)

    @abstractmethod
    def str2(self) -> str:
        return NotImplemented

    def instance(self, *args, **kwargs) -> TypeInstance:
        assert not args and not kwargs
        return self

    def resolve(self, prefer_lower: bool = True) -> TypeInstance:
        """
        Obtain a version of this type with all eligible variables with subtype
        constraints resolved to their most specific type.
        """
        a = self.follow()

        if isinstance(a, TypeOperation):
            for v, p in zip(a.operator.variance, a.params):
                p.resolve(prefer_lower ^ (v == Variance.CONTRA))
        elif isinstance(a, TypeVar):
            if prefer_lower and a.lower:
                a.bind(a.lower())
            elif not prefer_lower and a.upper:
                a.bind(a.upper())
        return a.follow()

    def apply(self, arg: TypeInstance) -> TypeInstance:
        """
        Apply an argument to a function type to get its output type.
        """

        f = self.follow()
        x = arg.follow()

        if isinstance(f, TypeVar):
            f.bind(Function(TypeVar(), TypeVar()))
            f = f.follow()

        if isinstance(f, TypeOperation) and f.operator == Function:
            x.unify(f.params[0], subtype=True)
            f.resolve()
            return f.params[1].resolve()
        else:
            raise error.NonFunctionApplication(f, x)

    def __contains__(self, value: TypeInstance) -> bool:
        return value == self or (
            isinstance(self, TypeOperation) and
            any(value in t for t in self.params))

    def variables(self, distinct: bool = False) -> Iterable[TypeVar]:
        """
        Obtain all distinct type variables currently in the type expression.
        """
        return {v.id: v for v in self._variables()}.values() \
            if distinct else self._variables()

    def _variables(self) -> Iterable[TypeVar]:
        a = self.follow()
        if isinstance(a, TypeVar):
            yield a
        elif isinstance(a, TypeOperation):
            for v in chain(*(t.variables() for t in a.params)):
                yield v

    def follow(self) -> TypeInstance:
        """
        Follow a unification until bumping into a type that is not yet bound.
        """
        if isinstance(self, TypeVar) and self.unified:
            return self.unified.follow()
        return self

    def skeleton(self) -> TypeInstance:
        """
        A copy in which base types are substituted with fresh variables.
        """
        if isinstance(self, TypeOperation):
            if self.operator.basic:
                return TypeVar()
            else:
                return TypeOperation(
                    self.operator,
                    *(p.follow().skeleton() for p in self.params))
        else:
            return self

    def subtype(self, other: TypeInstance) -> Optional[bool]:
        """
        Return True if self can definitely be a subtype of other, False if it
        is definitely not, and None if there is not enough information.
        """
        a = self.follow()
        b = other.follow()

        if isinstance(a, TypeOperation) and isinstance(b, TypeOperation):
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
        elif isinstance(a, TypeOperation) and isinstance(b, TypeVar):
            if (b.upper or b.lower) and a.operator.compound:
                return False
            if b.upper and b.upper.subtype(a.operator, True):
                return False
            if b.wildcard:
                return True
        elif isinstance(a, TypeVar) and isinstance(b, TypeOperation):
            if (a.upper or a.lower) and b.operator.compound:
                return False
            if a.lower and b.operator.subtype(a.lower, True):
                return False
            if a.wildcard:
                return True
        elif isinstance(a, TypeVar) and isinstance(b, TypeVar):
            if a.lower and b.upper and a.lower.subtype(b.upper, True):
                return False
        return None

    def unify(self, other: TypeInstance, subtype: bool = False) -> None:
        """
        Make sure that a is equal to, or a subtype of b. Like normal
        unification, but instead of just a substitution of variables, also
        produces new variables with subtype- and supertype bounds.
        """
        a = self.follow()
        b = other.follow()

        if isinstance(a, TypeOperation) and isinstance(b, TypeOperation):
            if a.operator.basic:
                if subtype and not a.operator.subtype(b.operator):
                    raise error.SubtypeMismatch(a, b)
                elif not subtype and not a.operator != b.operator:
                    raise error.TypeMismatch(a, b)
            elif a.operator == b.operator:
                for v, x, y in zip(a.operator.variance, a.params, b.params):
                    if v == Variance.CO:
                        x.unify(y, subtype=subtype)
                    elif v == Variance.CONTRA:
                        y.unify(x, subtype=subtype)
                    else:
                        raise ValueError
            else:
                raise error.TypeMismatch(a, b)

        elif isinstance(a, TypeVar) and isinstance(b, TypeVar):
            a.bind(b)

        elif isinstance(a, TypeVar) and isinstance(b, TypeOperation):
            if a in b:
                raise error.RecursiveType(a, b)
            elif b.operator.basic:
                if subtype:
                    a.below(b.operator)
                else:
                    a.bind(b)
            else:
                a.bind(b.skeleton())
                a.unify(b, subtype=subtype)

        elif isinstance(a, TypeOperation) and isinstance(b, TypeVar):
            if b in a:
                raise error.RecursiveType(b, a)
            elif a.operator.basic:
                if subtype:
                    b.above(a.operator)
                else:
                    b.bind(a)
            else:
                b.bind(a.skeleton())
                b.unify(a, subtype=subtype)


class TypeOperator(Type):
    """
    An n-ary type constructor. If 0-ary, can also be treated as an instance of
    the corresponding type operation (that is, a base type).
    """

    def __init__(
            self,
            name: str,
            params: Union[int, Iterable[Variance]] = 0,
            supertype: Optional[TypeOperator] = None):
        self.name = name
        self.supertype: Optional[TypeOperator] = supertype

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
            isinstance(other, TypeOperator) and
            self.name == other.name
            and self.variance == other.variance)

    def __call__(self, *params: Type) -> TypeOperation:  # type: ignore
        return TypeOperation(self, *(p.instance() for p in params))

    def subtype(self, other: TypeOperator, strict: bool = False) -> bool:
        return ((not strict and self == other) or
            bool(self.supertype and self.supertype.subtype(other)))

    def instance(self, *args, **kwargs) -> TypeInstance:
        assert not args and not kwargs
        return TypeOperation(self)

    @property
    def basic(self) -> bool:
        return self.arity == 0

    @property
    def compound(self) -> bool:
        return not self.basic


class TypeOperation(TypeInstance):
    """
    An instance of an n-ary type constructor.
    """

    def __init__(self, op: TypeOperator, *params: TypeInstance):
        self.operator = op
        self.params = list(params)

        if len(self.params) != self.operator.arity:
            raise ValueError(
                f"{self.operator} takes {self.operator.arity} "
                f"parameter{'' if self.operator.arity == 1 else 's'}; "
                f"{len(self.params)} given"
            )

    def str2(self) -> str:
        if self.operator == Function:
            inT, outT = self.params
            if isinstance(inT, TypeOperation) and inT.operator == Function:
                return f"({inT}) ** {outT}"
            return f"{inT} ** {outT}"
        elif self.params:
            return f'{self.operator}({", ".join(t.str2() for t in self.params)})'
        else:
            return str(self.operator)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TypeOperation):
            return (self.operator == other.operator and
                all(s == t for s, t in zip(self.params, other.params)))
        return False


class TypeVar(TypeInstance):
    """
    A type variable. This is not a schematic variable — it is instantiated!
    """
    counter = 0

    def __init__(self, name: Optional[str] = None, wildcard: bool = False):
        self.name = name
        self.wildcard = wildcard
        self.lower: Optional[TypeOperator] = None
        self.unified: Optional[TypeInstance] = None
        self.upper: Optional[TypeOperator] = None
        self.constraints: Set[Constraint] = set()
        cls = type(self)
        self.id = cls.counter
        cls.counter += 1

    def str2(self) -> str:
        if self.unified:
            return str(self.unified)
        return "_" if self.wildcard else self.name or f"_{self.id}"

    def __eq__(self, other: object) -> bool:
        if self.unified:
            return self.follow() == other
        elif isinstance(other, TypeVar) and other.unified:
            return self == other.follow()
        return super().__eq__(other)

    def bind(self, t: TypeInstance) -> None:
        assert (not self.unified or t == self.unified), \
            "variable cannot be unified twice"

        # Once a wildcard variable is bound, it is no longer wildcard
        self.wildcard = False

        if self is not t:
            self.unified = t

            if isinstance(t, TypeVar):
                constraints = set.union(self.constraints, t.constraints)
                self.constraints = constraints
                t.constraints = constraints

                t.wildcard = False

                if self.lower:
                    t.above(self.lower)
                if self.upper:
                    t.below(self.upper)

                if t.lower == t.upper and t.lower is not None:
                    t.bind(t.lower())

            elif isinstance(t, TypeOperation) and t.operator.basic:
                if self.lower and t.operator.subtype(self.lower, True):
                    raise error.SubtypeMismatch(t, self)
                if self.upper and self.upper.subtype(t.operator, True):
                    raise error.SubtypeMismatch(self, t)

            self.check_constraints(True)

    def above(self, new: TypeOperator) -> None:
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
            self.check_constraints(False)

        # new bound from another lineage (neither sub- nor supertype) fails
        else:
            raise error.SubtypeMismatch(lower, new)

    def below(self, new: TypeOperator) -> None:
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
            self.check_constraints(False)
        else:
            raise error.SubtypeMismatch(new, upper)

    def check_constraints(self, unify: bool) -> None:
        self.constraints = set(
            c for c in self.constraints if not c.fulfilled(unify))


"The special constructor for function types."
Function = TypeOperator('Function', params=(Variance.CONTRA, Variance.CO))

"A wildcard: fresh variable, unrelated to, and matchable with, anything else."
_ = TypeSchema(lambda: TypeVar(wildcard=True))


class Constraint(object):
    """
    A constraint enforces that its subject type is a subtype of one of its
    object types.
    """

    def __init__(self, subject: TypeInstance, *objects: TypeInstance):
        self.subject = subject
        self.objects = list(objects)
        self.fulfilled()

        # Inform variables about the constraint present on them
        for v in self.variables():
            assert not v.unified
            v.constraints.add(self)

    def __str__(self) -> str:
        return f"{self.subject.str2()} @ {[c.str2() for c in self.objects]}"

    def variables(self) -> Iterable[TypeVar]:
        return chain(
            self.subject.variables(),
            *(o.variables() for o in self.objects)
        )

    def fulfilled(self, unify: bool = True) -> bool:
        """
        Check that the constraint has not been violated and raise an error
        otherwise. Additionally, return True if it has been completely
        fulfilled and need not be enforced any longer.
        """
        compatibility = [self.subject.subtype(t) for t in self.objects]
        self.objects = [t for t, c in zip(self.objects, compatibility)
            if c is not False]

        if len(self.objects) == 0:
            raise error.ViolatedConstraint(self)

        # If there is only one possibility left, we can unify, but *only* with
        # the skeleton: the base types must remain variable, because we don't
        # want to resolve against an overly loose subtype bound.
        elif len(self.objects) == 1 and unify:
            self.subject.unify(self.objects[0].skeleton())

        # Fulfillment is achieved if the subject is fully concrete and there is
        # at least one definitely compatible object
        return not any(self.subject.variables()) and any(compatibility)


def operators(
        *ops: TypeOperator,
        param: Optional[Type] = None,
        at: int = None) -> List[TypeInstance]:
    """
    Generate a list of instances of type operations. Optionally, the generated
    type operations must contain a certain parameter (at some index, if given).
    """
    options: List[TypeInstance] = []
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
