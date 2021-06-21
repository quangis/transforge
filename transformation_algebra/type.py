"""
Generic type system. Inspired loosely by Hindley-Milner type inference in
functional programming languages, as well as Traytel et al (2011).
"""
from __future__ import annotations

from enum import Enum, auto
from abc import ABC, abstractmethod
from itertools import chain
from inspect import signature
from typing import Optional, Iterator, Iterable, Union, Callable, List, Set

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
    The base class for anything that can be treated as a type: type schemata
    and type instances.
    """

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

    def __lt__(self, other: Type) -> Optional[bool]:
        a, b = self.instance(), other.instance()
        return not a.match(b) and a <= b

    def __gt__(self, other: Type) -> Optional[bool]:
        a, b = self.instance(), other.instance()
        return not a.match(b) and a >= b

    def __le__(self, other: Type) -> Optional[bool]:
        return self.instance().match(other.instance(),
            accept_wildcard=True, subtype=True)

    def __ge__(self, other: Type) -> Optional[bool]:
        return self.instance().match(other.instance(),
            accept_wildcard=True, subtype=True)

    def apply(self, arg: Type) -> TypeInstance:
        """
        Apply an argument to a function type to get its output type.
        """

        f = self.instance().follow()
        x = arg.instance().follow()

        if isinstance(f, TypeVar):
            f.bind(Function(TypeVar(), TypeVar()))
            f = f.follow()

        if isinstance(f, TypeOperation) and f.operator == Function:
            x.unify(f.params[0], subtype=True)
            f.resolve()
            return f.params[1].resolve()
        else:
            raise error.FunctionApplicationError(f, x)

    def is_function(self) -> bool:
        """
        Does this type represent a function?
        """
        t = self.instance()
        return isinstance(t, TypeOperation) and t.operator == Function

    @abstractmethod
    def instance(self) -> TypeInstance:
        return NotImplemented

    @staticmethod
    def declare(name: str, params: Union[int, Iterable[Variance]] = 0,
            supertype: Optional[TypeOperator] = None) -> TypeOperator:
        """
        Convenience function for defining a type.
        """
        if isinstance(params, int):
            variance = list(Variance.CO for _ in range(params))
        else:
            variance = list(params)
        return TypeOperator(name=name, params=variance, supertype=supertype)


class TypeSchema(Type):
    """
    Provides a definition of a *schema* for function and data signatures, that
    is, a type containing some schematic type variable.
    """

    def __init__(self, schema: Callable[..., TypeInstance]):
        self.schema = schema
        self.n = len(signature(schema).parameters)

    def __str__(self) -> str:
        return self.schema(
            *(TypeVar(v) for v in signature(self.schema).parameters)
        ).resolve().str_with_constraints()

    def instance(self) -> TypeInstance:
        return self.schema(*(TypeVar() for _ in range(self.n)))


class TypeOperator(Type):
    """
    An n-ary type constructor. If 0-ary, can also be treated as an instance of
    the corresponding type operation (that is, a base type).
    """

    def __init__(
            self,
            name: str,
            params: List[Variance] = [],
            supertype: Optional[TypeOperator] = None):
        self.name = name
        self.supertype: Optional[TypeOperator] = supertype
        self.variance = params
        self.arity = len(params)

        if self.supertype and self.arity > 0:
            raise ValueError("only nullary types can have direct supertypes")

    def __str__(self) -> str:
        return self.name

    def __call__(self, *params: Type) -> TypeOperation:
        return TypeOperation(self, *(p.instance() for p in params))

    def subtype(self, other: TypeOperator, strict: bool = False) -> bool:
        assert isinstance(other, TypeOperator)
        return ((not strict and self == other) or
            bool(self.supertype and self.supertype.subtype(other)))

    def instance(self) -> TypeInstance:
        return TypeOperation(self)


class TypeInstance(Type):
    """
    Base class for type instances (type operations and -variables). Note that
    base types are just 0-ary type operators and functions are just particular
    2-ary type operators.
    """

    def str_with_constraints(self) -> str:
        """
        Like str(), but includes constraints.
        """
        result = [str(self)]
        for v in self.variables():
            if v.lower:
                result.append(f"{v} >= {v.lower}")
            if v.upper:
                result.append(f"{v} <= {v.upper}")
        result.extend(str(c) for c in self.constraints())
        return ' | '.join(result)

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

    def __iter__(self) -> Iterator[TypeInstance]:
        """
        Iterate through all substructures of this type instance.
        """
        a = self.follow()
        yield a
        if isinstance(a, TypeOperation):
            yield from chain(*a.params)

    def __contains__(self, value: TypeInstance) -> bool:
        a = self.follow()
        b = value.follow()
        return a.match(b) is True or (
            isinstance(a, TypeOperation) and
            any(b in t for t in a.params))

    def variables(self, recursive: bool = True) -> Set[TypeVar]:
        """
        Obtain all distinct type variables currently in the type instance.
        """
        result = set(t for t in self if isinstance(t, TypeVar))
        if not recursive:
            return result
        stack = list(result)
        while stack:
            var = stack.pop()
            for constraint in var._constraints:
                for element in constraint.parts():
                    for extra_var in element.variables(False):
                        if extra_var not in result:
                            stack.append(extra_var)
                            result.add(extra_var)
        return result

    def constraints(self, recursive: bool = True) -> Set[Constraint]:
        """
        Obtain all constraints attached to variables in the type instance.
        """
        result = set()
        for v in self.variables(recursive=recursive):
            result.update(v._constraints)
        return result

    def operators(self, recursive: bool = True) -> Set[TypeOperator]:
        """
        Obtain all distinct non-function operators in the type instance.
        """
        result = set(t.operator for t in self
            if isinstance(t, TypeOperation) and t.operator != Function)
        if recursive:
            for constraint in self.constraints():
                result.update(*(p.operators(False) for p in constraint.parts()))
        return result

    def follow(self) -> TypeInstance:
        """
        Follow a unification until bumping into a type that is not yet bound.
        """
        if isinstance(self, TypeVar) and self.unification:
            return self.unification.follow()
        return self

    def skeleton(self) -> TypeInstance:
        """
        A copy in which base types are substituted with fresh variables.
        """
        if isinstance(self, TypeOperation):
            if self.basic:
                return TypeVar()
            else:
                return TypeOperation(
                    self.operator,
                    *(p.follow().skeleton() for p in self.params))
        else:
            return self

    def match(self, other: TypeInstance, subtype: bool = False,
            accept_wildcard: bool = False) -> Optional[bool]:
        """
        Return True if self is definitely the same as (or a subtype of) other,
        False if it is definitely not, and None if there is not enough
        information.
        """
        a = self.follow()
        b = other.follow()

        if isinstance(a, TypeOperation) and isinstance(b, TypeOperation):
            if a.basic:
                return a.operator == b.operator or \
                    (subtype and a.operator.subtype(b.operator))
            elif a.operator != b.operator:
                return False
            else:
                result: Optional[bool] = True
                for v, s, t in zip(a.operator.variance, a.params, b.params):
                    r = TypeInstance.match(
                        *((s, t) if v == Variance.CO else (t, s)),
                        subtype=subtype, accept_wildcard=accept_wildcard)
                    if r is False:
                        return False
                    elif r is None:
                        result = None
                return result
        elif isinstance(a, TypeOperation) and isinstance(b, TypeVar):
            if (b.upper or b.lower) and not a.basic:
                return False
            if b.upper and b.upper.subtype(a.operator, strict=True):
                return False
            if accept_wildcard and b.wildcard:
                return True
        elif isinstance(a, TypeVar) and isinstance(b, TypeOperation):
            if (a.upper or a.lower) and not b.basic:
                return False
            if a.lower and b.operator.subtype(a.lower, strict=True):
                return False
            if accept_wildcard and a.wildcard:
                return True
        elif isinstance(a, TypeVar) and isinstance(b, TypeVar):
            if a == b or (a.wildcard and b.wildcard):
                return True
            if accept_wildcard and (a.wildcard or b.wildcard):
                return True
            if a.lower and b.upper and a.lower.subtype(b.upper, strict=True):
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

        if isinstance(a, TypeVar) and isinstance(b, TypeVar):
            a.bind(b)
        elif isinstance(a, TypeOperation) and isinstance(b, TypeOperation):
            if a.basic:
                if subtype and not a.operator.subtype(b.operator):
                    raise error.SubtypeMismatch(a, b)
                elif not subtype and a.operator != b.operator:
                    raise error.TypeMismatch(a, b)
            elif a.operator == b.operator:
                for v, x, y in zip(a.operator.variance, a.params, b.params):
                    if v == Variance.CO:
                        x.unify(y, subtype=subtype)
                    else:
                        assert v == Variance.CONTRA
                        y.unify(x, subtype=subtype)
            else:
                raise error.TypeMismatch(a, b)
        else:
            if isinstance(a, TypeOperation) and isinstance(b, TypeVar):
                op, var, f = a, b, TypeVar.above
            else:
                assert isinstance(a, TypeVar) and isinstance(b, TypeOperation)
                op, var, f = b, a, TypeVar.below

            if var in op:
                raise error.RecursiveType(var, op)
            elif op.basic:
                if subtype:
                    f(var, op.operator)
                else:
                    var.bind(op)
            else:
                var.bind(op.skeleton())
                var.unify(op, subtype=subtype)

    def instance(self) -> TypeInstance:
        return self

    @property
    def operator(self) -> Optional[TypeOperator]:
        # Overridden for TypeOperations
        return None

    def __or__(self, constraint: Constraint) -> TypeInstance:
        """
        Another abuse of Python's operators, allowing us to add constraints to
        a type instance by using the | operator.
        """
        # Constraints are attached to the relevant variables when the
        # constraint is instantiated; so at this point, we only attach context.
        constraint.set_context(self)
        return self

    def __matmul__(self, other: Union[Type, Iterable[Type]]) -> Constraint:
        """
        Allows us to write typeclass constraints using @.
        """
        if isinstance(other, Type):
            return Constraint(self, other.instance())
        else:
            assert isinstance(other, Iterable)
            return Constraint(self, *(o.instance() for o in other))


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

    def __str__(self) -> str:
        if self.operator == Function:
            inT, outT = self.params
            if isinstance(inT, TypeOperation) and inT.operator == Function:
                return f"({inT}) ** {outT}"
            return f"{inT} ** {outT}"
        elif self.params:
            return f'{self.operator}({", ".join(str(t) for t in self.params)})'
        else:
            return str(self.operator)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TypeInstance) and bool(self.match(other))

    def __getitem__(self, key: int) -> TypeInstance:
        return self.params[key]

    @property
    def basic(self) -> bool:
        return self.operator.arity == 0


class TypeVar(TypeInstance):
    """
    A type variable. This is not a schematic variable — it is instantiated!
    """

    def __init__(self, name: Optional[str] = None, wildcard: bool = False):
        self._name = name
        self.wildcard = wildcard
        self.unification: Optional[TypeInstance] = None
        self.lower: Optional[TypeOperator] = None
        self.upper: Optional[TypeOperator] = None
        self._constraints: Set[Constraint] = set()

    def __str__(self) -> str:
        if self.unification:
            return str(self.unification)
        return "_" if self.wildcard else self.name

    @property
    def name(self) -> str:
        return self._name or f"var{hash(self)}"

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    def bind(self, t: TypeInstance) -> None:
        assert (not self.unification or t == self.unification), \
            "variable cannot be unified twice"

        # Once a wildcard variable is bound, it is no longer wildcard
        self.wildcard = False

        if self is not t:
            self.unification = t

            if isinstance(t, TypeVar):
                t._constraints.update(self._constraints)
                self._constraints = t._constraints
                t.check_constraints()

                t.wildcard = False

                if self.lower:
                    t.above(self.lower)
                if self.upper:
                    t.below(self.upper)

                if t.lower == t.upper and t.lower is not None:
                    t.bind(t.lower())

            elif isinstance(t, TypeOperation):
                if t.basic:
                    if self.lower and t.operator.subtype(self.lower, strict=True):
                        raise error.SubtypeMismatch(t, self)
                    if self.upper and self.upper.subtype(t.operator, strict=True):
                        raise error.SubtypeMismatch(self, t)
                else:
                    variables = t.variables(recursive=False)

                    self._constraints.update(chain(*(
                        v._constraints for v in variables
                    )))
                    for v in variables:
                        v._constraints = self._constraints
                        v.check_constraints()

            self.check_constraints()

    def above(self, new: TypeOperator) -> None:
        """
        Constrain this variable to be a basic type with the given type as lower
        bound.
        """
        lower, upper = self.lower or new, self.upper or new
        if upper.subtype(new, True):  # fail when lower bound higher than upper
            raise error.SubtypeMismatch(new, upper)
        elif new.subtype(lower, True):  # ignore lower bound lower than current
            pass
        elif lower.subtype(new):  # tighten the lower bound
            self.lower = new
            self.check_constraints()
        else:  # fail on bound from other lineage (neither sub- nor supertype)
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
            self.check_constraints()
        else:
            raise error.SubtypeMismatch(new, upper)

    def check_constraints(self) -> None:
        self._constraints = set(
            c for c in self._constraints if not c.fulfilled())


class Constraint(object):
    """
    A constraint enforces that its reference is a subtype of one of the given
    alternatives.
    """

    def __init__(
            self,
            reference: TypeInstance,
            *alternatives: TypeInstance,
            context: Optional[TypeInstance] = None):
        self.reference = reference
        self.alternatives = list(alternatives)
        self.description = str(self)
        self.context = context
        self.skeleton: Optional[TypeInstance] = None

        # Inform variables about the constraint present on them
        for v in self.variables():
            assert not v.unification
            v._constraints.add(self)

        self.fulfilled()

    def __str__(self) -> str:
        return f"{self.reference} @ {self.alternatives}"

    def parts(self) -> Iterator[TypeInstance]:
        """
        An iterator of the constituent parts of this constraint.
        """
        return chain(self.reference, *self.alternatives)

    def variables(self) -> Set[TypeVar]:
        return self.reference.variables(True)

    def set_context(self, context: TypeInstance) -> None:
        self.context = context
        self.description = f"{context} | {self}"
        if not self.all_variables_occur_in_context():
            raise error.ConstrainFreeVariable(self)

    def all_variables_occur_in_context(self) -> bool:
        """
        Are all variables in this constraint bound to a variable in the
        context in which it is applied?
        """
        return bool(self.context and all(
            var.wildcard or var in self.context
            for el in self.parts()
            for var in el.variables(recursive=False)))

    def minimize(self) -> None:
        """
        Ensure that constraints don't contain extraneous alternatives:
        alternatives that are equal to, or more general versions of, other
        alternatives.
        """
        minimized: List[TypeInstance] = []
        for obj in self.alternatives:
            add = True
            for i in range(len(minimized)):
                if obj.match(minimized[i], subtype=True) is True:
                    if minimized[i].match(obj, subtype=True) is not True:
                        minimized[i] = obj
                    add = False
            if add:
                minimized.append(obj)
        self.alternatives = minimized

    def fulfilled(self) -> bool:
        """
        Check that the constraint has not been violated and raise an error
        otherwise. Additionally, return True if it has been completely
        fulfilled and need not be enforced any longer.
        """

        # TODO Minimization is expensive and it is performed on every variable
        # binding. Make this more efficient?
        self.minimize()

        compatibility = [
            self.reference.match(t, subtype=True, accept_wildcard=True)
            for t in self.alternatives]
        self.alternatives = [t for t, c in zip(self.alternatives, compatibility)
            if c is not False]

        if len(self.alternatives) == 0:
            raise error.ConstraintViolation(self)

        # If there is only one possibility left, we can unify, but *only* with
        # the skeleton: the base types must remain variable, because we don't
        # want to resolve against an overly loose subtype bound.
        elif len(self.alternatives) == 1 and not self.skeleton:
            self.skeleton = self.alternatives[0].skeleton()
            self.reference.unify(self.skeleton)

        # Fulfillment is achieved if the reference is fully concrete and there
        # is at least one definitely compatible alternative
        return (not any(self.reference.variables()) and any(compatibility)) or \
            self.reference in self.alternatives


"The special constructor for function types."
Function = TypeOperator('Function', params=[Variance.CONTRA, Variance.CO])

"A wildcard: fresh variable, unrelated to, and matchable with, anything else."
_ = TypeSchema(lambda: TypeVar(wildcard=True))


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
