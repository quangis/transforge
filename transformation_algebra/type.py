"""
Generic type system. Inspired loosely by Hindley-Milner type inference in
functional programming languages, as well as Traytel et al (2011).
"""
from __future__ import annotations

from enum import Enum, auto
from abc import ABC, abstractmethod
from functools import reduce
from itertools import chain
from inspect import signature
from typing import Optional, Iterator, Iterable, Callable

from transformation_algebra.label import Labels


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

    def __rpow__(self, other: Type | tuple[Type]) -> TypeInstance:
        """
        The reflected exponentiation operator allows for a tuple to occur as
        the first argument of a type. This allows us to make an uncurried type
        of the form `(α, β) ** γ` into the curried type `α ** β ** γ`
        transparently.
        """
        if isinstance(other, tuple):
            return reduce(lambda x, y: Type.__pow__(y, x),
                reversed(other), self).instance()
        else:
            return other.__pow__(self)

    def __mul__(self, other: Type) -> TypeInstance:
        """
        Tuple operator.
        """
        # See issue #78
        return Product(self.instance(), other.instance())

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

        if isinstance(f, TypeVariable):
            f.bind(Function(TypeVariable(), TypeVariable()))
            f = f.follow()

        if isinstance(f, TypeOperation) and f._operator == Function:
            x.unify(f.params[0], subtype=True)
            f.params[0].fix(prefer_lower=False)
            # f.fix()
            return f.params[1].fix()
        else:
            raise FunctionApplicationError(f, x)

    @abstractmethod
    def instance(self) -> TypeInstance:
        return NotImplemented


class TypeSchema(Type):
    """
    Provides a definition of a *schema* for function and data signatures, that
    is, a type containing some schematic type variable.
    """

    def __init__(self, schema: Callable[..., TypeInstance]):
        self.schema = schema
        self.n = len(signature(schema).parameters)

    def __str__(self) -> str:
        names = signature(self.schema).parameters
        variables = [TypeVariable() for _ in names]
        return self.schema(*variables).text(
            with_constraints=True,
            labels={v: k for k, v in zip(names, variables)})

    def instance(self, origin=None) -> TypeInstance:
        return self.schema(*(TypeVariable(origin=origin)
            for _ in range(self.n)))

    def validate_no_free_variables(self) -> None:
        """
        Raise an error if not all variables in the constraints are bound to a
        variable in the context in which it is applied.
        """
        context = self.instance()
        for constraint in context.constraints():
            if not all(v.wildcard or v in context
                    for v in constraint.variables(indirect=False)):
                raise ConstrainFreeVariable(constraint, self)

    def only_schematic(self) -> bool:
        """
        Return `True` if all variables occuring in the schema are wildcards or
        schematic variables.
        """
        variables = [TypeVariable() for _ in range(self.n)]
        return all((v in variables or v.wildcard)
            for v in self.schema(*variables).variables())


class TypeOperator(Type):
    """
    An n-ary type constructor. If 0-ary, can also be treated as an instance of
    the corresponding type operation (that is, a base type).
    """

    def __init__(self,
            name: Optional[str] = None,
            params: int | list[Variance] = 0,
            supertype: Optional[TypeOperator] = None):
        self._name = name
        self.supertype: Optional[TypeOperator] = supertype
        self.subtypes: set[TypeOperator] = set()
        self.variance = list(Variance.CO for _ in range(params)) \
            if isinstance(params, int) else list(params)
        self.arity = len(self.variance)
        self.depth: int = supertype.depth + 1 if supertype else 0

        if self.supertype and self.arity > 0:
            raise ValueError("only nullary types can have explicit supertypes")
        if supertype:
            supertype.subtypes.add(self)

    def __str__(self) -> str:
        return self._name or object.__repr__(self)

    def __call__(self, *params: Type) -> TypeOperation:
        return TypeOperation(self, *(p.instance() for p in params))

    def subtype(self, other: TypeOperator, strict: bool = False) -> bool:
        assert isinstance(other, TypeOperator)
        return ((not strict and self == other) or
            bool(self.supertype and self.supertype.subtype(other)))

    def normalize(self) -> TypeInstance:
        return self.instance()

    def instance(self) -> TypeInstance:
        return TypeOperation(self)

    @property
    def name(self) -> str:
        if self._name is None:
            raise RuntimeError("Unnamed operator.")
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        if value is None or (self._name is not None and value != self._name):
            raise RuntimeError(
                f"Cannot name operator {value}; already named {self._name}.")
        self._name = value


class TypeInstance(Type):
    """
    Base class for type instances (type operations and -variables). Note that
    base types are just 0-ary type operators and functions are just particular
    2-ary type operators.
    """

    def normalized(self) -> bool:
        return not (isinstance(self, TypeVariable) and self.unification)

    def __str__(self):
        return self.text(with_constraints=True)

    def nesting(self) -> int:
        """
        The maximum nesting level of type parameters.
        """
        if isinstance(self, TypeVariable):
            raise ValueError("Attempted to calculate nesting for variable")
        assert isinstance(self, TypeOperation)
        return max(p.nesting() for p in self.params) + 1 if self.params else 0

    def text(self,
            labels: dict[TypeVariable, str] = Labels("τ", subscript=True),
            sep: str = ", ",
            lparen: str = "(",
            rparen: str = ")",
            arrow: str = " → ",
            prod: str = " × ",
            with_constraints: bool = False) -> str:
        """
        Convert the given type to a textual representation.
        """

        args = labels, sep, lparen, rparen, arrow, prod

        if isinstance(self, TypeOperation):
            if self._operator == Function:
                i, o = self.params
                if isinstance(i, TypeOperation) and i._operator == Function:
                    result = f"{lparen}{i.text(*args)}{rparen}" \
                        f"{arrow}{o.text(*args)}"
                else:
                    result = f"{i.text(*args)}{arrow}{o.text(*args)}"
            elif self._operator == Product and prod:
                a, b = self.params
                result = f"{lparen}{a.text(*args)}{prod}{b.text(*args)}{rparen}"
            else:
                if self.params:
                    result = f"{self._operator}{lparen}" \
                        f"{sep.join(t.text(*args) for t in self.params)}{rparen}"
                else:
                    result = str(self._operator)
        else:
            assert isinstance(self, TypeVariable)
            if self.unification:
                result = self.unification.text(*args)
            elif self.wildcard:
                result = "_"
            else:
                try:
                    result = labels[self]
                except KeyError:
                    # TODO the fact that this is necessary exposes a
                    # fundamental flaw: sometimes variables that are the same
                    # nevertheless get different labels
                    result = ""
                    for var in labels:
                        if var.follow() is self:
                            result = labels[var]
                            break
                    assert result

        if with_constraints:
            result_aux = [result]
            for v in self.variables():
                if v.lower:
                    result_aux.append(f"{v.text(*args)} ≥ {v.lower}")
                if v.upper:
                    result_aux.append(f"{v.text(*args)} ≤ {v.upper}")

            result_aux.extend(
                c.text(*args) for c in self.constraints())
            return ' | '.join(result_aux)
        else:
            return result

    def fix(self, prefer_lower: bool = True) -> TypeInstance:
        """
        Obtain a version of this type with all eligible variables with subtype
        constraints fixed to their most specific type.
        """
        a = self.follow()

        if isinstance(a, TypeOperation):
            for v, p in zip(a._operator.variance, a.params):
                p.fix(prefer_lower ^ (v == Variance.CONTRA))
        elif isinstance(a, TypeVariable):
            if prefer_lower and a.lower:
                a.bind(a.lower())
            elif not prefer_lower and a.upper:
                a.bind(a.upper())
        return a.follow()

    def normalize(self) -> TypeInstance:
        """
        Ensure that all variables in a type are followed to their binding.

        Be aware that that normalizing a type instance is recommended before
        storing it in a set, using it as a dict key or otherwise hashing it.
        This is because variables are not automatically followed to their
        binding when hashing, and so a type `F(x)` may look the same and even
        be equal to `F(y)` when `y` is bound to `x`, and yet not have the same
        hash. As a result, binding a variable after hashing will also cause
        issues.
        """
        a = self.follow()
        if isinstance(a, TypeOperation):
            a.params = tuple(p.normalize() for p in a.params)
        return a

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

    def variables(self, indirect: bool = True,
            target: Optional[set[TypeVariable]] = None) -> set[TypeVariable]:
        """
        Obtain all distinct type variables currently in the type instance, and
        optionally also those variables indirectly related via constraints.
        """
        direct_variables = set(t for t in self
            if isinstance(t, TypeVariable) and (not target or t not in target))

        if indirect:
            target = target or set()
            target.update(direct_variables)
            for v in direct_variables:
                for c in v._constraints:
                    for e in chain(c.reference, *c.alternatives):
                        e.variables(indirect=True, target=target)
            return target
        else:
            if target:
                target.update(direct_variables)
            return target or direct_variables

    def operators(self, indirect: bool = True) -> set[TypeOperator]:
        """
        Obtain all distinct non-function operators in the type instance.
        """
        result = set(t._operator for t in self
            if isinstance(t, TypeOperation) and t._operator != Function)
        if indirect:
            for constraint in self.constraints(indirect=True):
                result.update(*(p.operators(indirect=False) for p in
                    chain(constraint.reference, *constraint.alternatives)))
        return result

    def constraints(self, indirect: bool = True) -> set[Constraint]:
        """
        Obtain all constraints attached to variables in the type instance.
        """
        result = set()
        for v in self.variables(indirect=indirect):
            result.update(v._constraints)
        return result

    def follow(self) -> TypeInstance:
        """
        Follow a unification until bumping into a type that is not yet bound.
        """
        a = self
        while isinstance(a, TypeVariable) and a.unification:
            a = a.unification
        return a

    def match(self, other: TypeInstance, subtype: bool = False,
            accept_wildcard: bool = False) -> Optional[bool]:
        """
        Return True if self is definitely the same as (or a subtype of) other,
        False if it is definitely not, and None if there is not enough
        information. Note that constraints are not taken into account!
        """
        a = self.follow()
        b = other.follow()

        if isinstance(a, TypeOperation) and isinstance(b, TypeOperation):
            if a.basic:
                return a._operator == b._operator or \
                    (subtype and a._operator.subtype(b._operator))
            elif a._operator != b._operator:
                return False
            else:
                result: Optional[bool] = True
                for v, s, t in zip(a._operator.variance, a.params, b.params):
                    r = TypeInstance.match(
                        *((s, t) if v == Variance.CO else (t, s)),
                        subtype=subtype, accept_wildcard=accept_wildcard)
                    if r is False:
                        return False
                    elif r is None:
                        result = None
                return result
        elif isinstance(a, TypeOperation) and isinstance(b, TypeVariable):
            if (b.upper or b.lower) and not a.basic:
                return False
            if b.upper and b.upper.subtype(a._operator, strict=True):
                return False
            if b.lower and (b.lower.subtype(a._operator) is False):
                return False
            if accept_wildcard and b.wildcard:
                return True
        elif isinstance(a, TypeVariable) and isinstance(b, TypeOperation):
            if (a.upper or a.lower) and not b.basic:
                return False
            if a.lower and b._operator.subtype(a.lower, strict=True):
                return False
            if a.upper and (a.upper.subtype(b._operator) is False):
                return False
            if accept_wildcard and a.wildcard:
                return True
        elif isinstance(a, TypeVariable) and isinstance(b, TypeVariable):
            if a == b or (a.wildcard and b.wildcard):
                return True
            if accept_wildcard and (a.wildcard or b.wildcard):
                return True
            if a.lower and b.upper and b.upper.subtype(a.lower, strict=True):
                return False
        return None

    def unify(self, other: TypeInstance, subtype: bool = False,
            skeletal: bool = False) -> None:
        """
        Make sure that a is equal to, or a subtype of b. Like normal
        unification, but instead of just a substitution of variables, also
        produces new variables with subtype- and supertype bounds.

        Skeletal unifications will not unify any variables to base types that
        have subtypes, so as to avoid fixing an overly general type.
        """
        a = self.follow()
        b = other.follow()

        if isinstance(a, TypeVariable) and isinstance(b, TypeVariable):
            a.bind(b)
        elif isinstance(a, TypeOperation) and isinstance(b, TypeOperation):
            if a.basic:
                subtype = subtype or skeletal
                if subtype and not a._operator.subtype(b._operator):
                    raise SubtypeMismatch(a._operator, b._operator)
                elif not subtype and a._operator != b._operator:
                    raise TypeMismatch(a, b)
            elif a._operator == b._operator:
                for v, x, y in zip(a._operator.variance, a.params, b.params):
                    if v == Variance.CO:
                        x.unify(y, subtype=subtype, skeletal=skeletal)
                    else:
                        assert v == Variance.CONTRA
                        y.unify(x, subtype=subtype, skeletal=skeletal)
            else:
                raise TypeMismatch(a, b)
        elif isinstance(a, TypeVariable) and isinstance(b, TypeOperation):
            if a in b:
                raise RecursiveType(a, b)
            elif b.basic:
                if not (skeletal and b._operator.subtypes):
                    if subtype:
                        a.below(b._operator)
                    else:
                        a.bind(b)
            else:
                skeleton = b._operator(*(TypeVariable() for _ in b.params))
                a.bind(skeleton)
                a.unify(b, subtype=subtype, skeletal=skeletal)
        else:
            assert isinstance(a, TypeOperation) and isinstance(b, TypeVariable)
            if b in a:
                raise RecursiveType(b, a)
            elif a.basic:
                if not (skeletal and a._operator.subtypes):
                    if subtype:
                        b.above(a._operator)
                    else:
                        b.bind(a)
            else:
                skeleton = a._operator(*(TypeVariable() for _ in a.params))
                b.bind(skeleton)
                a.unify(b, subtype=subtype, skeletal=skeletal)

    def instance(self) -> TypeInstance:
        return self.follow()

    def output(self) -> TypeInstance:
        """
        Obtain the final uncurried output type of a given function type.
        """
        if isinstance(self, TypeOperation) and self._operator is Function:
            return self.params[1].output()
        else:
            return self

    @property
    def operator(self) -> Optional[TypeOperator]:
        return self._operator if isinstance(self, TypeOperation) else None

    def __rrshift__(self, other: Iterable[Type]) -> Type:
        """
        Abuse of Python's notation to go along with __getitem__. Allows an
        alternative method to specify constraints upfront. Example:

            >>> TypeSchema(lambda x, y: {x[A]} >> F(x) ** x)

        This also allows you to set constraints on type combinations that don't
        occur in that configuration in the context:

            >>> TypeSchema(lambda x, y: {(x * B)[A * y]} >> F(x) ** y)
        """
        return self

    def __getitem__(self, value: Type | tuple[Type]) -> TypeInstance:
        """
        Another abuse of Python's notation, allowing us to add constraints to a
        type by using indexing notation. Example:

            >>> TypeSchema(lambda x, y: F(x[A]) ** x)
        """

        # Constraints are attached to the relevant variables as the constraint
        # is instantiated
        Constraint(self, (value,) if isinstance(value, Type) else value)
        return self

class TypeOperation(TypeInstance):
    """
    An instance of an n-ary type constructor.
    """

    def __init__(self, op: TypeOperator, *params: TypeInstance):
        self._operator = op
        self.params = tuple(params)

        if len(self.params) != self._operator.arity:
            raise ValueError(
                f"{self._operator} takes {self._operator.arity} "
                f"parameter{'' if self._operator.arity == 1 else 's'}; "
                f"{len(self.params)} given"
            )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TypeInstance) and bool(self.match(other))

    def __hash__(self) -> int:
        """
        Hash this type. Be aware that variables are not normalized, so a type
        `F(A)` may look and act the same (including __eq__) and yet not have
        the same hash as a type `F(x)` in which `x` is bound to `A`. Call
        `.normalize()` before hashing to avoid issues.
        """
        return hash((self._operator,) + self.params)

    @property
    def basic(self) -> bool:
        return self._operator.arity == 0


class TypeVariable(TypeInstance):
    """
    A type variable. This is not a schematic variable — it is instantiated!
    """

    def __init__(self, wildcard: bool = False, origin=None):
        self.wildcard = wildcard
        self.unification: Optional[TypeInstance] = None
        self.lower: Optional[TypeOperator] = None
        self.upper: Optional[TypeOperator] = None
        self._constraints: set[Constraint] = set()
        self.origin = origin

    def bind(self, t: TypeInstance) -> None:
        assert not self.unification, "variable cannot be unified twice"

        # Once a wildcard variable is bound, it is no longer wildcard
        self.wildcard = False

        if self is not t:
            self.unification = t

            if isinstance(t, TypeVariable):
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
                    if self.lower and t._operator.subtype(self.lower, True):
                        raise SubtypeMismatch(t._operator, self.lower)
                    if self.upper and self.upper.subtype(t._operator, True):
                        raise SubtypeMismatch(self.upper, t._operator)
                else:
                    variables = t.variables(indirect=False)

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
        a = self.follow()
        assert isinstance(a, TypeVariable)
        lower, upper = a.lower or new, a.upper or new
        if upper.subtype(new, True):  # fail when lower bound higher than upper
            raise SubtypeMismatch(new, upper)
        elif new.subtype(lower, True):  # ignore lower bound lower than current
            pass
        elif lower.subtype(new):  # tighten the lower bound
            a.lower = new
            a.check_constraints()
        else:  # fail on bound from other lineage (neither sub- nor supertype)
            raise SubtypeMismatch(lower, new)

    def below(self, new: TypeOperator) -> None:
        """
        Constrain this variable to be a basic type with the given subtype as
        upper bound.
        """
        # symmetric to `above`
        a = self.follow()
        assert isinstance(a, TypeVariable)
        lower, upper = a.lower or new, a.upper or new
        if new.subtype(lower, True):
            raise SubtypeMismatch(lower, new)
        elif upper.subtype(new, True):
            pass
        elif new.subtype(upper):
            a.upper = new
            a.check_constraints()
        else:
            raise SubtypeMismatch(new, upper)

    def check_constraints(self) -> None:
        self._constraints = set(
            c for c in self._constraints if not c.fulfilled())


class TypeAlias(Type):
    """
    A type alias is a type synonym. It cannot contain variables, but it may be
    parameterized. By default, any type alias is canonical, which means that it
    has some special significance among the potentially infinite number of
    types.
    """

    def __init__(self, alias: Type | Callable[..., TypeOperation], *args: Type,
            canonical: bool = True, name: str | None = None):

        self.name = name
        self.canonical = canonical
        self.args = args
        self.arity = len(args)
        self.alias: TypeOperation | Callable[..., TypeOperation]

        if self.arity == 0:
            assert isinstance(alias, Type)
            alias = alias.instance()
            assert isinstance(alias, TypeOperation)
            self.alias = alias
        else:
            assert callable(alias) and \
                self.arity == len(signature(alias).parameters)
            self.alias = alias

        if any(self.instance().variables()):
            raise RuntimeError("Type alias must not contain variables.")

    def instance(self) -> TypeOperation:
        if self.arity > 0:
            assert callable(self.alias)
            return self.alias(*self.args)
        else:
            assert isinstance(self.alias, TypeOperation)
            return self.alias

    def __call__(self, *args: Type) -> TypeOperation:
        if len(args) == 0:
            assert isinstance(self.alias, TypeOperation)
            return self.alias
        else:
            assert callable(self.alias)
            return self.alias(*args)

            # TODO do we want this?
            # for p, q in zip(args, self.args):
            #     p.unify(q, subtype=True)


class Constraint(object):
    """
    A constraint enforces that its reference is a subtype of one of the given
    alternatives.
    """

    def __init__(
            self,
            reference: TypeInstance,
            alternatives: Iterable[Type]):
        self.reference = reference
        self.alternatives = list(a.instance() for a in alternatives)
        self.check = True

        # Inform variables about the constraint present on them
        for v in self.variables():
            assert not v.unification
            v._constraints.add(self)

        self.fulfilled()

    def text(self, *args, **kwargs) -> str:
        return (
            f"{self.reference.text(*args, **kwargs)} << ["
            f"{', '.join(a.text(*args, **kwargs) for a in self.alternatives)}]"
        )

    def variables(self, indirect: bool = True) -> set[TypeVariable]:
        result: set[TypeVariable] = set()
        for e in chain(self.reference, *self.alternatives):
            result.update(e.variables(indirect=indirect, target=result))
        return result

    def minimize(self) -> None:
        """
        Ensure that constraints don't contain extraneous alternatives:
        alternatives that are equal to, or more specific versions of, other
        alternatives.
        """
        minimized: list[TypeInstance] = []
        for obj in self.alternatives:
            add = True
            for i in range(len(minimized)):
                if minimized[i].match(obj, subtype=True):
                    minimized[i] = obj.follow()
                if obj.match(minimized[i], subtype=True):
                    add = False
            if add:
                minimized.append(obj.follow())
        self.reference = self.reference.follow()
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

        assert self.reference.normalized() and all(p.normalized() for p in
                self.alternatives)

        compatibility = [
            self.reference.match(t, subtype=True, accept_wildcard=True)
            for t in self.alternatives]
        self.alternatives = [t for t, c
            in zip(self.alternatives, compatibility) if c is not False]

        if len(self.alternatives) == 0:
            raise ConstraintViolation(self)

        # If there is only one possibility left, we can unify, but *only*
        # in a way that makes sure that base types with subtypes remain
        # variable, because we don't want to fix an overly loose subtype.
        elif self.check and len(self.alternatives) == 1:
            self.check = False
            self.reference.unify(self.alternatives[0], subtype=True,
                skeletal=True)
            self.check = True

        # Fulfillment is achieved if the reference is fully concrete and there
        # is at least one definitely compatible alternative
        # TODO
        return (not any(self.reference.variables()) and any(compatibility)) \
            or self.reference in self.alternatives


"The special constructor for function types."
Function = TypeOperator('Function', params=[Variance.CONTRA, Variance.CO])

"The special constructor for tuple (or product) types."
Product = TypeOperator('Product', params=2)

"The special constructor for the unit type."
Unit = TypeOperator('Unit')

"A wildcard: fresh variable, unrelated to, and matchable with, anything else."
_ = TypeSchema(lambda: TypeVariable(wildcard=True))


def with_parameters(
        *type_operators: TypeOperator | Callable[..., TypeOperation],
        param: Optional[Type] = None,
        at: int = None) -> list[TypeInstance]:
    """
    Generate a list of instances of type operations. Optionally, the generated
    type operations must contain a certain parameter (at some index, if given).
    """
    options: list[TypeInstance] = []
    for op in type_operators:
        if isinstance(op, TypeOperator):
            arity = op.arity
        else:
            arity = len(signature(op).parameters)
        for i in ([at - 1] if at else range(arity)) if param else [-1]:
            if i < arity:
                options.append(op(*(
                    param if param and i == j else _ for j in range(arity)
                )))
    return options


# Errors #####################################################################

class TypingError(Exception):
    "There is a typechecking issue."


class TypeMismatch(TypingError):
    "Raised when compound types cannot be unified."

    def __init__(self, left: Type, right: Type):
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f"Could not unify type `{self.left}` with `{self.right}`."


class SubtypeMismatch(TypeMismatch):
    "Raised when base types are not subtypes of eachother."

    def __init__(self, left: TypeOperator, right: TypeOperator):
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f"Could not satisfy subtype `{self.left}` <= `{self.right}`."


class FunctionApplicationError(TypeMismatch):
    "Raised when an argument is passed to a non-function type."

    def __str__(self) -> str:
        return f"Could not apply non-function type `{self.left}` to " \
            f"`{self.right}`."


class RecursiveType(TypingError):
    "Raised for infinite types."

    def __init__(self, inner: TypeInstance, outer: TypeInstance):
        self.inner = inner
        self.outer = outer

    def __str__(self) -> str:
        return f"Encountered the recursive type {self.inner}~{self.outer}."


class ConstraintViolation(TypingError):
    "Raised when there remains no way in which a constraint can be satisfied."

    def __init__(self, constraint: Constraint):
        self.constraint = constraint

    def __str__(self) -> str:
        return f"Violated typeclass constraint {self.constraint}."


class ConstrainFreeVariable(TypingError):
    """
    Raised when a constraint refers to a variable that does not occur in its
    context.
    """

    def __init__(self, constraint: Constraint, context: TypeSchema):
        self.constraint = constraint
        self.context = context

    def __str__(self) -> str:
        return f"A free variable occurs in constraint {self.constraint}"
