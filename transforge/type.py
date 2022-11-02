"""
Generic type system. Inspired loosely by Hindley-Milner type inference in
functional programming languages, as well as Traytel et al (2011).
"""
from __future__ import annotations

from enum import Enum, auto
from abc import ABC, abstractmethod
from functools import reduce
from itertools import chain, count
from inspect import signature
from typing import Optional, Iterator, Iterable, Callable

from transforge.label import Labels


class Direction(Enum):
    UP = auto()
    DOWN = auto()

    def variant(self, variance: Variance) -> Direction:
        if variance is Variance.CO:
            return self
        elif self is Direction.UP:
            return Direction.DOWN
        else:
            return Direction.UP


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
        Product (tuple) type operator.
        """
        # See issue #78
        return Product(self.instance(), other.instance())

    def __getitem__(self, cs: Constraint | tuple[Constraint]) -> TypeInstance:
        """
        An abuse of Python's indexing notation, which allows us to add
        constraints to type.

            >>> TypeSchema(lambda x: F(x) ** x [x < A])
        """

        # Constraints are attached to the relevant variables as the constraint
        # is instantiated
        for c in cs if isinstance(cs, tuple) else (cs,):
            if not isinstance(c, Constraint):
                raise ValueError("Indexing notation only accepts constraints.")
        return self.instance()

    def __lshift__(self,
            other: Type | Iterable[Type]) -> EliminationConstraint:
        """
        Abuse of Python's notation to write elimination constraints using the
        `<<` operator. Example:

            >>> TypeSchema(lambda x, y: F(x) ** y [x * y << {A * B, B * C}])
            >>> TypeSchema(lambda x: x [x << A])
        """
        return EliminationConstraint(self.instance(),
            (other,) if isinstance(other, Type) else other)

    def __lt__(self, other: Type) -> SubtypeConstraint:
        """
        Write strict subtype constraints using the `<` operator.
        """
        return SubtypeConstraint(self.instance(), other.instance(),
            strict=True)

    def __le__(self, other: Type) -> SubtypeConstraint:
        """
        Write subtype constraints using the `<=` operator.
        """
        return SubtypeConstraint(self.instance(), other.instance())

    def is_subtype(self, other: Type, strict: bool = False) -> Optional[bool]:
        """
        Test whether one type is a subtype of another. Return `None` if the
        answer depends on the binding of variables.
        """
        a, b = self.instance(), other.instance()
        return a.match(b, accept_wildcard=True, subtype=True) \
            and (not strict or not a.match(b))

    def apply(self, arg: Type) -> TypeInstance:
        """
        Apply an argument to a function type to get its output type.
        """

        f = self.instance().follow()
        x = arg.instance().follow()

        if isinstance(f, TypeVariable):
            f.bind(Function(TypeVariable(), TypeVariable()))
            f = f.follow()

        if isinstance(f, TypeOperation) and f.operator == Function:
            x.unify(f.params[0], subtype=True)
            left, right = f.params
            left.fix(prefer_lower=False)
            # f.fix()
            if isinstance(right, TypeOperation) and right.operator == Function:
                return right
            else:
                return right.fix()
        elif isinstance(f, TypeOperation) and f.operator == Top:
            return Top()
        else:
            raise FunctionApplicationError(f, x)

    @abstractmethod
    def instance(self) -> TypeInstance:
        return NotImplemented

    def concretize(self, replace: bool = False) -> TypeOperation:
        """
        Make sure that this type contains no variables. Optionally replace
        unconstrained variables with the catch-all `Top` type.
        """
        a = self.instance().follow()
        if isinstance(a, TypeOperation):
            return a.operator(*(p.concretize(replace) for p in a.params))
        elif replace and isinstance(a, TypeVariable) and not a._constraints:
            return Top()
        else:
            raise UnexpectedVariableError(
                "Types containing constrained variables cannot be concretized")


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
        return self.schema(*variables).fix(prefer_lower=True).text(
            with_constraints=True,
            labels={v: k for k, v in zip(names, variables)})

    def instance(self, origin=None) -> TypeInstance:
        return self.schema(*(TypeVariable(origin=origin)
            for _ in range(self.n))).fix(prefer_lower=True)

    def validate_no_free_variables(self) -> None:
        """
        Raise an error if not all variables in the constraints are bound to a
        variable in the context in which it is applied.
        """
        context = self.instance()
        for constraint in context.constraints():
            if not all(v.wildcard or v in context
                    for v in constraint.variables(indirect=False)):
                raise ConstrainFreeVariableError(constraint, self)

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
        self.parent: Optional[TypeOperator] = supertype
        self.children: set[TypeOperator] = set()
        self.variance = list(Variance.CO for _ in range(params)) \
            if isinstance(params, int) else list(params)
        self.arity = len(self.variance)
        self.depth: int = supertype.depth + 1 if supertype else 0

        if self.parent and self.arity > 0:
            raise ValueError("only nullary types can have explicit supertypes")
        if supertype:
            supertype.children.add(self)

    def __str__(self) -> str:
        return self._name or object.__repr__(self)

    def __call__(self, *params: Type) -> TypeOperation:
        return TypeOperation(self, *(p.instance() for p in params))

    def subtype(self, other: TypeOperator, strict: bool = False) -> bool:
        assert isinstance(other, TypeOperator)
        return (not strict and self is other) or (
            self is Bottom or other is Top or
            bool(self.parent and self.parent.subtype(other))
        )

    def floor(self) -> Iterator[TypeOperation]:
        if self.arity == 0:
            if self.children:
                yield from chain.from_iterable(c.floor()
                    for c in self.children)
            else:
                yield TypeOperation(self)
        else:
            yield TypeOperation(self, *(
                Bottom() if v == Variance.CO else Top() for v in self.variance
            ))

    def ceiling(self) -> Iterator[TypeOperation]:
        if self.arity == 0:
            if self.parent:
                yield from self.parent.ceiling()
            else:
                yield TypeOperation(self)
        else:
            yield TypeOperation(self, *(
                Top() if v == Variance.CO else Bottom() for v in self.variance
            ))

    def normalize(self) -> TypeInstance:
        return self.instance()

    def instance(self) -> TypeInstance:
        return TypeOperation(self)

    @property
    def name(self) -> str:
        if self._name is None:
            raise ValueError("Type operator must have a name.")
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        if value is None or (self._name is not None and value != self._name):
            raise ValueError(
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
            raise ValueError("A variable can not have a nesting level.")
        assert isinstance(self, TypeOperation)
        return max(p.nesting() for p in self.params) + 1 if self.params else 0

    def text(self,
            labels: dict[TypeVariable, str] = Labels("τ"),
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
            if self.operator == Function:
                i, o = self.params
                if isinstance(i, TypeOperation) and i.operator == Function:
                    result = f"{lparen}{i.text(*args)}{rparen}" \
                        f"{arrow}{o.text(*args)}"
                else:
                    result = f"{i.text(*args)}{arrow}{o.text(*args)}"
            elif self.operator == Product and prod:
                a, b = self.params
                result = f"{lparen}{a.text(*args)}{prod}{b.text(*args)}{rparen}"
            else:
                if self.params:
                    result = f"{self.operator}{lparen}" \
                        f"{sep.join(t.text(*args) for t in self.params)}{rparen}"
                else:
                    result = str(self.operator)
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
            result_aux = []
            for v in self.variables():
                if v.lower:
                    result_aux.append(f"{v.text(*args)} >= {v.lower}")
                if v.upper:
                    result_aux.append(f"{v.text(*args)} <= {v.upper}")

            result_aux.extend(
                c.text(*args) for c in self.constraints())
            if result_aux:
                result += f" [{', '.join(result_aux)}]"
        return result

    def fix(self, prefer_lower: bool = True) -> TypeInstance:
        """
        Obtain a version of this type with all eligible variables with subtype
        constraints fixed to their most specific type.
        """
        a = self.follow()

        if isinstance(a, TypeOperation):
            for v, p in zip(a.operator.variance, a.params):
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
                for constraint in v._constraints:
                    for e in constraint:
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
        result = set(t.operator for t in self
            if isinstance(t, TypeOperation) and t.operator != Function)
        if indirect:
            for constraint in self.constraints(indirect=True):
                result.update(*(p.operators(indirect=False) for p in
                    constraint))
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
            if subtype and (a.operator is Bottom or b.operator is Top):
                return True
            elif a.basic:
                return a.operator == b.operator or \
                    (subtype and a.operator.subtype(b.operator))
            elif a.operator != b.operator:
                return a.operator is Bottom or b.operator is Top
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
        elif isinstance(a, TypeOperation) and isinstance(b, TypeVariable):
            if subtype and a.operator is Bottom:
                return True
            if (b.upper or b.lower) and not a.basic:
                return False
            if b.upper and b.upper.subtype(a.operator, strict=True):
                return False
            if b.lower and (b.lower.subtype(a.operator) is False):
                return False
            if accept_wildcard and b.wildcard:
                return True
        elif isinstance(a, TypeVariable) and isinstance(b, TypeOperation):
            if subtype and b.operator is Top:
                return True
            if (a.upper or a.lower) and not b.basic:
                return False
            if a.lower and b.operator.subtype(a.lower, strict=True):
                return False
            if a.upper and (a.upper.subtype(b.operator) is False):
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
            skip_basic: bool = False, skip_wildcard: bool = False) -> None:
        """
        Make sure that a is equal to, or a subtype of b. Like normal
        unification, but instead of just a substitution of variables, also
        produces new variables with subtype- and supertype bounds.

        Optionally, unifying base types or wildcard variables can be skipped.
        """
        a = self.follow()
        b = other.follow()

        if isinstance(a, TypeVariable) and isinstance(b, TypeVariable):
            if not skip_wildcard or not (a.wildcard and b.wildcard):
                a.bind(b)
        elif isinstance(a, TypeOperation) and isinstance(b, TypeOperation):
            if a.operator is Bottom or b.operator is Top:
                return
            elif a.basic:
                if skip_basic:
                    pass
                elif subtype and not a.operator.subtype(b.operator):
                    raise SubtypeMismatch(a.operator, b.operator)
                elif not subtype and a.operator != b.operator:
                    raise TypeMismatch(a, b)
            elif a.operator == b.operator:
                for v, x, y in zip(a.operator.variance, a.params, b.params):
                    if v == Variance.CO:
                        x.unify(y, subtype=subtype, skip_basic=skip_basic,
                            skip_wildcard=skip_wildcard)
                    else:
                        assert v == Variance.CONTRA
                        y.unify(x, subtype=subtype, skip_basic=skip_basic,
                            skip_wildcard=skip_wildcard)
            else:
                raise TypeMismatch(a, b)
        elif isinstance(a, TypeVariable) and isinstance(b, TypeOperation):
            if b.operator is Top:
                return
            elif a in b:
                raise RecursiveTypeError(a, b)
            elif b.basic:
                if skip_basic or (skip_wildcard and a.wildcard):
                    pass
                elif subtype:
                    a.below(b.operator)
                else:
                    a.bind(b)
            else:
                if skip_wildcard or skip_basic:
                    a.bind(b.operator(*(TypeVariable() for _ in b.params)))
                    a.unify(b, subtype=subtype, skip_basic=skip_basic,
                        skip_wildcard=skip_wildcard)
                else:
                    a.bind(b)
        else:
            assert isinstance(a, TypeOperation) and isinstance(b, TypeVariable)
            if a.operator is Bottom:
                return
            elif b in a:
                raise RecursiveTypeError(b, a)
            elif a.basic:
                if skip_basic or (skip_wildcard and b.wildcard):
                    pass
                elif subtype:
                    b.above(a.operator)
                else:
                    b.bind(a)
            else:
                if skip_wildcard or skip_basic:
                    b.bind(a.operator(*(TypeVariable() for _ in a.params)))
                    b.unify(b, subtype=subtype, skip_basic=skip_basic,
                        skip_wildcard=skip_wildcard)
                else:
                    b.bind(a)

    @staticmethod
    def common(types: Iterable[TypeInstance]) -> TypeInstance | None:
        """
        Find the common structure between multiple types.
        """
        ftypes = [t.follow() for t in types]
        if not ftypes:
            return None
        operators = [t.operator for t in ftypes if isinstance(t, TypeOperation)]
        operator = operators[0]

        if operator is None or not all(o == operator for o in operators):
            return None
        else:
            return operator(*(
                TypeInstance.common(t.params[i] for t in ftypes) or _
                for i in range(operator.arity)
            ))

    def instance(self) -> TypeInstance:
        return self.follow()

    def output(self) -> TypeInstance:
        """
        Obtain the final uncurried output type of a given function type.
        """
        if isinstance(self, TypeOperation) and self.operator is Function:
            return self.params[1].output()
        else:
            return self


class TypeOperation(TypeInstance):
    """
    An instance of an n-ary type constructor.
    """

    def __init__(self, op: TypeOperator, *params: TypeInstance):
        self.operator = op
        self.params = tuple(params)

        if len(self.params) != self.operator.arity:
            raise TypeParameterError(
                f"{self.operator} takes {self.operator.arity} "
                f"parameter{'' if self.operator.arity == 1 else 's'}; "
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
        return hash((self.operator,) + self.params)

    @property
    def basic(self) -> bool:
        return self.operator.arity == 0

    def successors(self, dir: Direction,
            include_custom: bool = True,
            include_bottom: bool = False,
            include_top: bool = False,
            universe: Iterable[TypeOperator] = ()) \
            -> Iterator[TypeOperation]:
        """
        Find successor types (ie. direct subtypes or supertypes of this type).

        The universe consists of the types that you can get to by travelling
        down from the Top or up from the Bottom. However, if there is a
        parameterized type present, this results in infinite types when
        travelling recursively.
        """
        op = self.operator

        if op.arity == 0:
            if dir is Direction.DOWN:
                if op is Top:
                    if universe:
                        for t in universe:
                            yield from t.ceiling()
                    elif include_bottom:
                        yield Bottom()
                elif include_custom and op.children:
                    yield from (c() for c in op.children)
                elif include_bottom and op is not Bottom:
                    yield Bottom()
            else:
                assert dir is Direction.UP
                if op is Bottom:
                    if universe:
                        for t in universe:
                            yield from t.floor()
                    elif include_top:
                        yield Top()
                elif include_custom and op.parent:
                    yield op.parent()
                elif include_top and op is not Top:
                    yield Top()
        else:
            empty = True
            for i, v, p in zip(count(), op.variance, self.params):
                if isinstance(p, TypeOperation):
                    for q in p.successors(dir.variant(v),
                            include_custom=include_custom,
                            include_top=include_top,
                            include_bottom=include_bottom,
                            universe=universe):
                        empty = False
                        yield op(*(q if i == j else p
                            for j, p in enumerate(self.params)))
                else:
                    raise UnexpectedVariableError(
                        "Only concrete types can have parents or children."
                    )

            if empty:
                if include_bottom and dir is Direction.DOWN:
                    yield Bottom()
                elif include_top and dir is Direction.UP:
                    yield Top()


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

    def check_constraints(self) -> None:
        for c in list(self._constraints):
            if c.fulfill():
                try:
                    self._constraints.remove(c)
                except KeyError:
                    pass

    def bind(self, t: TypeInstance) -> None:
        assert not self.unification, "variable cannot be unified twice"

        self.wildcard = False  # once bound, lose wildcard status
        if self is not t:
            self.unification = t

            if isinstance(t, TypeVariable):
                t._constraints.update(self._constraints)
                self._constraints = t._constraints

                t.wildcard = False

                if self.lower:
                    t.above(self.lower)
                if self.upper:
                    t.below(self.upper)

                if t.lower == t.upper and t.lower is not None:
                    t.bind(t.lower())

            elif isinstance(t, TypeOperation):
                if t.basic:
                    if self.lower and t.operator.subtype(self.lower, True):
                        raise SubtypeMismatch(t.operator, self.lower)
                    if self.upper and self.upper.subtype(t.operator, True):
                        raise SubtypeMismatch(self.upper, t.operator)
                else:
                    variables = t.variables(indirect=False)

                    self._constraints.update(chain(*(
                        v._constraints for v in variables
                    )))
                    for v in variables:
                        v._constraints = self._constraints

            self.check_constraints()

    def above(self, new: TypeOperator) -> None:
        """
        Constrain this variable to be a basic type with the given type as lower
        bound.
        """
        if new is Top:
            self.bind(Top())
            return
        self.wildcard = False  # once constrained, lose wildcard status
        a = self.follow()
        assert isinstance(a, TypeVariable)
        lower, upper = a.lower or new, a.upper or new
        if upper.subtype(new, True):  # fail when lower bound exceeds upper
            raise SubtypeMismatch(new, upper)
        elif not new.subtype(upper):  # new type must be in same 'family line'
            raise SubtypeMismatch(new, upper)
        elif new.subtype(lower, True):  # ignore lower bound lower than current
            pass
        elif lower.subtype(new):  # tighten the lower bound
            a.lower = new
            a.check_constraints()
        else:  # fail on bound from other lineage (neither sub- nor supertype)
            raise SubtypeMismatch(lower, new)
        if a.lower and a.lower == a.upper and a.lower is not None:
            self.bind(a.lower())  # if A <= x <= A, immediately bind x to A

    def below(self, new: TypeOperator) -> None:
        """
        Constrain this variable to be a basic type with the given subtype as
        upper bound.
        """
        # symmetric to `above`
        if new is Bottom:
            self.bind(Bottom())
            return
        self.wildcard = False
        a = self.follow()
        assert isinstance(a, TypeVariable)
        lower, upper = a.lower or new, a.upper or new
        if new.subtype(lower, True):
            raise SubtypeMismatch(lower, new)
        elif not lower.subtype(upper):
            raise SubtypeMismatch(new, upper)
        elif upper.subtype(new, True):
            pass
        elif new.subtype(upper):
            a.upper = new
            a.check_constraints()
        else:
            raise SubtypeMismatch(new, upper)
        if a.upper and a.upper == a.lower and a.upper is not None:
            self.bind(a.upper())


class TypeAlias(Type):
    """
    A type alias is a type synonym. It cannot contain variables, but it may be
    parameterized. By default, any type alias is canonical, which means that it
    has some special significance among the potentially infinite number of
    types.
    """

    def __init__(self, alias: Type | Callable[..., TypeOperation],
            name: str | None = None):

        self.name = name
        self.alias: Type | Callable[..., TypeOperation] = alias
        self.arity = 0 if isinstance(alias, Type) else \
            len(signature(alias).parameters)

        if isinstance(alias, Type) and any(alias.instance().variables()):
            raise UnexpectedVariableError(
                f"Type alias {name} must not contain variables")

    def instance(self) -> TypeInstance:
        if self.arity > 0:
            raise TypeParameterError(
                f"Type alias {self.name} cannot be used as a type "
                "without providing parameters")
        else:
            assert isinstance(self.alias, Type)
            return self.alias.instance()

    def __call__(self, *args: Type) -> TypeOperation:
        assert callable(self.alias)
        if len(args) == self.arity:
            return self.alias(*args)
        else:
            raise TypeParameterError(
                f"Type alias {self.name} requires {self.arity} parameters, "
                f"but {len(args)} were given")


class Constraint(object):
    @abstractmethod
    def __init__(self):
        self.fulfilled = False
        self.inform()
        self.fulfill()

    @abstractmethod
    def __iter__(self) -> Iterator[TypeInstance]:
        return NotImplemented

    @abstractmethod
    def text(self, *args, **kwargs) -> str:
        return NotImplemented

    @abstractmethod
    def fulfill(self) -> bool:
        """
        Check that the constraint has not been violated and raise an error
        otherwise. Additionally, return `True` if it has been completely
        fulfilled and needs not be enforced any longer.
        """
        return NotImplemented

    def variables(self, indirect: bool = True) -> set[TypeVariable]:
        result: set[TypeVariable] = set()
        for e in self:
            result.update(e.variables(indirect=indirect, target=result))
        return result

    def inform(self) -> None:
        """
        Inform relevant variables that this constraint is present.
        """
        for v in self.variables():
            assert not v.unification
            v._constraints.add(self)


class SubtypeConstraint(Constraint):
    """
    This constraint enforces that a type variable, or a type operation
    containing any, can only ever be fixed to a subtype of a certain type.
    """
    # This is different from a lower or upper bound on type variables. Lower
    # and upper bounds are used for fixing type operators to type variables in
    # the presence of subtyping; these never fix variables to anything, but
    # only act as guard to make sure that, once they a variable is fixed, it is
    # a subtype of the given type operator.

    def __init__(self, reference: Type, target: Type, strict: bool = False):
        self.reference = reference.instance()
        self.target = target.instance()
        self.strict = strict
        super().__init__()

    def __iter__(self) -> Iterator[TypeInstance]:
        yield from chain(self.reference, self.target)

    def text(self, *args, **kwargs) -> str:
        symbol = '<' if self.strict else '<='
        return (
            f"{self.reference.text(*args, **kwargs)} "
            f"{symbol} {self.target.text(*args, **kwargs)}"
        )

    def fulfill(self) -> bool:
        self.reference.unify(self.target, subtype=True, skip_basic=True)
        result = self.reference.match(self.target, subtype=True)
        if result is True:
            self.fulfilled = True
        elif result is False:
            raise ConstraintViolation(self)
        return self.fulfilled


class EliminationConstraint(Constraint):
    """
    An elimination constraint offers a choice between multiple alternatives:
    different possible unifications. Once only one alternative remains
    possible, the unification is performed.
    """

    def __init__(
            self,
            reference: TypeInstance,
            alternatives: Iterable[Type]):
        self.reference = reference
        self.alternatives = list(a.instance() for a in alternatives)
        super().__init__()

    def text(self, *args, **kwargs) -> str:
        return (
            f"{self.reference.text(*args, **kwargs)} << ("
            f"{', '.join(a.text(*args, **kwargs) for a in self.alternatives)})"
        )

    def __iter__(self) -> Iterator[TypeInstance]:
        yield from chain(self.reference, *self.alternatives)

    def minimize(self) -> None:
        """
        Ensure that constraints don't contain extraneous alternatives:
        alternatives that are equal to, or more specific versions of, other
        alternatives.
        """
        # TODO more *general* versions
        minimized: list[TypeInstance] = []
        for obj in self.alternatives:
            add = True
            for i in range(len(minimized)):
                if minimized[i].match(obj, subtype=True):
                    minimized[i] = obj.follow()
                if obj.match(minimized[i], subtype=True):
                    add = False
            if add:
                minimized.append(obj.follow().fix())
        self.reference = self.reference.follow()
        self.alternatives = minimized

    def fulfill(self) -> bool:
        if self.fulfilled:
            return True

        # TODO Minimization is expensive and it is performed on every variable
        # binding. Make this more efficient?
        self.minimize()

        assert self.reference.normalized() and \
            all(p.normalized() for p in self.alternatives)

        # number_before = len(self.alternatives)
        self.alternatives = [t for t in self.alternatives
            if self.reference.match(t, subtype=True, accept_wildcard=True)
            is not False]
        # number_after = len(self.alternatives)

        # Every time alternatives are narrowed, see if there's a commonality
        # between the remaining alternatives
        # if number_after < number_before or True:
        #     common = TypeInstance.common(self.alternatives)
        #     if common:
        #         self.reference.unify(common, skip_wildcard=True)
        #     self.minimize()

        if len(self.alternatives) == 0:
            raise ConstraintViolation(self)

        # If there is only one possibility left, we can unify, but *only*
        # in a way that makes sure that base types with subtypes remain
        # variable, because we don't want to fix an overly loose subtype.
        elif len(self.alternatives) == 1:
            self.fulfilled = True
            self.reference.unify(self.alternatives[0], subtype=True)

        return self.fulfilled


"The special constructor for function types."
Function = TypeOperator('Function', params=[Variance.CONTRA, Variance.CO])

"The special constructor for tuple (or product) types."
Product = TypeOperator('Product', params=2)

"The special constructor for the unit type."
Unit = TypeOperator('Unit')

"The bottom type contains no values and is a subtype to everything."
Bottom = TypeOperator('Bottom')

"The top type contains all values and is a supertype to everything."
Top = TypeOperator('Top')

"A wildcard: fresh variable, unrelated to, and matchable with, anything else."
_ = TypeSchema(lambda: TypeVariable(wildcard=True))

builtins = (Unit, Top, Bottom, Product, Function)


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
    "Raised when types cannot be unified."

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


class RecursiveTypeError(TypingError):
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


class ConstrainFreeVariableError(TypingError):
    """
    Raised when a constraint refers to a variable that does not occur in its
    context.
    """

    def __init__(self, constraint: Constraint, context: TypeSchema):
        self.constraint = constraint
        self.context = context

    def __str__(self) -> str:
        return f"A free variable occurs in constraint {self.constraint}"


class UnexpectedVariableError(TypingError):
    """
    Raised when a variable is encountered in what should be a concrete type.
    """


class TypeParameterError(TypingError):
    """
    Raised when too many or not enough parameters are passed to a type
    operator, or when an instance is taken of a parameterized type.
    """
