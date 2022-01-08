"""
This module allows you to define a transformation language as a collection of
types and operators. It also handles parsing expressions of the algebra.
"""

from __future__ import annotations

from itertools import groupby, product
from typing import Optional, Iterator, Any, TYPE_CHECKING

from transformation_algebra.type import \
    TypeOperator, TypeInstance, TypeVariable, TypeOperation, Variance
from transformation_algebra.expr import \
    Operator, Expr, Application, Source

if TYPE_CHECKING:
    from rdflib import Namespace


class Language(object):
    def __init__(self, scope: dict[str, Any] = {},
            namespace: Namespace | None = None):
        self.operators: dict[str, Operator] = dict()
        self.types: dict[str, TypeOperator] = dict()
        self.synonyms: dict[str, TypeOperation] = dict()

        self._namespace = namespace
        if scope:
            self.add_scope(scope)

    def taxonomy(self) -> dict[TypeOperation, TypeOperation | None]:
        """
        Generate a taxonomy of canonical types, mapping each of them to their
        supertype. The canonical types consist of all base types, plus those
        compound types that have an explicit synonym, or that are subtypes of
        types that do.

        These types are of special interest among the potentially infinite
        number of types.
        """
        # Start with the taxonomy of base types
        taxonomy: dict[TypeOperation, TypeOperation | None] = {
            op(): op.supertype() if op.supertype else None
            for op in self.types.values()
            if not op.arity
        }

        # Extract the direct sub- or supertypes from the growing taxonomy
        def successors(t: TypeInstance, sub: bool) -> Iterator[TypeOperation]:
            assert isinstance(t, TypeOperation)
            yield t
            if sub:
                yield from (subtype for subtype, supertype in taxonomy.items()
                    if supertype == t)
            elif supertype := taxonomy[t]:
                yield supertype

        # By sorting on depth, we get a guarantee that, by the time we are
        # adding a type of depth n, we know all successor types of depth n-1.
        stack = sorted(self.synonyms.values(), key=TypeInstance.depth)
        while stack:
            t = stack.pop()
            subtypes = (t._operator(*params) for params in product(*(
                successors(p, v is Variance.CO)
                for v, p in zip(t._operator.variance, t.params)
            )))

            for s in subtypes:
                if s != t and s not in taxonomy:
                    taxonomy[s] = t
                    stack.append(s)
        return taxonomy

    @property
    def namespace(self) -> Namespace:
        if self._namespace is None:
            raise RuntimeError("No associated namespace.")
        else:
            return self._namespace

    def add_scope(self, scope: dict[str, Any]) -> None:
        """
        For convenience, you may add types and operations in bulk via a
        dictionary. This allows you to simply pass `globals()` or `locals()`.
        Irrelevant items are filtered out without complaint.
        """
        for k, v in dict(scope).items():
            if isinstance(v, (Operator, TypeOperator)) or (
                    isinstance(v, TypeOperation) and not any(v.variables())):
                self.add(item=v, name=k)

    def add(self, item: Operator | TypeOperator | TypeOperation,
            name: str | None = None):

        # The item must already have a name or be named here
        if isinstance(item, (Operator, TypeOperator)):
            if name:
                name = item.name = name.rstrip("_")
            else:
                name = item.name
        elif not name:
            raise ValueError("no name provided for synonym")

        if name in self:
            raise ValueError(f"symbol {name} already exists in the language")

        if isinstance(item, TypeOperation) and any(item.variables()):
            raise ValueError("synonyms must not contain variable instance")

        if isinstance(item, Operator):
            self.operators[name] = item
        elif isinstance(item, TypeOperator):
            self.types[name] = item
        else:
            self.synonyms[name] = item

    def __getattr__(self, name: str) -> Operator | TypeOperator:
        result = self.operators.get(name) or self.types.get(name)
        if result:
            return result
        else:
            raise AttributeError

    def __setattr__(self, name: str, value: Operator | TypeOperator) -> None:
        if isinstance(value, (Operator, TypeOperator)):
            self.add(value, name)
        else:
            super().__setattr__(name, value)

    def __contains__(self, key: str | Operator | TypeOperator) -> bool:
        if isinstance(key, str):
            return key in self.operators or key in self.types \
                or key in self.synonyms
        elif isinstance(key, TypeOperator):
            assert isinstance(key.name, str)
            return self.types.get(key.name) is key
        elif isinstance(key, Operator):
            assert isinstance(key.name, str)
            return self.operators.get(key.name) is key
        else:
            return False

    def validate(self) -> None:
        # Validation can only happen once all operations have been defined. If
        # we did it at define-time, it would lead to issues --- see issue #3

        for op in self.operators.values():
            # Check declared types of operations against their inferred type
            op.validate()

            # Check that every type is known to the algebra
            for t in op.type.instance().operators():
                if t not in self:
                    raise ValueError

    def parse(self, string: str) -> Expr:
        # This used to be done via pyparsing, but the structure is so simple
        # that I opted to remove the dependency --- this is *much* faster

        return self.parse_expr(string)

    def parse_expr(self, val: str | Iterator[str]) -> Expr:
        tokens = tokenize(val, "(,):;~") if isinstance(val, str) else val
        stack: list[Expr | None] = [None]
        sources: dict[int, Source] = dict()

        while token := next(tokens, None):

            if token in "(,)":
                if token in "),":
                    try:
                        y = stack.pop()
                        if y:
                            x = stack.pop()
                            stack.append(Application(x, y) if x else y)
                    except IndexError as e:
                        raise BracketMismatch('(') from e
                if token in "(,":
                    stack.append(None)
            elif token == ":":
                previous = stack[-1]
                assert isinstance(previous, Expr)
                t = self.parse_type(tokens)
                previous.type.unify(t, subtype=True)
                previous.type.fix(prefer_lower=False)
            elif token == ";":
                stack.clear()
                stack.append(None)
            elif token == "~":
                previous = stack.pop()
                if previous:
                    stack.append(previous)
                t = self.parse_type(tokens)
                stack.append(Source(type=t))

            else:
                current: Optional[Expr]
                previous = stack.pop()
                if previous and isinstance(previous, Source):
                    current = None

                    # TODO Deprecated: labelling a source by suffixing it with
                    # "xn". Will be removed in next version.
                    assert token[0] == 'x'
                    n = int(token[1:])

                    try:
                        previous = sources[n]
                    except KeyError:
                        assert not previous.label
                        previous.label = n
                        sources[n] = previous
                    stack.append(previous)
                    continue
                if token == "-":
                    current = Source()
                elif token.isnumeric():
                    n = int(token)
                    try:
                        current = sources[n]
                    except KeyError:
                        current = sources[n] = Source(label=n)
                else:
                    try:
                        current = self.operators[token].instance()
                    except KeyError as e:
                        raise UndefinedToken(token) from e
                if previous and current:
                    current = Application(previous, current)
                stack.append(current)

        if len(stack) == 1:
            result = stack[0]
            if not result:
                raise EmptyParse
            else:
                return result
        else:
            raise BracketMismatch(")")

        raise NotImplementedError

    def parse_type(self, value: str | Iterator[str]) -> TypeInstance:
        tokens = tokenize(value, "(,)") if isinstance(value, str) else value
        stack: list[TypeInstance | TypeOperator | list[TypeInstance]] = []

        while token := next(tokens, None):
            if token == "(":
                stack.append([])
            elif token in "),":
                unit = stack.pop()
                assert isinstance(unit, TypeInstance)
                params = stack.pop()
                assert isinstance(params, list)
                params.append(unit)
                stack.append(params)
                if token == ")":
                    params = stack.pop()
                    assert isinstance(params, list)
                    op = stack.pop()
                    assert isinstance(op, TypeOperator)
                    stack.append(op(*params))
            elif token == "_":
                stack.append(TypeVariable())
            else:
                t: TypeInstance | TypeOperator
                try:
                    op = self.types[token]
                    t = op() if op.arity == 0 else op
                except KeyError:
                    try:
                        t = self.synonyms[token]
                    except KeyError as e:
                        raise UndefinedToken(token) from e
                stack.append(t)
            if len(stack) == 1 and isinstance(stack[0], TypeInstance):
                final = stack.pop()
                assert isinstance(final, TypeInstance)
                return final
        raise RuntimeError


def tokenize(string: str, specials: str = "") -> Iterator[str]:
    """
    Break up a string into special characters and identifiers (that is, any
    sequence of non-special characters). Skip spaces.
    """
    for group, tokens in groupby(string,
            lambda x: -2 if x.isspace() else specials.find(x)):
        if group == -1:
            yield "".join(tokens)
        elif group >= 0:
            yield from tokens


# Errors #####################################################################

class ParseError(Exception):
    pass


class BracketMismatch(ParseError):
    def __str__(self) -> str:
        return "Mismatched bracket."


class EmptyParse(ParseError):
    def __str__(self) -> str:
        return "Empty parse."


class UndefinedToken(ParseError):
    def __init__(self, token: str):
        self.token = token

    def __str__(self) -> str:
        return f"Operator or type operator '{self.token}' is undefined."
