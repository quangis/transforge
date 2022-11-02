"""
This module allows you to define a transformation language as a collection of
types and operators. It also handles parsing expressions of the algebra.
"""

from __future__ import annotations

from itertools import groupby, chain
from typing import Optional, Iterator, Any, Iterable
from rdflib import URIRef
from rdflib.namespace import ClosedNamespace

from transforge.namespace import TF, EX
from transforge.type import (builtins, Product, TypeOperator,
    TypeInstance, TypeVariable, TypeOperation, TypeAlias, Direction, Type,
    TypeSchema, TypingError, Top, Bottom)
from transforge.expr import Operator, Expr, Application, Source


class Language(object):
    def __init__(self, scope: dict[str, Any] = {},
            namespace: tuple[str, str] | str | None = None,
            canon: Iterable[TypeOperator | TypeOperation] = ()):
        """
        The canonical types consist of all base types, plus those compound
        types that have an explicit synonym, or that are subtypes or parameters
        of types that do. These types are of special interest among the
        potentially infinite number of types.
        """
        self.operators: dict[str, Operator] = dict()
        self.types: dict[str, TypeOperator] = dict()
        self.synonyms: dict[str, TypeAlias] = dict()
        self.include_top: bool = Top in canon
        self.include_bottom: bool = Bottom in canon

        self._closed = False
        self.prefix: None | str = None
        self._namespace: LanguageNamespace | str | None = None
        if isinstance(namespace, tuple):
            self.prefix, self._namespace = namespace
        elif isinstance(namespace, str):
            self._namespace = namespace

        if scope:
            self.add_scope(scope)

        self.canon: set[TypeOperation] = set()
        if canon:
            for t in canon:
                if isinstance(t, TypeOperator):
                    if t not in (Top, Bottom):
                        self.canon.add(t())
                else:
                    assert isinstance(t, TypeOperation)
                    self.canon.add(t)
        else:
            self.canon = set(op()
                for op in self.types.values() if op.arity == 0)
        self.expand_canon()

    @property
    def namespace(self) -> LanguageNamespace:
        if self._namespace is None:
            self._namespace = LanguageNamespace(EX, self)
        elif not isinstance(self._namespace, LanguageNamespace):
            assert isinstance(self._namespace, str)
            self._namespace = LanguageNamespace(self._namespace, self)
            self._closed = True

        return self._namespace

    def uri(self, x: Operator | Type) -> URIRef:
        """
        Convert an operator or a canonical type to its associated URI.
        """

        # concretize `F(x)` to `F(Top)` for unconstrained variables
        if ((isinstance(x, TypeOperation) and any(x.variables(indirect=False)))
                or isinstance(x, (TypeVariable, TypeSchema))):
            x = x.concretize(replace=True)

        # handle base (type) operators
        if isinstance(x, TypeOperation) and x.operator.arity == 0:
            x = x.operator
        if isinstance(x, (Operator, TypeOperator)):
            return (TF if x in builtins else self.namespace)[x.name]

        # handle compound operators
        if x in self.canon:
            assert isinstance(x, TypeOperation)
            return self.namespace[
                x.text(sep="-", lparen="-", rparen="", prod="")]
        else:
            raise NonCanonicalTypeError(x)

    def expand_canon(self) -> None:
        """
        Expand the canonical types to include all subtypes of existing
        canonical types. If appropriate, supertypes like `Top`, `F(Top)`,
        etcetera, are also included.
        """
        # TODO this should be done automatically; make a CanonicalTypes object

        stack: list[TypeOperation] = list(self.canon)
        while stack:
            current = stack.pop()
            self.canon.add(current)
            for s in current.successors(Direction.UP,
                    include_custom=False,
                    include_top=self.include_top,
                    include_bottom=self.include_bottom):
                if s not in self.canon:
                    stack.append(s)
            for s in current.successors(Direction.DOWN,
                    include_top=self.include_top,
                    include_bottom=self.include_bottom):
                if s not in self.canon:
                    stack.append(s)

    def successors(self, d: Direction, t: TypeOperation,
            transitive: bool) -> Iterator[TypeOperation]:
        """
        Find canonical successor types of `t`.
        """
        assert t in self.canon, f"{t} is not canonical"

        for s in t.successors(d,
                include_top=self.include_top,
                include_bottom=self.include_bottom,
                universe=self.types.values()):
            if s in self.canon:
                yield s
                if transitive:
                    yield from self.successors(d, s, True)
            # It might be the case that a type that is noncanonical has a
            # canonical supertype, e.g. A is a supertype of B, with F(B)
            # canonical, then F(Top) is canonical but F(A) is not
            else:
                for u in s.successors(d,
                        include_top=self.include_top,
                        include_bottom=self.include_bottom,
                        universe=self.types.values()):
                    if u in self.canon:
                        yield u
                        if transitive:
                            yield from self.successors(d, u, True)

    def subtypes(self, t: TypeOperation,
            transitive: bool = False) -> Iterator[TypeOperation]:
        return self.successors(Direction.DOWN, t, transitive)

    def supertypes(self, t: TypeOperation,
            transitive: bool = False) -> Iterator[TypeOperation]:
        return self.successors(Direction.UP, t, transitive)

    def add_scope(self, scope: dict[str, Any]) -> None:
        """
        For convenience, you may add types and operations in bulk via a
        dictionary. This allows you to simply pass `globals()` or `locals()`.
        Irrelevant items are filtered out without complaint.
        """
        for k, v in dict(scope).items():
            if isinstance(v, (Operator, TypeOperator, TypeAlias)):
                self.add(item=v, name=k)

    def add(self, item: Operator | TypeOperator | TypeAlias,
            name: str | None = None):

        if item in builtins:
            return

        reserved = ("via", "type", "signature", "expression",
            "Unit", "Top", "Bottom", "Product", "Intersection", "Union")

        if self._closed:
            raise RuntimeError("Cannot add to language after closing.")

        # The item must already have a name or be named here
        if name:
            name = item.name = name.rstrip("_")
        elif item.name:
            name = item.name
        else:
            raise ValueError("Unnamed operator")

        if name in self or name in reserved:
            raise ValueError(f"Symbol {name} already exists in the language")

        if isinstance(item, Operator):
            self.operators[name] = item
        elif isinstance(item, TypeOperator):
            self.types[name] = item
        else:
            self.synonyms[name] = item

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
        self._closed = True

        for op in self.operators.values():
            # Check declared types of operations against their inferred type
            op.validate()

            # Check that every type is known to the algebra
            for t in op.type.instance().operators():
                if t not in self:
                    raise ValueError(f"Operator {op} contains unknown type {t}")

    def parse(self, string: str, *args: Expr) -> Expr:
        # This used to be done via pyparsing, but the structure is so simple
        # that I opted to remove the dependency --- this is *much* faster

        return self.parse_expr(string, *args)

    def parse_expr(self, val: str | Iterator[str], *args: Expr) -> Expr:
        previous_token = ""
        tokens = tokenize(val, "*(,):;~") if isinstance(val, str) else val
        stack: list[Expr | None] = [None]

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
                pt = previous.type
                try:
                    if isinstance(pt, TypeVariable) and pt.wildcard:
                        pt.unify(t)
                    else:
                        pt.unify(t, subtype=True)
                except TypingError as e:
                    if previous_token.isnumeric():
                        input = int(previous_token)
                    else:
                        input = None
                    raise TypeAnnotationError(previous, t, input) from e
            elif token == ";":
                stack.clear()
                stack.append(None)
            else:
                current: Optional[Expr]
                if token == "-":
                    current = Source()
                elif token.isnumeric():
                    input = int(token)
                    try:
                        current = args[input - 1]
                    except KeyError:
                        raise MissingInputError(input)
                else:
                    current = self.parse_operator(token).instance()
                previous = stack.pop()
                if previous:
                    current = Application(previous, current)
                stack.append(current)

            previous_token = token

        if len(stack) == 1:
            result = stack[0]
            if not result:
                raise EmptyParse
            else:
                return result
        else:
            raise BracketMismatch(")")

        raise NotImplementedError

    def parse_atom(self, token: str) -> Operator | TypeInstance:
        try:
            return self.parse_operator(token)
        except UndefinedTokenError:
            return self.parse_type(token)

    def parse_operator(self, token: str) -> Operator:
        try:
            return self.operators[token]
        except KeyError as e:
            raise UndefinedTokenError(token) from e

    def parse_type(self, value: str | Iterator[str]) -> TypeInstance:
        if isinstance(value, str):
            consume_all = True
            tokens = tokenize(value, "*(,)")
        else:
            consume_all = False
            tokens = value

        stack: list[None | TypeInstance | TypeOperator | TypeAlias] = [None]

        def backtrack():
            args: list[TypeInstance] = []
            while stack:
                arg = stack.pop()
                if arg is None:
                    if len(args) == 1 and isinstance(args[0], TypeInstance):
                        stack.append(args[0])
                        return
                    else:
                        raise ParseError("Could not parse type instance")
                elif isinstance(arg, (TypeOperator, TypeAlias)):
                    stack.append(arg(*reversed(args)))
                    args = []
                else:
                    args.append(arg)
            raise BracketMismatch

        level = 0
        while token := next(tokens, None):
            if token == "(":
                stack.append(None)
                level += 1
            elif token in "),":
                backtrack()
                if token == ")":
                    level -= 1
                else:
                    stack.append(None)
            elif token == "_":
                stack.append(TypeVariable())
            elif token == "*":
                t1 = stack.pop()
                assert isinstance(t1, TypeInstance)
                stack.append(Product)
                stack.append(t1)
            else:
                t: TypeInstance | TypeOperator | TypeAlias
                try:
                    op = self.types[token]
                    t = op() if op.arity == 0 else op
                except KeyError:
                    try:
                        t = self.synonyms[token]
                        t = t.instance() if t.arity == 0 else t
                    except KeyError as e:
                        raise UndefinedTokenError(token) from e
                stack.append(t)

            if not consume_all and (
                    isinstance(stack[1], TypeInstance) or (
                        isinstance(stack[1], (TypeOperator, TypeAlias))
                        and level == 0 and len(stack) > 2)):
                break

        backtrack()

        if len(stack) == 1 and isinstance(stack[0], TypeInstance):
            return stack[0]
        else:
            raise ParseError("Could not parse as type instance")


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


class LanguageNamespace(ClosedNamespace):
    """
    A algebra-aware namespace for rdflib. That is, it allows URIs to be written
    as `NS[f]` for an operation or base type `f`. It is also closed: it fails
    when referencing URIs for types or operations that are not part of the
    relevant transformation algebra.
    """

    def __new__(cls, uri, lang: Language):
        terms = chain(
            ("type", "via", "signature", "expression"),
            lang.operators.keys(),
            lang.types.keys(),
            (t.text(sep="-", lparen="-", rparen="", prod="")
                for t in lang.canon)
        )
        rt = super().__new__(cls, uri, terms)
        cls.lang = lang
        return rt

    def term(self, value) -> URIRef:
        if isinstance(value, (Operator, TypeOperator)):
            return super().term(value.name)
        elif isinstance(value, TypeOperation):
            assert value in self.lang.canon
            return super().term(value.text(sep="-", lparen="-", rparen="",
                prod=""))
        else:
            return super().term(value)


# Errors #####################################################################

class ParseError(Exception):
    pass


class BracketMismatch(ParseError):
    def __str__(self) -> str:
        return "Mismatched bracket."


class EmptyParse(ParseError):
    def __str__(self) -> str:
        return "Empty parse."


class UndefinedTokenError(ParseError):
    def __init__(self, token: str):
        self.token = token

    def __str__(self) -> str:
        return f"Operator or type operator '{self.token}' is undefined."


class MissingInputError(ParseError):
    "Raised when an input is referenced but no input was provided as argument."


class TypeAnnotationError(TypingError):
    def __init__(self, expr: Expr, type: Type, input: int | None = None):
        self.expr = expr
        self.type = type
        self.input = input

    def __str__(self) -> str:
        assert self.__cause__, "must be caused by another error"
        if self.input:
            source = f"Input #{self.input}"
        elif isinstance(self.expr, Source):
            source = "The anonymous source"
        else:
            source = f"The expression `{self.expr}`"

        return (
            f"{source} was declared `{self.type}`, "
            f"but it is actually `{self.expr.type}`. \n"
            f"\t{self.__cause__}"
        )


class NonCanonicalTypeError(Exception):
    "Raised when a non-canonical type is referenced"

    def __init__(self, type: Type):
        self.type = type
