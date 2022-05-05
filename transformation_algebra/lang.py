"""
This module allows you to define a transformation language as a collection of
types and operators. It also handles parsing expressions of the algebra.
"""

from __future__ import annotations

from itertools import groupby, count, chain
from typing import Optional, Iterator, Any, TYPE_CHECKING
from rdflib.namespace import ClosedNamespace

from transformation_algebra.type import Variance, \
    TypeOperator, TypeInstance, TypeVariable, TypeOperation, TypeAlias, Product
from transformation_algebra.expr import \
    Operator, Expr, Application, Source

if TYPE_CHECKING:
    from rdflib import URIRef

# Map all types to their direct subtypes
Taxonomy = dict[TypeOperation, set[TypeOperation]]


class Language(object):
    def __init__(self, scope: dict[str, Any] = {},
            namespace: str | None = None):
        self.operators: dict[str, Operator] = dict()
        self.types: dict[str, TypeOperator] = dict()
        self.synonyms: dict[str, TypeAlias] = dict()

        self._closed = False
        self._taxonomy: Taxonomy | None = None
        self._namespace: LanguageNamespace | str | None = namespace

        if scope:
            self.add_scope(scope)

    @property
    def taxonomy(self) -> Taxonomy:
        if not self._taxonomy:
            self._taxonomy = self.generate_taxonomy()
            self._closed = True
        return self._taxonomy

    @property
    def namespace(self) -> LanguageNamespace:
        if self._namespace is None:
            raise RuntimeError("No associated namespace.")
        elif not isinstance(self._namespace, LanguageNamespace):
            assert isinstance(self._namespace, str)
            self._namespace = LanguageNamespace(self._namespace, self)
            self._closed = True

        return self._namespace

    def generate_taxonomy(self) -> Taxonomy:
        """
        Generate a taxonomy of canonical types, mapping each of them to their
        subtypes. The canonical types consist of all base types, plus those
        compound types that have an explicit synonym, or that are subtypes or
        parameters of types that do.

        These types are of special interest among the potentially infinite
        number of types.
        """
        taxonomy: dict[TypeOperation, set[TypeOperation]] = dict()

        # Start with the taxonomy of base types
        for op in sorted(self.types.values(), key=lambda op: op.depth):
            if op.arity == 0:
                t = op()
                taxonomy[t] = taxonomy.get(t, set())
                if op.supertype:
                    taxonomy[op.supertype()].add(t)

        # By sorting on nesting level, we get a guarantee that by the time we
        # add a type of level n, we know all successor types of level n-1.
        stack: list[TypeOperation] = sorted(
            (s.instance() for s in self.synonyms.values() if s.canonical),
            key=TypeInstance.nesting, reverse=True)
        while stack:
            t = stack.pop()
            if t not in taxonomy:

                # Any non-basic type that occurs as parameter of a canonical
                # type must itself be canonical, and it must already in the
                # taxonomy; if it is not, we handle it before going forward
                for p in t.params:
                    if p not in taxonomy:
                        assert isinstance(p, TypeOperation)
                        stack.append(t)
                        stack.append(p)
                        break
                else:
                    taxonomy[t] = set()
                    for i, v, p in zip(count(), t._operator.variance, t.params):
                        assert isinstance(p, TypeOperation)
                        p_successors = taxonomy[p] if v is Variance.CO else (
                            supertype
                            for supertype, subtypes in taxonomy.items()
                            if any(s.match(t) for s in subtypes))

                        for p_succ in p_successors:
                            q = t._operator(*(p_succ if i == j else p_orig
                                for j, p_orig in enumerate(t.params)))
                            taxonomy[t].add(q)
                            stack.append(q)

        return taxonomy

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

        if self._closed:
            raise RuntimeError("Cannot add to language after closing.")

        # The item must already have a name or be named here
        if name:
            name = item.name = name.rstrip("_")
        elif item.name:
            name = item.name
        else:
            raise RuntimeError("unnamed")

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
                    raise ValueError

    def parse(self, string: str, *args: Expr) -> Expr:
        # This used to be done via pyparsing, but the structure is so simple
        # that I opted to remove the dependency --- this is *much* faster

        return self.parse_expr(string, *args)

    def parse_expr(self, val: str | Iterator[str], *args: Expr) -> Expr:
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
                previous.type.unify(t, subtype=True)
                # previous.type.fix(prefer_lower=False)
            elif token == ";":
                stack.clear()
                stack.append(None)
            elif token == "~":
                previous = stack.pop()
                t = self.parse_type(tokens)
                current_: Expr = Source(type=t)
                if previous:
                    current_ = Application(previous, current_)
                stack.append(current_)

            else:
                current: Optional[Expr]
                if token.isnumeric():
                    try:
                        current = args[int(token) - 1]
                    except KeyError:
                        raise RuntimeError(
                            f"{token} should be replaced with expression, but "
                            f"none was given")
                else:
                    current = self.parse_operator(token).instance()
                previous = stack.pop()
                if previous:
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

    def parse_atom(self, token: str) -> Operator | TypeInstance:
        try:
            return self.parse_operator(token)
        except UndefinedToken:
            return self.parse_type(token)

    def parse_operator(self, token: str) -> Operator:
        try:
            return self.operators[token]
        except KeyError as e:
            raise UndefinedToken(token) from e

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
                        raise RuntimeError
                elif isinstance(arg, (TypeOperator, TypeAlias)):
                    if not len(args) == arg.arity:
                        raise RuntimeError(
                            f"Tried to apply {len(args)} arguments to "
                            f"operator {arg} of arity {arg.arity}")
                    stack.append(arg(*reversed(args)))
                    args = []
                else:
                    args.append(arg)
            raise RuntimeError(args)

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
                        raise UndefinedToken(token) from e
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
            raise RuntimeError(stack)


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
            lang.operators.keys(),
            lang.types.keys(),
            (t.text(sep=".", lparen="_", rparen="", prod="")
                for t in lang.taxonomy)
        )
        rt = super().__new__(cls, uri, terms)
        cls.lang = lang
        return rt

    def term(self, value) -> URIRef:
        if isinstance(value, (Operator, TypeOperator)):
            return super().term(value.name)
        elif isinstance(value, TypeOperation):
            assert value in self.lang.taxonomy
            return super().term(value.text(sep=".", lparen="_", rparen="",
                prod=""))
        else:
            return super().term(value)


# Errors #####################################################################

class ParseError(RuntimeError):
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
