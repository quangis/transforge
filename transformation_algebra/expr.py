"""
Classes to define generic transformation algebras.
"""

from __future__ import annotations

from abc import ABC
import pyparsing as pp
from functools import reduce, partial
from inspect import signature
from typing import Optional, Any, Dict, Callable, Union

from transformation_algebra import error
from transformation_algebra.type import \
    Type, TypeSchema, TypeInstance, Function, _


class Definition(ABC):
    """
    A definition represents a non-instantiated data input or transformation.
    """

    def __init__(
            self,
            type: Union[Type, Callable[..., Type]],
            doc: Optional[str] = None):
        self.name: Optional[str] = None
        self.type = type if isinstance(type, Type) else TypeSchema(type)
        self.description = doc

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"{self.name or 'anonymous'} : {self.type}"

    def __call__(self, *args: Union[Definition, PartialExpr]) -> PartialExpr:
        return self.instance().__call__(*args)

    def instance(self, identifier: Optional[str] = None) -> Expr:
        return Base(self, label=identifier)


class Data(Definition):
    """
    The definition of a data input. An instance of such a definition is a base
    expression.
    """

    def __init__(self, *nargs, **kwargs):
        self.composition = None
        super().__init__(*nargs, **kwargs)
        assert not self.type.is_function()


class Operation(Definition):
    """
    The definition of a transformation. An instance of such a definition is a
    base expression.
    """

    def __init__(
            self, *nargs,
            derived: Optional[Callable[..., PartialExpr]] = None, **kwargs):
        self.composition = derived  # some transformations may be non-primitive
        super().__init__(*nargs, **kwargs)
        assert self.type.is_function()

        # If the operation is composite, check that its declared type is no
        # more general than the type we can infer from the composition function
        if self.composition:
            mock = Definition(_)
            args = [Base(mock) for a in signature(self.composition).parameters]
            output = self.composition(*args)
            if isinstance(output, Expr):
                declared_type = self.type.instance()
                inferred_type = output.type
                nvars = sum(1 for v in declared_type.variables())
                for a in reversed(args):
                    inferred_type = Function(a.type, inferred_type)
                try:
                    declared_type.unify(inferred_type, subtype=True)
                    declared_type = declared_type.resolve()
                except error.AlgebraTypeError:
                    raise error.DefinitionTypeMismatch(
                        self, self.type, inferred_type)
                else:
                    # If some variables we declared were unified, we know that
                    # the inferred type is more specific than the declared type
                    if sum(1 for v in declared_type.variables()) != nvars:
                        raise error.DefinitionTypeMismatch(
                            self, self.type, inferred_type)

            else:
                # The type could not be derived because the result is not a
                # full expression. Shouldn't happen?
                raise RuntimeError


class PartialExpr(ABC):
    """
    An expression that may contain abstractions.
    """

    def __call__(self, *args: Union[PartialExpr, Definition]) -> PartialExpr:
        a = (e.instance() if isinstance(e, Definition) else e for e in args)
        return reduce(PartialExpr.partial_apply, a, self).complete()

    def partial_apply(self, x: PartialExpr) -> PartialExpr:
        f = self.complete()
        if isinstance(f, Abstraction):
            f.composition = partial(f.composition, x)
            return f.complete()
        elif isinstance(x, Abstraction):
            raise RuntimeError(
                "cannot apply abstraction to primitive expression")
        else:
            assert isinstance(f, Expr) and isinstance(x, Expr)
            return Application(f, x)

    def complete(self) -> PartialExpr:
        """
        Once all parameters of an abstraction have been provided, turn into
        full expression.
        """
        if isinstance(self, Abstraction):
            n = len(signature(self.composition).parameters)
            if n == 0:
                return self.composition()
        return self


class Expr(PartialExpr):
    def __init__(self, type: TypeInstance):
        self.type = type

    def __repr__(self) -> str:
        return self.tree()

    def tree(self, lvl: str = "") -> str:
        """
        Obtain a tree representation using Unicode block characters.
        """
        if isinstance(self, Base):
            return f"╼ {self} : {self.definition.type}"
        else:
            assert isinstance(self, Application)
            return (
                f"{self.type.strc()}\n"
                f"{lvl} ├─{self.f.tree(lvl + ' │ ')}\n"
                f"{lvl} └─{self.x.tree(lvl + '   ')}"
            )

    def substitute(self, label: str, expr: Expr) -> Expr:
        """
        Replace the given expression for all expressions with the given label.
        """
        if isinstance(self, Base):
            if self.label == label:
                self.type.plain().unify(expr.type.plain())
                return expr
            else:
                return self
        elif isinstance(self, Application):
            self.f = self.f.substitute(label, expr)
            self.x = self.x.substitute(label, expr)
            return self
        raise ValueError

    def primitive(self) -> Expr:
        """
        Expand this expression into its simplest form.
        """
        f = self.partial_primitive()
        if isinstance(f, Abstraction):
            raise RuntimeError("cannot express partial primitive")
        elif isinstance(f, Expr):
            return f
        raise ValueError

    def partial_primitive(self) -> PartialExpr:
        """
        Expand this expression into its simplest form. May contain
        abstractions.
        """
        if isinstance(self, Base):
            d = self.definition
            if isinstance(d, Operation) and d.composition:
                return Abstraction(d.composition).complete()
            else:
                return self
        elif isinstance(self, Application):
            f = self.f.partial_primitive()
            x = self.x.partial_primitive()
            return f.partial_apply(x)
        raise ValueError


class Base(Expr):
    """
    A base expression represents either a single transformation or a data
    input. Base expressions may be unfolded into multiple applications of
    primitive transformations. Data input can be seen as a typed variable in an
    expression: in this case, it should be labelled and substituted.
    """

    def __init__(self, definition: Definition, label: Optional[str] = None):
        self.definition = definition
        self.label: Optional[str] = label
        super().__init__(type=definition.type.instance())

    def __str__(self) -> str:
        name = self.definition.name or "anonymous"
        return f"{name} {self.label}" if self.label else name


class Application(Expr):
    """
    A comlex expression, representing an application of the transformation in
    its first argument to the expression in its second argument.
    """

    def __init__(self, f: Expr, x: Expr):
        try:
            result = f.type.apply(x.type)
        except error.AlgebraTypeError as e:
            e.add_expression(f, x)
            raise e
        else:
            self.f: Expr = f
            self.x: Expr = x
            super().__init__(type=result)

    def __str__(self) -> str:
        return f"({self.f} {self.x})"


class Abstraction(PartialExpr):
    """
    An incomplete expression that needs to be supplied with arguments. Not
    normally part of an expression tree --- only used for expanding primitives.
    """

    def __init__(self, composition: Callable[..., PartialExpr]):
        self.composition: Callable[..., PartialExpr] = composition


class TransformationAlgebra(object):
    def __init__(self, *nargs: Definition, **kwargs: Definition):
        """
        Create a transformation algebra with pre-named (positional arguments)
        or to-be-named (keyword arguments) data and operation definitions.
        """
        self.parser: Optional[pp.Parser] = None
        self.definitions: Dict[str, Definition] = {}
        for v in nargs:
            assert v.name
            self.definitions[v.name] = v
        for k, v in kwargs.items():
            assert not v.name
            v.name = k
            self.definitions[v.name] = v

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return "\n".join(str(d) for d in self.definitions.values()) + "\n"

    def generate_parser(self) -> pp.Parser:

        label = pp.Word(pp.alphanums + ':_').setName('identifier')

        expr = pp.MatchFirst(
            pp.CaselessKeyword(d.name) + pp.Optional(label)
            if isinstance(d, Data) else
            pp.CaselessKeyword(d.name)
            for d in self.definitions.values()
        ).setParseAction(
            lambda s, l, t: self.definitions[t[0]].instance(
                t[1] if len(t) > 1 else None)
        )

        return pp.infixNotation(expr, [(
            None, 2, pp.opAssoc.LEFT, lambda s, l, t: reduce(Application, t[0])
        )])

    def parse(self, string: str) -> Expr:
        if not self.parser:
            self.parser = self.generate_parser()
        expr = self.parser.parseString(string, parseAll=True)[0]
        return expr

    def tree(self, string: str) -> None:
        """
        Print a tree corresponding to the given algebra expression.
        """
        print(self.parse(string).tree())

    @staticmethod
    def from_dict(obj: Dict[str, Any]) -> TransformationAlgebra:
        """
        Create transformation algebra from an object, filtering out relevant
        definitions.
        """
        return TransformationAlgebra(**{
            k.rstrip("_"): v for k, v in obj.items()
            if isinstance(v, Definition)
        })
