"""
The core concept transformation algebra.

This module deviates from Python's convention of using capitalized names for
classes and lowercase for values. These decisions were made to get an interface
that is as close as possible to its formal type system counterpart.
"""

from __future__ import annotations

import pyparsing as pp
from functools import partial, reduce
from itertools import chain
from abc import ABC
from typing import List, Iterable, Union

from quangis.transformation.type import TypeOperator, TypeVar, AlgebraType, \
    Definition

# Some type variables for convenience
x, y, z = (TypeVar() for _ in range(0, 3))
define = Definition


def has(t: AlgebraType, at=None) -> List[AlgebraType]:
    """
    Typeclass for relationship types that contain another type somewhere.
    """
    R = partial(TypeOperator, "R")
    if at == 1:
        return [R(t), R(t, TypeVar()), R(t, TypeVar(), TypeVar())]
    elif at == 2:
        return [R(TypeVar(), t), R(TypeVar(), t, TypeVar())]
    elif at == 3:
        return [R(TypeVar(), TypeVar(), t)]
    return list(chain(*(has(t, at=i) for i in range(1, 4))))


class Expr(object):
    """
    An expression of a transformation algebra.
    """

    def __init__(self, tokens: List[Union[str, Expr]], type: AlgebraType):
        self.tokens = tokens
        self.type = type

    def __str__(self) -> str:
        if self.type.is_function():
            return "{}".format(" ".join(map(str, self.tokens)))
        else:
            return "({tokens} : \033[1m{type}\033[0m)".format(
                tokens=" ".join(map(str, self.tokens)),
                type=str(self.type)
            )

    @staticmethod
    def apply(fn: Expr, arg: Expr) -> Expr:
        if isinstance(fn.type, TypeOperator):
            res = Expr([fn, arg], fn.type.apply(arg.type))
            fn.type = fn.type.instantiate()
            arg.type = arg.type.instantiate()
            return res
        else:
            raise RuntimeError("applying to non-function value")


class TransformationAlgebra(ABC):
    """
    Abstract base for transformation algebras. To make a concrete
    transformation algebra, subclass this class and add properties of type
    `Definition`.
    """

    def __init__(self):
        self.parser = self.make_parser()

    def definitions(self) -> Iterable[Definition]:
        for attr in dir(self):
            val = getattr(self, attr)
            if isinstance(val, Definition):
                if attr == 'in_':
                    attr = 'in'
                val.name = attr
                yield val

    def make_parser(self) -> pp.Parser:

        defs = {d.name: d for d in self.definitions()}

        identifier = pp.Word(
            pp.alphas + '_', pp.alphanums + ':_'
        ).setName('identifier')

        fn = pp.MatchFirst(
            pp.CaselessKeyword(k) + t.data * identifier
            if t.data else
            pp.CaselessKeyword(k)
            for k, t in sorted(defs.items())
        ).setParseAction(lambda s, l, t: Expr(t, defs[t[0]].instance()))

        return pp.infixNotation(
            fn,
            [(None, 2, pp.opAssoc.LEFT, lambda s, l, t: reduce(Expr.apply, t[0]))]
        )

    def parse(self, string: str) -> Expr:
        return self.parser.parseString(string, parseAll=True)[0]


class CCT(TransformationAlgebra):
    """
    Core concept transformation algebra.
    """

    # Entity value types
    V = partial(TypeOperator)
    Ent = V("entity")
    Obj = V("object", supertype=Ent)  # O
    Reg = V("region", supertype=Ent)  # S
    Loc = V("location", supertype=Ent)  # L
    Qlt = V("quality", supertype=Ent)  # Q
    Nom = V("nominal", supertype=Qlt)
    Bool = V("boolean", supertype=Nom)
    Ord = V("ordinal", supertype=Nom)
    Count = V("count", supertype=Ord)
    Ratio = V("ratio", supertype=Count)
    Itv = V("interval", supertype=Ratio)

    # Relation types and type synonyms
    R = partial(TypeOperator, "R")
    SpatialField = R(Loc, Qlt)
    InvertedField = R(Qlt, Reg)
    FieldSample = R(Reg, Qlt)
    ObjectExtent = R(Obj, Reg)
    ObjectQuality = R(Obj, Qlt)
    NominalField = R(Loc, Nom)
    BooleanField = R(Loc, Bool)
    NominalInvertedField = R(Nom, Reg)
    BooleanInvertedField = R(Bool, Reg)

    # Function type definitions

    # data inputs
    pointmeasures = define(R(Reg, Itv), data=1)
    amountpatches = define(R(Reg, Nom), data=1)
    contour = define(R(Ord, Reg), data=1)
    objects = define(R(Obj, Ratio), data=1)
    objectregions = define(R(Obj, Reg), data=1)
    contourline = define(R(Itv, Reg), data=1)
    objectcounts = define(R(Obj, Count), data=1)
    field = define(R(Loc, Ratio), data=1)
    object = define(R(Obj), data=1)
    region = define(R(Reg), data=1)
    in_ = Nom | ()
    countV = define(Count, data=1)
    ratioV = define(Ratio, data=1)
    interval = define(Itv, data=1)
    ordinal = define(Ord, data=1)
    nominal = define(Nom, data=1)

    # transformations (without implementation)

    # functional
    compose = (y ** z) ** (x ** y) ** (x ** z) | ()

    # derivations
    ratio = Ratio ** Ratio ** Ratio | ()

    # aggregations of collections
    count = R(Obj) ** Ratio | ()
    size = R(Loc) ** Ratio | ()
    merge = R(Reg) ** Reg | ()
    centroid = R(Loc) ** Loc | ()

    # statistical operations
    avg = R(Ent, Itv) ** Itv | ()
    min = R(Ent, Ord) ** Ord | ()
    max = R(Ent, Ord) ** Ord | ()
    sum = R(Ent, Count) ** Count | ()

    # conversions
    reify = R(Loc) ** Reg | ()
    deify = Reg ** R(Loc) | ()
    get = R(x) ** x | x << [Ent]
    invert = x ** y | x ** y << [R(Loc, Ord) ** R(Ord, Reg), R(Loc, Nom) ** R(Reg, Nom)]
    revert = x ** y | x ** y << [R(Ord, Reg) ** R(Loc, Ord), R(Reg, Nom) ** R(Loc, Nom)]

    # quantified relations
    oDist = R(Obj, Reg) ** R(Obj, Reg) ** R(Obj, Ratio, Obj) | ()
    lDist = R(Loc) ** R(Loc) ** R(Loc, Ratio, Loc) | ()
    loDist = R(Loc) ** R(Obj, Reg) ** R(Loc, Ratio, Obj) | ()
    oTopo = R(Obj, Reg) ** R(Obj, Reg) ** R(Obj, Nom, Obj) | ()
    loTopo = R(Loc) ** R(Obj, Reg) ** R(Loc, Nom, Obj) | ()
    nDist = R(Obj) ** R(Obj) ** R(Obj, Ratio, Obj) ** R(Obj, Ratio, Obj) | ()
    lVis = R(Loc) ** R(Loc) ** R(Loc, Itv) ** R(Loc, Bool, Loc) | ()
    interpol = R(Reg, Itv) ** R(Loc) ** R(Loc, Itv) | ()

    # amount operations
    fcont = R(Loc, Itv) ** Ratio | ()
    ocont = R(Obj, Ratio) ** Ratio | ()

    # relational
    pi1 = x ** y | () # | [x << has(y, at=1)]
    pi2 = x ** y | () # | [x << has(y, at=2)]
    pi3 = x ** y | () #| [x << has(y, at=3)]
    sigmae = x ** y ** x | () #| [x << [Qlt], y << has(x)]
    sigmale = x ** y ** x | () #| [x << [Ord], y << has(x)]
    bowtie = x ** R(y) ** x | ()  # | [y << [Ent], y << has(x)]
    bowtiestar = R(x, y, x) ** R(x, y) ** R(x, y, x) | () #| [y << [Qlt], x << [Ent]]
    bowtie_ = (Qlt ** Qlt ** Qlt) ** R(Ent, Qlt) ** R(Ent, Qlt) ** R(Ent, Qlt) | ()
    groupbyL = (R(y, Qlt) ** Qlt) ** R(x, Qlt, y) ** R(x, Qlt) | () # | [x << [Ent], y << [Ent]]
    groupbyR = (R(x, Qlt) ** Qlt) ** R(x, Qlt, y) ** R(y, Qlt) | () # | [x << [Ent], y << [Ent]]
    groupbyR_simpler = (R(Ent) ** z) ** R(x, Qlt, y) ** R(y, z) | ()
