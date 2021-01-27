"""
The core concept transformation algebra.

This module deviates from Python's convention of using capitalized names for
classes and lowercase for values. These decisions were made to get an interface
that is as close as possible to its formal type system counterpart.
"""

from functools import partial
from abc import ABC, abstractmethod
from typing import Dict

from quangis.transformation.parser import make_parser, Expr
from quangis.transformation.type import TypeOperator, TypeVar, AlgebraType

# Some type variables for convenience
x, y, z = (TypeVar() for _ in range(0, 3))


class TransformationAlgebra(ABC):
    """
    Abstract base for transformation algebras. To make a concrete
    transformation algebra, subclass this class and add a mapping "functions"
    of function names to corresponding AlgebraTypes.
    """

    def __init__(self):
        self.parser = make_parser(self.functions)

    @property
    @abstractmethod
    def functions(self) -> Dict[str, AlgebraType]:
        return NotImplemented

    def parse(self, string: str) -> Expr:
        return self.parser.parseString(string)[0]


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

    functions = {

        # data constructors
        "pointmeasures": R(Reg, Itv),
        "amountpatches": R(Reg, Nom),
        "contour": R(Ord, Reg),
        "objects": R(Obj, Ratio),
        "objectregions": R(Obj, Reg),
        "contourline": R(Itv, Reg),
        "objectcounts": R(Obj, Count),
        "field": R(Loc, Ratio),
        "object": R(Obj),
        "region": R(Reg),
        "in": R(Nom),
        "countV": R(Count),
        "ratioV": R(Ratio),
        "interval": R(Itv),
        "ordinal": R(Ord),
        "nominal": R(Nom),

        # transformations (without implementation)

        # functional
        "compose": (y ** z) ** (x ** y) ** (x ** z),

        # derivations
        "ratio": Ratio ** Ratio ** Ratio,

        # aggregations of collections
        "count": R(Obj) ** Ratio,
        "size": R(Loc) ** Ratio,
        "merge": R(Reg) ** Reg,
        "centroid": R(Loc) ** Loc,

        # statistical operations
        "avg": R(Ent, Itv) ** Itv,
        "min": R(Ent, Ord) ** Ord,
        "max": R(Ent, Ord) ** Ord,
        "sum": R(Ent, Count) ** Count,

        # conversions
        "reify": R(Loc) ** Reg,
        "deify": Reg ** R(Loc),
        "get": R(x) ** x | [x << [Ent]],
        "invert": R(Loc, Ord) ** R(Ord, Reg),  # TODO overload R(Loc, Nom) ** R(Reg, Nom)
        "revert": R(Ord, Reg) ** R(Loc, Ord),  # TODO overload

        # quantified relations
        "oDist": R(Obj, Reg) ** R(Obj, Reg) ** R(Obj, Ratio, Obj),
        "lDist": R(Loc) ** R(Loc) ** R(Loc, Ratio, Loc),
        "loDist": R(Loc) ** R(Obj, Reg) ** R(Loc, Ratio, Obj),
        "oTopo": R(Obj, Reg) ** R(Obj, Reg) ** R(Obj, Nom, Obj),
        "loTopo": R(Loc) ** R(Obj, Reg) ** R(Loc, Nom, Obj),
        "nDist": R(Obj) ** R(Obj) ** R(Obj, Ratio, Obj) ** R(Obj, Ratio, Obj),
        "lVis": R(Loc) ** R(Loc) ** R(Loc, Itv) ** R(Loc, Bool, Loc),
        "interpol": R(Reg, Itv) ** R(Loc) ** R(Loc, Itv),

        # amount operations
        "fcont": R(Loc, Itv) ** Ratio,
        "ocont": R(Obj, Ratio) ** Ratio,

        # relational
        "pi1":
            x ** y | [x << [R(y, TypeVar(), TypeVar())]],
        "pi2":
            x ** y | [x << [R(TypeVar(), y, TypeVar())]],
        "pi3":
            x ** y | [x << [R(TypeVar(), TypeVar(), y)]],
        "sigmae":
            x ** y ** x | [x << [Qlt]],  #  y must contain x
        "sigmale":
            x ** y ** x | [x << [Ord]],  # y must contain x
        "bowtie":
            x ** R(y) ** x | [y << [Ent]],  # y must contain x
        "bowtie*":
            R(x, y, x) ** R(x, y) ** R(x, y, x) | [y << [Qlt], x << [Ent]],
        "bowtie_":
            (Qlt ** Qlt ** Qlt) ** R(Ent, Qlt) ** R(Ent, Qlt) ** R(Ent, Qlt),
        "groupbyL":
            (R(y, Qlt) ** Qlt) ** R(x, Qlt, y) ** R(x, Qlt) | [x << [Ent], y << [Ent]],
        "groupbyR":
            (R(x, Qlt) ** Qlt) ** R(x, Qlt, y) ** R(y, Qlt) | [x << [Ent], y << [Ent]],
    }
