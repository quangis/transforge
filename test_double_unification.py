#!/usr/bin/env python3

from transformation_algebra.type import Type, _, operators
from transformation_algebra.expr import Data, Operation, TransformationAlgebra

Val = Type.declare('Val')
Loc = Type.declare('Loc', supertype=Val)
Bool = Type.declare('Bool', supertype=Val)
R2 = Type.declare('R2', params=2)
R3 = Type.declare('R3', params=3)
fieldratios = Data(R2(Loc, Bool))

product = Operation(Bool ** Bool ** Bool)
eq = Operation(Val ** Val ** Bool)

select2 = Operation(lambda x, y, rel:
    (x ** y ** Bool) ** rel ** rel
    | rel @ operators(R2, R3, param=x)
    | rel @ operators(R2, R3, param=y)
)
prod = Operation(lambda x, y, z, u, w:
    (y ** z ** u) ** R2(x, y) ** R2(w, z) ** R2(w, u),
)
apply2 = Operation(lambda x1, x2, x3, y:
    (x1 ** x2 ** x3) ** R2(y, x1) ** R2(y, x2) ** R2(y, x3),
    derived=lambda f, x, y: select2(eq, prod(f, x, y))
)

apply2(eq, fieldratios, fieldratios).primitive()

