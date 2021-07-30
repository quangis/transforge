#!/usr/bin/env python3

from transformation_algebra.type import Type, _, operators
from transformation_algebra.expr import Data, Operation, TransformationAlgebra

Val = Type.declare('Val')
Loc = Type.declare('Loc', supertype=Val)
Ratio = Type.declare('Ratio', supertype=Val)
Bool = Type.declare('Bool', supertype=Val)
R1 = Type.declare('R1', params=1)
R2 = Type.declare('R2', params=2)
R3 = Type.declare('R3', params=3)
product = Operation(Ratio ** Ratio ** Ratio)
eq = Operation(Val ** Val ** Bool, doc="equal", derived=None)
fieldratios = Data(R2(Loc, Ratio))
pi2 = Operation(lambda rel, x: rel ** R1(x) | rel @ operators(R1, R2, R3, param=x, at=2))
pi12 = Operation(lambda x, y: R3(x, y, _) ** R2(x, y))
select2 = Operation(lambda x, y, rel:
    (x ** y ** Bool) ** rel ** rel
    | rel @ operators(R1, R2, R3, param=x)
    | rel @ operators(R1, R2, R3, param=y)
)
swap = Operation(lambda α, β, γ: (α ** β ** γ) ** (β ** α ** γ))
compose = Operation(lambda α, β, γ: (β ** γ) ** (α ** β) ** (α ** γ))
join = Operation(lambda x, y, z: R2(x, y) ** R2(y, z) ** R2(x, z))
apply = Operation(lambda x, y: (x ** y) ** R1(x) ** R2(x, y))
apply1 = Operation(lambda x1, x2, y:
    (x1 ** x2) ** R2(y, x1) ** R2(y, x2),
    derived=lambda f, y: join (y) (apply (f) (pi2 (y))))
prod3 = Operation(lambda x, y, z: R2(z, R2(x, y)) ** R3(x, y, z))
prod = Operation(lambda x, y, z, u, w:
    (y ** z ** u) ** R2(x, y) ** R2(w, z) ** R2(x, R2(w, u)),
    derived=lambda f, x, y: apply1 (compose ((swap (apply1)) (y)) (f)) (x)
)
apply2 = Operation(lambda x1, x2, x3, y:
    (x1 ** x2 ** x3) ** R2(y, x1) ** R2(y, x2) ** R2(y, x3),
    name="apply2",
    derived=lambda f, x, y: pi12 (select2 (eq) (prod3 (prod (f) (x) (y))))
)

algebra = TransformationAlgebra()
algebra.add(**globals())
t = algebra.parse("apply2 product (fieldratios x1) (fieldratios x2)").primitive()
