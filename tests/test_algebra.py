import unittest

from transformation_algebra.type import Operator, Schema
from transformation_algebra.expr import \
    TransformationAlgebra, Definition

Int = Operator('Int')
one = Definition(Int)
add = Definition(Int ** Int ** Int)
add1 = Definition(
    type=Int ** Int,
    term=lambda x: add(x, one)
)
compose = Definition(
    type=Schema(lambda α, β, γ: (β ** γ) ** (α ** β) ** (α ** γ)),
    term=lambda f, g, x: f(g(x))
)
algebra = TransformationAlgebra.from_dict(globals())


class TestAlgebra(unittest.TestCase):
    def test_primitive(self):
        t = algebra.parse("compose add1 add1 one")
        # TODO should be equal to:
        # Int
        #  ├─Int ** Int
        #  │  ├─(Int ** Int) ** Int ** Int
        #  │  │  ├─╼ compose : (β ** γ) ** (α ** β) ** α ** γ
        #  │  │  └─╼ add1 : Int ** Int
        #  │  └─╼ add1 : Int ** Int
        #  └─╼ one : Int

        t = t.primitive()
        # TODO should be equal to:
        # Int
        #  ├─Int ** Int
        #  │  ├─╼ add : Int ** Int ** Int
        #  │  └─Int
        #  │     ├─Int ** Int
        #  │     │  ├─╼ add : Int ** Int ** Int
        #  │     │  └─╼ one : Int
        #  │     └─╼ one : Int
        #  └─╼ one : Int


if __name__ == '__main__':
    unittest.main()
