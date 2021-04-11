import unittest

from transformation_algebra import error
from transformation_algebra.type import Type
from transformation_algebra.expr import \
    TransformationAlgebra, Data, Operation

Int = Type.declare('Int')

one = Data(Int)
add = Operation(Int ** Int ** Int)
add1 = Operation(
    Int ** Int,
    derived=lambda x: add(x, one)
)
compose = Operation(
    lambda α, β, γ: (β ** γ) ** (α ** β) ** (α ** γ),
    derived=lambda f, g, x: f(g(x))
)
algebra = TransformationAlgebra(**globals())


A = Type.declare('A')
B = Type.declare('B')
f = Operation(A ** B)
g = Operation(lambda α: α ** B)


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

    def test_exact_declared_type_in_definition(self):
        self.assertRaises(
            error.TypeAnnotationError, Operation,
            B ** B, derived=lambda x: f(x)
        )
        Operation(A ** B, derived=lambda x: f(x))

    def test_tighter_declared_type_in_definition(self):
        Operation(A ** B, derived=lambda x: g(x))
        Operation(B ** B, derived=lambda x: g(x))

    def test_looser_declared_type_in_definition(self):
        Operation(lambda α: α ** B, derived=lambda x: g(x))
        self.assertRaises(
            error.TypeAnnotationError, Operation,
            lambda α: α ** B, derived=lambda x: f(x))


if __name__ == '__main__':
    unittest.main()
