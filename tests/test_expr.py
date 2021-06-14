import unittest

from transformation_algebra import error
from transformation_algebra.type import Type
from transformation_algebra.expr import \
    TransformationAlgebra, Data, Operation


class TestAlgebra(unittest.TestCase):
    def test_primitive(self):
        Int = Type.declare('Int')
        add = Operation(Int ** Int ** Int)
        one = Data(Int)
        algebra = TransformationAlgebra()
        algebra.add(
            one=one,
            add1=Operation(
                Int ** Int,
                derived=lambda x: add(x, one)
            ),
            compose=Operation(
                lambda α, β, γ: (β ** γ) ** (α ** β) ** (α ** γ),
                derived=lambda f, g, x: f(g(x))
            )
        )
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
        A, B = Type.declare('A'), Type.declare('B')
        f = Operation(A ** B)
        self.assertRaises(
            error.SubtypeMismatch,
            Operation.validate,
            Operation(B ** B, derived=lambda x: f(x))
        )
        Operation(A ** B, derived=lambda x: f(x)).validate()

    def test_tighter_declared_type_in_definition(self):
        A, B = Type.declare('A'), Type.declare('B')
        g = Operation(lambda α: α ** B)
        Operation(A ** B, derived=lambda x: g(x)).validate()
        Operation(B ** B, derived=lambda x: g(x)).validate()

    def test_looser_declared_type_in_definition(self):
        A, B = Type.declare('A'), Type.declare('B')
        f, g = Operation(A ** B), Operation(lambda α: α ** B)
        Operation(lambda α: α ** B, derived=lambda x: g(x)).validate()
        self.assertRaises(
            error.DeclaredTypeTooGeneral,
            Operation.validate,
            Operation(lambda α: α ** B, derived=lambda x: f(x)))

    def test_same_labels_unify(self):
        # See issue #10
        A, B = Type.declare('A'), Type.declare('B')
        algebra = TransformationAlgebra()
        algebra.add(
            d1=Data(A),
            d2=Data(B),
            f=Operation(A ** B ** A))
        algebra.parse("f (d1 x) (d2 y)")
        self.assertRaises(error.TATypeError, algebra.parse, "f (d1 x) (d2 x)")


if __name__ == '__main__':
    unittest.main()
