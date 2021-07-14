import unittest

from transformation_algebra import error
from transformation_algebra.type import Type
from transformation_algebra.expr import \
    TransformationAlgebra, Data, Operation


class TestAlgebra(unittest.TestCase):

    def test_currying(self):
        """
        Multiple arguments may be provided through partial or full application.
        """
        A = Type.declare('A')
        x = Data(A)
        f = Operation(A ** A ** A)
        self.assertTrue(f(x, x).match(f(x)(x)))

    def test_primitive(self):
        """
        Expressions can be converted to primitive form.
        """
        Int = Type.declare('Int')
        one = Data(Int, name='one')
        add = Operation(Int ** Int ** Int, name='add')
        add1 = Operation(
            Int ** Int,
            derived=lambda x: add(x, one),
            name='add1'
        )
        compose = Operation(
            lambda α, β, γ: (β ** γ) ** (α ** β) ** (α ** γ),
            derived=lambda f, g, x: f(g(x)),
            name='compose'
        )
        algebra = TransformationAlgebra()
        algebra.add(add, one, add1, compose)
        a = compose(add1, add1, one)
        b = add(add(one, one), one)
        self.assertTrue(a.primitive().match(b))

    def test_matching(self):
        """
        Expressions can be compared to one another.
        """
        A = Type.declare('A')
        x = Data(A)
        f = Operation(A ** A, name='f')
        g = Operation(A ** A, name='g')
        self.assertTrue(f(x).match(f(x)))
        self.assertFalse(f(x).match(g(x)))

    def test_primitive_high(self):
        """
        Test that non-primitive expressions defined in terms of
        other non-primitive expressions are expanded properly.
        """
        A = Type.declare('A')
        x = Data(A)
        f = Operation(A ** A, name='f')
        g = Operation(A ** A, name='g',
            derived=lambda x: f(x))
        h = Operation(A ** A, name='h',
            derived=lambda x: g(x))
        algebra = TransformationAlgebra()
        algebra.add(f, g, h)
        self.assertTrue(f(x).match(f(x)))
        self.assertFalse(h(x).match(f(x)))
        self.assertTrue(g(x).primitive().match(f(x)))
        self.assertTrue(h(x).primitive().match(f(x)))
        self.assertTrue(h(x).primitive().match(g(x).primitive()))

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
