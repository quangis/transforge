import unittest

from transformation_algebra import error
from transformation_algebra.type import Type, _
from transformation_algebra.expr import Operator, Source
from transformation_algebra.alg import TransformationAlgebra


class TestAlgebra(unittest.TestCase):

    def test_currying(self):
        """
        Multiple arguments may be provided through partial or full application.
        """
        A = Type.declare('A')
        x = Source(A)
        f = Operator(A ** A ** A)
        self.assertTrue(f(x, x).match(f(x)(x)))

    def test_primitive(self):
        """
        Expressions can be converted to primitive form.
        """
        Int = Type.declare('Int')
        one = Source(Int)
        add = Operator(Int ** Int ** Int, name='add')
        add1 = Operator(
            Int ** Int,
            define=lambda x: add(x, one),
            name='add1'
        )
        compose = Operator(
            lambda α, β, γ: (β ** γ) ** (α ** β) ** (α ** γ),
            define=lambda f, g, x: f(g(x)),
            name='compose'
        )
        algebra = TransformationAlgebra()
        algebra.add(add, add1, compose)
        a = compose(add1, add1, one)
        b = add(add(one, one), one)
        self.assertTrue(a.primitive().match(b))

    def test_matching(self):
        """
        Expressions can be compared to one another.
        """
        A = Type.declare('A')
        x = Source(A)
        f = Operator(A ** A, name='f')
        g = Operator(A ** A, name='g')
        self.assertTrue(f(x).match(f(x)))
        self.assertFalse(f(x).match(g(x)))

    def test_primitive_high(self):
        """
        Test that non-primitive expressions defined in terms of
        other non-primitive expressions are expanded properly.
        """
        A = Type.declare('A')
        x = Source(A)
        f = Operator(A ** A, name='f')
        g = Operator(A ** A, name='g',
            define=lambda x: f(x))
        h = Operator(A ** A, name='h',
            define=lambda x: g(x))
        algebra = TransformationAlgebra()
        algebra.add(f, g, h)
        self.assertTrue(f(x).match(f(x)))
        self.assertFalse(h(x).match(f(x)))
        self.assertTrue(g(x).primitive().match(f(x)))
        self.assertTrue(h(x).primitive().match(f(x)))
        self.assertTrue(h(x).primitive().match(g(x).primitive()))

    def test_primitives_retain_types(self):
        """
        Make sure that primitives have the correct type.
        """
        A = Type.declare('A')
        x = Source(A)
        f = Operator(lambda α: α ** α, name='f')
        g = Operator(lambda α: α ** α, name='g', define=lambda x: f(x))
        self.assertTrue(g(x).primitive().type.match(A.instance()))

    def test_double_binding(self):
        """
        An issue once arised in which the assertion that variables may only be
        bound once was violated, due to complex interactions of primitive
        Operators. This test makes sure that issue no longer exists.
        """
        Value = Type.declare('Val')
        Bool = Type.declare('Bool', supertype=Value)
        Map = Type.declare('Map', params=2)
        data = Source(Map(Value, Bool))
        eq = Operator(Value ** Value ** Bool)
        select = Operator(
            lambda α, β, τ: (α ** β ** Bool) ** τ ** τ
            | τ @ [Map(α, _), Map(_, α)]
            | τ @ [Map(β, _), Map(_, β)]
        )
        prod = Operator(lambda α, β, γ, τ:
            (α ** β ** γ) ** Map(τ, α) ** Map(τ, β) ** Map(τ, γ),
        )
        app = Operator(lambda α, β, γ, τ:
            (α ** β ** γ) ** Map(τ, α) ** Map(τ, β) ** Map(τ, γ),
            define=lambda f, x, y: select(eq, prod(f, x, y))
        )
        self.assertTrue(app(eq, data, data).primitive().match(
            select(eq, prod(eq, data, data))
        ))

    def test_exact_declared_type_in_definition(self):
        A, B = Type.declare('A'), Type.declare('B')
        f = Operator(A ** B)
        self.assertRaises(
            error.SubtypeMismatch,
            Operator.validate_type,
            Operator(B ** B, define=lambda x: f(x))
        )
        Operator(A ** B, define=lambda x: f(x)).validate_type()

    def test_tighter_declared_type_in_definition(self):
        A, B = Type.declare('A'), Type.declare('B')
        g = Operator(lambda α: α ** B)
        Operator(A ** B, define=lambda x: g(x)).validate_type()
        Operator(B ** B, define=lambda x: g(x)).validate_type()

    def test_looser_declared_type_in_definition(self):
        A, B = Type.declare('A'), Type.declare('B')
        f, g = Operator(A ** B), Operator(lambda α: α ** B)
        Operator(lambda α: α ** B, define=lambda x: g(x)).validate_type()
        self.assertRaises(
            error.DeclaredTypeTooGeneral,
            Operator.validate_type,
            Operator(lambda α: α ** B, define=lambda x: f(x)))

    def test_same_labels_unify(self):
        # See issue #10
        A, B = Type.declare('A'), Type.declare('B')
        algebra = TransformationAlgebra()
        algebra.add(
            d1=Operator(A),
            d2=Operator(B),
            f=Operator(A ** B ** A))
        algebra.parse("f (d1 x) (d2 y)")
        self.assertRaises(error.TATypeError, algebra.parse, "f (d1 x) (d2 x)")


if __name__ == '__main__':
    unittest.main()
