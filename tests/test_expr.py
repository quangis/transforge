import unittest
from .testcase import TestCase  # type: ignore

from transforge.type import TypeOperator, _, \
    SubtypeMismatch, TypingError
from transforge.expr import Expr, Operator, Source, \
    DeclaredTypeTooGeneralError, DeclarationError
from transforge.lang import Language, \
    TypeAnnotationError


class TestAlgebra(TestCase):

    def assertMatch(self, *es: Expr, fix: bool = False):
        if fix:
            for e in es:
                e.fix()
        es2 = iter(es)
        next(es2)
        for e1, e2 in zip(es, es2):
            with self.subTest(msg=f"{e1} != {e2}"):
                self.assertTrue(e1.match(e2))

    def test_currying(self):
        """
        Multiple arguments may be provided through partial or full application.
        """
        A = TypeOperator('A')
        x = Source(type=A)
        f = Operator(type=A ** A ** A)
        self.assertTrue(f(x, x).match(f(x)(x)))

    def test_primitive(self):
        """
        Expressions can be converted to primitive form.
        """
        Int = TypeOperator('Int')
        one = Source(type=Int)
        add = Operator(type=Int ** Int ** Int, name='add')
        add1 = Operator(
            type=Int ** Int,
            body=lambda x: add(x, one),
            name='add1'
        )
        compose = Operator(
            type=lambda α, β, γ: (β ** γ) ** (α ** β) ** (α ** γ),
            body=lambda f, g, x: f(g(x)),
            name='compose'
        )
        a = compose(add1, add1, one)
        b = add(add(one, one), one)
        self.assertTrue(a.primitive().match(b))

    def test_matching(self):
        """
        Expressions can be compared to one another.
        """
        A = TypeOperator('A')
        x = Source(type=A)
        f = Operator(type=A ** A, name='f')
        g = Operator(type=A ** A, name='g')
        self.assertTrue(f(x).match(f(x)))
        self.assertFalse(f(x).match(g(x)))

    def test_primitive_high(self):
        """
        Test that non-primitive expressions defined in terms of
        other non-primitive expressions are expanded properly.
        """
        A = TypeOperator('A')
        x = Source(type=A)
        f = Operator(type=A ** A, name='f')
        g = Operator(type=A ** A, name='g',
            body=lambda x: f(x))
        h = Operator(type=A ** A, name='h',
            body=lambda x: g(x))
        self.assertTrue(f(x).match(f(x)))
        self.assertFalse(h(x).match(f(x)))
        self.assertTrue(g(x).primitive().match(f(x)))
        self.assertTrue(h(x).primitive().match(f(x)))
        self.assertTrue(h(x).primitive().match(g(x).primitive()))

    def test_primitives_retain_types(self):
        """
        Make sure that primitives have the correct type.
        """
        A = TypeOperator('A')
        x = Source(type=A)
        f = Operator(type=lambda α: α ** α, name='f')
        g = Operator(type=lambda α: α ** α, name='g', body=lambda x: f(x))
        self.assertTrue(g(x).primitive().type.match(A.instance()))

    def test_double_binding(self):
        """
        An issue once arised in which the assertion that variables may only be
        bound once was violated, due to complex interactions of primitive
        Operators. This test makes sure that issue no longer exists.
        """
        Value = TypeOperator('Val')
        Bool = TypeOperator('Bool', supertype=Value)
        Map = TypeOperator('Map', params=2)
        data = Source(type=Map(Value, Bool))
        eq = Operator(type=Value ** Value ** Bool)
        select = Operator(
            type=lambda α, β, τ:
                (α ** β ** Bool) ** τ ** τ
                [τ << {Map(α, _), Map(_, α)}, τ << {Map(β, _), Map(_, β)}]
        )
        prod = Operator(type=lambda α, β, γ, τ:
            (α ** β ** γ) ** Map(τ, α) ** Map(τ, β) ** Map(τ, γ),
        )
        app = Operator(type=lambda α, β, γ, τ:
            (α ** β ** γ) ** Map(τ, α) ** Map(τ, β) ** Map(τ, γ),
            body=lambda f, x, y: select(eq, prod(f, x, y))
        )
        self.assertTrue(app(eq, data, data).primitive().match(
            select(eq, prod(eq, data, data))
        ))

    def test_exact_declared_type_in_definition(self):
        A, B = TypeOperator('A'), TypeOperator('B')
        f = Operator(type=A ** B)
        self.assertRaisesChain(
            [DeclarationError, SubtypeMismatch],
            Operator.validate,
            Operator(type=B ** B, body=lambda x: f(x))
        )
        Operator(type=A ** B, body=lambda x: f(x)).validate()

    def test_tighter_declared_type_in_definition(self):
        A, B = TypeOperator('A'), TypeOperator('B')
        g = Operator(name='g', type=lambda α: α ** B)
        Operator(type=A ** B, body=lambda x: g(x)).validate()
        Operator(type=B ** B, body=lambda x: g(x)).validate()

    def test_looser_declared_type_in_definition(self):
        A, B = TypeOperator('A'), TypeOperator('B')
        f, g = Operator(type=A ** B), Operator(type=lambda α: α ** B)
        Operator(type=lambda α: α ** B, body=lambda x: g(x)).validate()
        self.assertRaisesChain(
            [DeclarationError, DeclaredTypeTooGeneralError],
            Operator.validate,
            Operator(type=lambda α: α ** B, body=lambda x: f(x)))

    def test_same_labels_unify(self):
        # See issue #10
        A = TypeOperator()
        B = TypeOperator()
        f = Operator(type=A ** B ** A)
        lang = Language(locals())

        lang.parse("f (1 : A) (2 : B)", Source(), Source())
        self.assertRaises(TypeAnnotationError, lang.parse, "f (1 : A) (1 : B)",
            Source())

    def test_source_reuse_does_not_affect_type_fixing(self):
        # See issue #110: Reusing sources may cause overly general types to be
        # inferred the first time it's encountered, causing type mismatches
        # the second time around. This is an issue both on the graph level and
        # on the expression level.
        A = TypeOperator()
        B = TypeOperator(supertype=A)
        f = Operator(type=lambda x: x ** x ** x)
        lang = Language(locals())
        lang.parse("f (1: A) (1: B)", Source())
        lang.parse("f (f (1: A) (1: A)) (1: B)", Source())

    def test_type_annotations_carry_over(self):
        # What do type annotations really mean? If it just means that an
        # expression 'is of a certain type', we can simply unify the type of
        # the expression with the given type. But usually it does not *just*
        # mean that. For example, if we annotate input `1` with a type `B`, we
        # still want to be able to connect a source of a more specific type
        # (say `C`) to input `1`. Not only that, we would also like the more
        # specific type to *carry over* to the rest of the expression. See
        # issue #110 for where this used to cause problems.
        A = TypeOperator()
        B = TypeOperator(supertype=A)
        C = TypeOperator(supertype=B)
        f = Operator(type=A ** A)
        g = Operator(type=lambda x: x ** x)
        lang = Language(locals())

        # For `1: B`, it should still be valid to provide a subtype...
        self.assertMatch(
            lang.parse("1", Source(C)),
            lang.parse("1: B", Source(C)),
            Source(C),
        )
        # ... but not a supertype.
        self.assertRaises(TypingError, lang.parse, "1: B", Source(A))
        # If the source type is not known but fixed via the annotations:
        self.assertMatch(
            lang.parse("1: B", Source()),
            lang.parse("1", Source(B)),
            Source(B),
            fix=True)

        # Now in a function. Keep in mind: When we apply `g: x ** x` to an `A`,
        # deduce `x >= A`; when we apply `f: A ** A` to an `x`, deduce `x <=
        # A`: more specific/general types would mismatch. So, for `f 1`, we
        # know that the type `x` of `1` is at most `x <= A`. For `f (1: B)`, we
        # still have `x <= A`; `1: B` simply lowers that upper bound to `B`.
        self.assertMatch(
            lang.parse("f 1", Source(C)),
            lang.parse("f (1: B)", Source(C)),
            f(Source(C))
        )
        self.assertMatch(
            lang.parse("g 1", Source(C)),
            lang.parse("g (1: B)", Source(C)),
            g(Source(C))
        )
        self.assertRaises(TypingError, lang.parse, "f (1: B)", Source(A))
        self.assertRaises(TypingError, lang.parse, "g (1: B)", Source(A))
        # Crucially, when we figure out sources via annotations, we get B
        # again, not the more general A:
        self.assertMatch(
            lang.parse("f (1: B)", Source()),
            lang.parse("f 1", Source(B)),
            f(Source(B)),
            fix=True)
        self.assertMatch(
            lang.parse("g (1: B)", Source()),
            lang.parse("g 1", Source(B)),
            g(Source(B)),
            fix=True
        )


if __name__ == '__main__':
    unittest.main()
