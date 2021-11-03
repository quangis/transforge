import unittest

from transformation_algebra.type import \
    Type, TypeOperator, TypeSchema, TypeVariable, _, with_parameters, \
    FunctionApplicationError, TypeMismatch, \
    ConstraintViolation, ConstrainFreeVariable


Ω = TypeOperator('Ω')
A, B = TypeOperator('A', supertype=Ω), TypeOperator('B', supertype=Ω)
A1, A2 = TypeOperator('A1', supertype=A), TypeOperator('A2', supertype=A)
B1, B2 = TypeOperator('B1', supertype=B), TypeOperator('B2', supertype=B)
F, G = TypeOperator('F', params=2), TypeOperator('G', params=2)
Z, W = TypeOperator('Z', params=1), TypeOperator('W', params=1)


class TestType(unittest.TestCase):

    def apply(self, f: Type, *xs: Type, result=None):
        """
        Test the application of an argument type to a function type.
        """

        def do_apply():
            res = f.instance()
            for x in xs:
                res = res.apply(x.instance())
            return res

        if isinstance(result, type) and issubclass(result, Exception):
            self.assertRaises(result, do_apply)
        elif result:
            assert isinstance(result, Type)
            self.assertEqual(
                do_apply(),
                result.instance())
        else:
            do_apply()

    def test_with_parameters(self):
        "Test the auxiliary function `with_parameters`."

        self.assertEqual(
            with_parameters(F),
            [F(_, _)]
        )
        self.assertEqual(
            with_parameters(F, param=A),
            [F(A, _), F(_, A)]
        )
        self.assertEqual(
            with_parameters(F, G, param=A, at=1),
            [F(A, _), G(A, _)]
        )

    def test_apply_non_function(self):
        self.apply(A, A, result=FunctionApplicationError)

    def test_basic(self):
        self.apply(A ** B, A, result=B)
        self.apply(A ** B, B, result=TypeMismatch)

    def test_basic_sub(self):
        self.apply(A ** B, A1, result=B)
        self.apply(A1 ** B, A, result=TypeMismatch)

    def test_compound(self):
        self.apply(Z(A) ** B, Z(A), result=B)
        self.apply(Z(A) ** B, Z(B), result=TypeMismatch)

    def test_compound_sub(self):
        self.apply(Z(A) ** B, Z(A1), result=B)
        self.apply(Z(A1) ** B, Z(A), result=TypeMismatch)

    def test_variable(self):
        wrap = TypeSchema(lambda α: α ** Z(α))
        self.apply(wrap, A, result=Z(A))

    def test_compose(self):
        compose = TypeSchema(lambda x, y, z: (y ** z) ** (x ** y) ** (x ** z))
        self.apply(compose, A ** B, B ** A, result=B ** B)
        self.apply(compose, A ** B, B ** A1, result=B ** B)
        self.apply(compose, A ** B, B ** B1, result=TypeMismatch)
        self.apply(compose, A1 ** B, B ** A, result=TypeMismatch)

    def test_variable_sub(self):
        f = TypeSchema(lambda x: (x ** A1) ** x)
        # self.apply(f, A ** A1, result=A)
        self.apply(f, A1 ** A, result=TypeMismatch)

    def test_functions_as_arguments1(self):
        swap = TypeSchema(lambda α, β, γ: (α ** β ** γ) ** (β ** α ** γ))
        f = TypeSchema(lambda x: A ** x ** x)
        self.apply(swap, f, B, A, result=B)

    def test_functions_as_arguments2(self):
        identity = TypeSchema(lambda x: x ** x)
        self.apply(identity, A ** A, A1, result=A)

    def test_order_of_subtype_application(self):
        """
        The order in which subtypes are applied does not matter. Test inspired
        by the discussion in Traytel et al (2011).
        """
        f = TypeSchema(lambda α: α ** α ** B)
        self.apply(f, A1, A, result=B)
        self.apply(f, A, A1, result=B)
        self.apply(f, A, B, result=TypeMismatch)

    def test_order_of_subtype_application_with_constraints(self):
        f = TypeSchema(lambda α: α ** α ** B | α << A)
        self.apply(f, A1, A, result=B)
        self.apply(f, A, A1, result=B)

        # TODO shouldn't this work?
        # self.apply(f, B, result=ConstraintViolation)


    def test_preservation_of_subtypes_in_constraints(self):
        f = TypeSchema(lambda α: α ** α | α << [A, Z(A)])
        self.apply(f, A1, result=A1)
        self.apply(f, Z(A1), result=Z(A1))
        self.apply(f, B, result=ConstraintViolation)

    def test_unification_of_compound_types_in_constraints(self):
        f = TypeSchema(lambda xs, x: xs ** x | xs << [W(x), Z(x)])
        self.apply(f, W(A), result=A)

    def test_non_unification_of_base_types(self):
        # We can't unify with base types from constraints, as they might be
        # subtypes. So in this case, we know that x is an F, but we don't know
        # that its parameters is exactly A: that might be too general a bound.
        f = TypeSchema(lambda x: x ** x | x << Z(A))
        result = f.apply(TypeVariable())
        self.assertEqual(result.operator, Z)
        self.assertTrue(isinstance(result.params[0], TypeVariable))

    def test_multiple_bounds(self):
        f = TypeSchema(lambda x: (x ** x) ** x)

        # This works because A1 ** A1 is acceptable for A ** A1.
        self.apply(f, A ** A1, result=A1)

        # This doesn't work because the upper bound A1 cannot be reconciled
        # with the lower bound A.
        self.apply(f, A1 ** A, result=TypeMismatch)

    def test_constrain_wildcard(self):
        f = TypeSchema(lambda x: x ** x | x << [_])
        self.apply(f, A, result=A)

    def test_constrain_free_variable(self):
        f = TypeSchema(lambda x, y, z: x ** x | y << [x, z])
        g = TypeSchema(lambda x, y, z: x ** x | x << [y, z])
        self.assertRaises(ConstrainFreeVariable, TypeSchema.instance, f)
        self.assertRaises(ConstrainFreeVariable, TypeSchema.instance, g)

    def test_global_subtype_resolution(self):
        f = TypeSchema(lambda x: x ** (x ** x) ** x)
        self.apply(f, A1, A ** A1, result=A1)
        self.apply(f, A, A ** A1, result=A)

    def test_interdependent_types(self):
        f = TypeSchema(lambda α, β: α ** β | α << [Z(β), F(_, β)])
        self.apply(f, Z(A), result=A)
        self.apply(f, A, result=ConstraintViolation)

    def test_subtyping_of_concrete_functions(self):
        self.assertTrue(A ** A <= A1 ** A)
        self.assertTrue(A ** A <= A ** Ω)
        self.assertFalse(A ** A <= Ω ** A)
        self.assertFalse(A ** A <= A ** A1)

    def test_subtyping_of_variables(self):
        x = TypeVariable()
        self.assertEqual(x < x, False)
        self.assertEqual(x <= x, True)

    def test_subtyping_of_variable_functions(self):
        x = TypeVariable()
        self.assertEqual(x ** A <= A1 ** A, None)
        self.assertEqual(A ** x <= A ** Ω, None)
        self.assertEqual(A ** A <= x ** A, None)
        self.assertEqual(A ** A <= A ** x, None)

    def test_subtyping_of_wildcard_functions(self):
        self.assertTrue(_ ** A <= A1 ** A)
        self.assertTrue(A ** _ <= A ** Ω)
        self.assertTrue(A ** A <= _ ** A)
        self.assertTrue(A ** A <= A ** _)
        self.assertFalse(_ ** Ω <= A1 ** A)
        self.assertFalse(A1 ** _ <= A ** Ω)

    def test_constrained_to_base_type(self):
        # See issue #2, which caused an infinite loop
        f = TypeSchema(lambda x: x ** x | x << A)
        g = TypeSchema(lambda x, y: (x ** y) ** y)
        self.apply(g, f)

    def test_constrained_to_compound_type(self):
        # See issue #2
        f = TypeSchema(lambda x: x ** x | x << Z(A))
        g = TypeSchema(lambda x, y: (x ** y) ** y)
        self.apply(g, f)

    def test_unification_of_constraint_with_variables(self):
        # See issue #13
        A, B, C = TypeOperator('A'), TypeOperator('B'), TypeOperator('C')
        R2, R3 = TypeOperator('R2', 2), TypeOperator('R3', 3)
        actual = TypeSchema(lambda x:
            R3(A, x, C) | R2(C, B) << [R2(A, x), R2(C, x)])
        expected = R3(A, B, C)
        self.assertEqual(actual.instance(), expected.instance())

    def test_timely_constraint_check(self):
        # See issue #13
        f = TypeSchema(lambda r, x: r ** x | r << [F(A, x), F(B, x)])
        self.apply(f, F(A, B), result=B)

    def test_unification_of_constraint_options(self):
        # See issue #11
        non_function = TypeSchema(lambda x: x | F(A, A) << [F(A, x), F(A, x)])
        self.apply(non_function, result=A)

    def test_overeager_unification_of_constraint_options(self):
        # See issue #17
        self.assertEqual(F(A, _) <= F(_, A), True)
        x = TypeVariable()
        c = x << [F(A, _), F(_, A)]
        self.assertEqual(len(c.alternatives), 2)
        c = x << [F(_, _), F(_, _)]
        self.assertEqual(len(c.alternatives), 1)

    def test_constraints_extraneous_alternatives(self):
        # Subtypes should be considered extraneous in constraint alternatives,
        # see issue #58
        x = TypeVariable()
        c = x << [A1, A]
        self.assertEqual(c.alternatives, [A()])

        # ... regardless of order, see issue #57
        x = TypeVariable()
        c = x << [A, A1]
        self.assertEqual(c.alternatives, [A()])

    def test_unification_of_constraint_option_subtypes(self):
        # See issue #16
        f = TypeSchema(lambda x, y: F(x, y) | F(y, A) << [F(A, x), F(A1, x)])
        self.assertEqual(
            f.instance().params[0],
            A()
        )

    def test_constraint_check_on_intertwined_variables1(self):
        # See issue #18
        f = TypeSchema(lambda x, y, z: (x ** y) ** z | z << F(x, y))
        g = TypeSchema(lambda x, y: x ** y)
        x, y = f.apply(g).params
        self.assertEqual(len(x._constraints), 0)
        self.assertEqual(len(y._constraints), 0)

    def test_constraint_check_on_intertwined_variables2(self):
        # See issue #18
        f = TypeSchema(lambda x, y: F(x, y) ** y)
        g = TypeSchema(lambda a, b: F(a, b) | F(a, b) << F(a, b))
        y = f.apply(g)
        self.assertEqual(len(y._constraints), 0)

    def test_reach_all_constraints(self):
        f = TypeSchema(lambda a, b, c: a ** b ** c
            | c << [b, _] | b << [c, a] | a << [b, c]).instance()
        self.assertEqual(len(f[1][1].constraints()), 3)

    def test_reach_all_operators(self):
        f = TypeSchema(lambda a, b, c: a ** b ** c
            | c << [b, _] | b << [a, _] | a << [A, _]).instance()
        self.assertEqual(f[1][1].operators(), {A})

    def test_curried_function_signature_same_as_uncurried(self):
        # See issue #53
        self.assertEqual(
            A1 ** A2 ** A,
            (A1, A2) ** A
        )


if __name__ == '__main__':
    unittest.main()
