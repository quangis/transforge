import unittest

from transformation_algebra.type import \
    Type, TypeOperator, TypeSchema, TypeOperation, TypeVariable, _, with_parameters, \
    FunctionApplicationError, TypeMismatch, \
    ConstraintViolation, ConstrainFreeVariable, EliminationConstraint


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

        if result is TypeVariable:
            self.assertIsInstance(do_apply(), TypeVariable)
        elif isinstance(result, type) and issubclass(result, Exception):
            self.assertRaises(result, do_apply)
        elif result:
            assert isinstance(result, Type)
            self.assertEqual(
                do_apply(),
                result.instance())
        else:
            assert result is None
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

    def test_base_super_subtypes(self):
        # Test that base type operators are aware of their sub- and supertypes

        A = TypeOperator('A')
        B, C = TypeOperator('B', supertype=A), TypeOperator('C', supertype=A)
        self.assertEqual(A.parent, None)
        self.assertEqual(B.children, set())
        self.assertEqual(C.children, set())
        self.assertEqual(B.parent, A)
        self.assertEqual(C.parent, A)
        self.assertEqual(A.children, {B, C})

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
        A, B = TypeOperator('A'), TypeOperator('B')
        A1 = TypeOperator('A1', supertype=A)
        B1 = TypeOperator('B1', supertype=B)
        compose = TypeSchema(lambda x, y, z: (y ** z) ** (x ** y) ** (x ** z))
        self.apply(compose, A ** B, B ** A)  # , result=B ** B)
        self.apply(compose, A ** B, B ** A1)  # , result=B ** B)
        self.apply(compose, A ** B, B ** B1, result=TypeMismatch)
        self.apply(compose, A1 ** B, B ** A, result=TypeMismatch)

    @unittest.skip("unfinished")
    def test_equal_lower_and_upper_bound(self):
        # Test that a situation in which a variable has an equal lower and
        # upper bound is resolved. I could not quickly reproduce a MWE of the
        # situation, but it *does* arise naturally, specifically in the output
        # of the `compose2` operator in the
        # `SelectLayerByRatioGEQPlainRegionObjects` tool of the
        # `SolarPowerPotentialGloverPark` workflow at commit
        # `8b2f38242860c924d003c8d19dfb781248b53fea` of
        # <https://github.com/quangis/cct>
        # f = TypeSchema(lambda x, y: x ** y [x << A, A << y])
        pass

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
        # The order in which subtypes are applied does not matter. Test
        # inspired by the discussion in Traytel et al (2011).
        A = TypeOperator('A')
        A1 = TypeOperator('A1', supertype=A)
        B = TypeOperator('B')
        f = TypeSchema(lambda α: α ** α ** B)
        self.apply(f, A1, A, result=B)
        self.apply(f, A, A1, result=B)
        self.apply(f, A, B, result=TypeMismatch)

    def test_order_of_subtype_application_variable_return_type(self):
        # The order in which subtypes are applied does not matter, and the most
        # specific type that encompasses both of them is selected.
        A = TypeOperator('A')
        A1 = TypeOperator('A1', supertype=A)
        B = TypeOperator('B')
        f = TypeSchema(lambda α: α ** α ** α)
        self.apply(f, A1, A, result=A)
        self.apply(f, A, A1, result=A)
        self.apply(f, A, B, result=TypeMismatch)

    def test_order_of_subtype_application_with_elimination_constraint(self):
        # The above is true even in the presence of an elimination constraint
        # which may 
        A = TypeOperator('A')
        A1 = TypeOperator('A1', supertype=A)
        B = TypeOperator('B')
        f = TypeSchema(lambda x: x ** x ** x [x << {A, B}])
        self.apply(f, A1, A, result=A)
        self.apply(f, A, A1, result=A)

    def test_order_of_subtype_application_with_subtype_constraints(self):
        f = TypeSchema(lambda α: α ** α ** B [α <= A])
        self.apply(f, A1, A, result=B)
        self.apply(f, A, A1, result=B)

    @unittest.skip("later")
    def test_order_of_subtype_application_with_constraints2(self):
        f = TypeSchema(lambda α: α ** α ** B [α <= A])
        self.apply(f, B, result=ConstraintViolation)

    def test_preservation_of_subtypes_in_constraints(self):
        f = TypeSchema(lambda α, β: α ** α [α << {β, Z(β)}, β <= A])
        self.apply(f, A1, result=A1)
        self.apply(f, Z(A1), result=Z(A1))
        self.apply(f, B, result=ConstraintViolation)

    def test_unification_of_compound_types_in_constraints(self):
        f = TypeSchema(lambda xs, x: xs ** x [xs << {W(x), Z(x)}])
        self.apply(f, W(A), result=A)

    def test_non_unification_of_base_types(self):
        # We can't unify with base types from constraints, as they might be
        # subtypes. So in this case, we know that x is a Z, but we don't know
        # that its parameters is exactly A: that might be too general a bound.
        f = TypeSchema(lambda x: x ** x [x <= Z(A)])
        result = f.apply(TypeVariable())
        self.assertEqual(result.operator, Z)
        self.assertIsInstance(result.params[0], TypeVariable)

    def test_multiple_bounds(self):
        f = TypeSchema(lambda x: (x ** x) ** x)

        # This works because A1 ** A1 is acceptable for A ** A1.
        self.apply(f, A ** A1, result=A1)

        # This doesn't work because the upper bound A1 cannot be reconciled
        # with the lower bound A.
        self.apply(f, A1 ** A, result=TypeMismatch)

    def test_constrain_wildcard(self):
        f = TypeSchema(lambda x: x ** x [x <= _])
        self.apply(f, A, result=A)

    def test_constrain_free_variable(self):
        f = TypeSchema(lambda x, y, z: x ** x [y << {x, z}])
        g = TypeSchema(lambda x, y, z: x ** x [x << {y, z}])
        self.assertRaises(ConstrainFreeVariable,
            TypeSchema.validate_no_free_variables, f)
        self.assertRaises(ConstrainFreeVariable,
            TypeSchema.validate_no_free_variables, g)

    def test_global_subtype_resolution(self):
        f = TypeSchema(lambda x: x ** (x ** x) ** x)
        self.apply(f, A1, A ** A1, result=A1)
        self.apply(f, A, A ** A1, result=A)

    def test_interdependent_types(self):
        f = TypeSchema(lambda α, β: α ** β [α << {Z(β), F(_, β)}])
        self.apply(f, Z(A), result=A)
        self.apply(f, A, result=ConstraintViolation)

    def test_subtyping_of_concrete_functions(self):
        self.assertTrue((A ** A).is_subtype(A1 ** A))
        self.assertTrue((A ** A).is_subtype(A ** Ω))
        self.assertFalse((A ** A).is_subtype(Ω ** A))
        self.assertFalse((A ** A).is_subtype(A ** A1))

    def test_subtyping_of_variables(self):
        x = TypeVariable()
        self.assertEqual(x.is_subtype(x, strict=True), False)
        self.assertEqual(x.is_subtype(x), True)

    def test_subtyping_of_variable_functions(self):
        x = TypeVariable()
        self.assertEqual((x ** A).is_subtype(A1 ** A), None)
        self.assertEqual((A ** x).is_subtype(A ** Ω), None)
        self.assertEqual((A ** A).is_subtype(x ** A), None)
        self.assertEqual((A ** A).is_subtype(A ** x), None)

    def test_subtyping_of_wildcard_functions(self):
        self.assertTrue((_ ** A).is_subtype(A1 ** A))
        self.assertTrue((A ** _).is_subtype(A ** Ω))
        self.assertTrue((A ** A).is_subtype(_ ** A))
        self.assertTrue((A ** A).is_subtype(A ** _))
        self.assertFalse((_ ** Ω).is_subtype(A1 ** A))
        self.assertFalse((A1 ** _).is_subtype(A ** Ω))

    def test_constrained_to_base_type(self):
        # See issue #2, which caused an infinite loop
        f = TypeSchema(lambda x: x ** x [x << {A}])
        g = TypeSchema(lambda x, y: (x ** y) ** y)
        self.apply(g, f)

    def test_constrained_to_compound_type(self):
        # See issue #2
        f = TypeSchema(lambda x: x ** x [x << {Z(A)}])
        g = TypeSchema(lambda x, y: (x ** y) ** y)
        self.apply(g, f)

    def test_unification_of_constraint_with_variables(self):
        # See issue #13
        A, B, C = TypeOperator('A'), TypeOperator('B'), TypeOperator('C')
        R2, R3 = TypeOperator('R2', 2), TypeOperator('R3', 3)
        actual = TypeSchema(lambda x:
            R3(A, x, C) [R2(C, B) << {R2(A, x), R2(C, x)}])
        expected = R3(A, B, C)
        self.assertEqual(actual.instance(), expected.instance())

    def test_timely_constraint_check(self):
        # See issue #13
        f = TypeSchema(lambda r, x: r ** x [r << {F(A, x), F(B, x)}])
        self.apply(f, F(A, B), result=B)

    def test_concrete_base_types_in_constraint_dont_prevent_unifying(self):
        # See #85
        # We used to unify with the skeleton of the final option in any
        # constraint, meaning that, if the last option is F(A), we would unify
        # with F(_). The intention was to avoid unifying with an overly loose
        # subtype constraint. However, this has the undesirable effect that
        # type variables that are subsequently bound are used for unifying,
        # while concrete types aren't.
        A = TypeOperator('A')
        F = TypeOperator('F', params=2)
        f = TypeSchema(lambda x, y: x ** y [y << {F(x, A)}])
        g = TypeSchema(lambda x, y: x ** y [y << {F(x, x)}])
        self.apply(f, A, result=F(A, A))
        self.apply(g, A, result=F(A, A))

    def test_unify_subtypable_constraint_options(self):
        # See issues #11, #85
        # If options in a constraint are merely subtypes of another, they
        # should unify
        A = TypeOperator('A')
        B = TypeOperator('B', supertype=A)
        F = TypeOperator('F', params=2)
        f = TypeSchema(lambda x, y: x ** y [y << {F(A, x), F(B, x)}])
        self.assertEqual(f.apply(A).params[1], A())

    @unittest.skip("not important")
    def test_unify_bottom_types(self):
        # If a variable must be a subtype of a type that has no subtypes, we
        # can already unify with the bottom type.
        # See #85
        A = TypeOperator('A')
        B = TypeOperator('B', supertype=A)
        f = TypeSchema(lambda x: x [x <= A])
        g = TypeSchema(lambda x: x [x <= B])
        self.apply(f, result=TypeVariable)
        self.apply(g, result=B)

    @unittest.skip("later")
    def test_unify_unifiable_constraint_options1(self):
        # In case there are multiple types in the constraint, but they all have
        # the same type operator, we can already unify with that operator, in
        # case that narrows down other constraints.
        # See #85
        A = TypeOperator('A')
        B = TypeOperator('B')
        F = TypeOperator('F', params=1)
        f = TypeSchema(lambda x: x [x << {F(A), F(B)}])
        self.assertEqual(f.instance().operator, F)

    def test_unify_unifiable_constraint_options2(self):
        # See issues #11, #85
        # If all options in a constraint turn out to come down to the same
        # thing, they should unify.
        A = TypeOperator('A')
        F = TypeOperator('F', params=2)
        f = TypeSchema(lambda x, y, z: F(x, y) ** z [z << {F(x, A), F(y, A)}])
        self.apply(f, F(A, A), result=F(A, A))

        g = TypeSchema(lambda x: x [F(A, A) << {F(A, x), F(A, x)}])
        self.apply(g, result=A)

    # def test_asdf3(self):
    #     # See #85
    #     # When there is only a single remaining option in a constraint left, we
    #     # used to unify with its skeleton. That is, if the remaining option is
    #     # F(A), we unified with F(_). Instead, at least for compound types, we
    #     # should unify with the entire type and just pass the subtype
    #     # constraints along. So we would unify with F(x) where x is constrained
    #     # by A. I think.
    #     A = TypeOperator()
    #     TypeOperator(supertype=A)
    #     F = TypeOperator(params=2)
    #     f = TypeSchema(lambda x: x[F(A)])
    #     fi = f.instance()
    #     self.assertEqual(fi.operator, F)
    #     self.assertEqual(fi.params[0], TypeVariable)

    def test_overeager_unification_of_constraint_options(self):
        # See issue #17
        self.assertEqual(F(A, _).is_subtype(F(_, A)), True)
        x = TypeVariable()
        c = EliminationConstraint(x, [F(A, _), F(_, A)])
        self.assertEqual(len(c.alternatives), 2)
        c = EliminationConstraint(x, [F(_, _), F(_, _)])
        self.assertEqual(len(c.alternatives), 1)


    # No longer relevant after acd78d3d681fecbc76751030407af54f8b18d5d6
    # def test_constraints_extraneous_alternatives(self):
    #     # Subtypes should be considered extraneous in constraint alternatives,
    #     # see issue #58
    #     x = TypeVariable()
    #     c = EliminationConstraint(x, [A1, A])
    #     self.assertEqual(c.alternatives, [A()])

    #     # ... regardless of order, see issue #57
    #     x = TypeVariable()
    #     c = Constraint(x, [A, A1])
    #     self.assertEqual(c.alternatives, [A()])

    @unittest.skip("later")
    def test_unification_of_constraint_option_subtypes(self):
        # See issue #16
        f = TypeSchema(lambda x, y: F(x, y) [F(y, A) << {F(A, x), F(A1, x)}])
        self.assertEqual(
            f.instance().params[0],
            A()
        )

    def test_constraint_check_on_intertwined_variables1(self):
        # See issue #18
        f = TypeSchema(lambda x, y, z: (x ** y) ** z [z << {F(x, y)}])
        g = TypeSchema(lambda x, y: x ** y)
        x, y = f.apply(g).params
        self.assertEqual(len(x._constraints), 0)
        self.assertEqual(len(y._constraints), 0)

    def test_constraint_check_on_intertwined_variables2(self):
        # See issue #18
        f = TypeSchema(lambda x, y: F(x, y) ** y)
        g = TypeSchema(lambda a, b: F(a, b) [F(a, b) << {F(a, b)}])
        y = f.apply(g)
        self.assertEqual(len(y._constraints), 0)

    def test_reach_all_constraints(self):
        f = TypeSchema(lambda a, b, c:
            a ** b ** c [c << {b, _}, b << {c, a}, a << {b, c}]
        ).instance()
        self.assertEqual(len(f.params[1].params[1].constraints()), 3)

    def test_reach_all_operators(self):
        f = TypeSchema(lambda a, b, c:
            a ** b ** c [c << {b, _}, b << {a, _}, a << {A, _}]
        ).instance()
        self.assertEqual(f.params[1].params[1].operators(), {A})

    def test_curried_function_signature_same_as_uncurried(self):
        # See issue #53
        self.assertEqual(
            A1 ** A2 ** A,
            (A1, A2) ** A
        )


if __name__ == '__main__':
    unittest.main()
