import unittest

from transforge.type import (
    Type, TypeOperator, TypeSchema, TypeVariable, _, with_parameters,
    FunctionApplicationError, TypeMismatch, Top, Bottom,
    ConstraintViolation, ConstrainFreeVariableError, EliminationConstraint,
    Direction, UnexpectedVariableError)


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

    def test_deduce_variable_types(self):
        # When we apply an `x ** x` to an `A`, deduce `x >= A`; when we apply
        # an `A ** A` to an `x`, deduce `x <= A`.
        A = TypeOperator('A')
        x = TypeVariable()
        (A ** A).apply(x)
        self.assertEqual(x.upper, A)
        self.assertEqual(x.lower, None)

        y = TypeVariable()
        (y ** y).apply(A)
        self.assertEqual(y.upper, None)
        self.assertEqual(y.lower, A)

    def test_compose(self):
        A, B = TypeOperator('A'), TypeOperator('B')
        A1 = TypeOperator('A1', supertype=A)
        B1 = TypeOperator('B1', supertype=B)
        compose = TypeSchema(lambda x, y, z: (y ** z) ** (x ** y) ** (x ** z))
        self.apply(compose, A ** B, B ** A)  # , result=B ** B)
        self.apply(compose, A ** B, B ** A1)  # , result=B ** B)
        self.apply(compose, A ** B, B ** B1, result=TypeMismatch)
        self.apply(compose, A1 ** B, B ** A, result=TypeMismatch)

    def test_equal_lower_and_upper_bound(self):
        # Test that a situation in which a variable has an equal lower and
        # upper bound is resolved. This does sometimes arise naturally,
        # specifically in the outputs of the `compose2` operators in the
        # <https://github.com/quangis/cct> repository. Initial reproduction:
        # cct.parse("select (compose2 notj leq)").x.type.output()
        A, R = TypeOperator('A'), TypeOperator('R', params=2)
        select = TypeSchema(lambda x, y, rel: (x ** y ** A) ** rel ** y ** rel
                [rel << (R(x, _), R(_, x))]).instance()
        compose2 = TypeSchema(lambda α, β, γ, δ: (β ** γ) ** (δ ** α ** β)
            ** (δ ** α ** γ)).instance()
        notj = A ** A
        leq = A ** A ** A
        select.apply(compose2.apply(notj).apply(leq))
        self.assertEqual(compose2.output(), A())

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
        self.assertRaises(ConstrainFreeVariableError,
            TypeSchema.validate_no_free_variables, f)
        self.assertRaises(ConstrainFreeVariableError,
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
        # See #85
        # When there is only a single remaining option in a constraint left, we
        # used to unify with its skeleton. That is, if the remaining option is
        # F(A), we unified with F(_). Instead, at least for compound types, we
        # should unify with the entire type and just pass the subtype
        # constraints along. So we would unify with F(x) where x is constrained
        # by A. I think.
        # A = TypeOperator()
        # TypeOperator(supertype=A)
        # F = TypeOperator(params=2)
        # f = TypeSchema(lambda x: x[F(A)])
        # fi = f.instance()
        # self.assertEqual(fi.operator, F)
        # self.assertEqual(fi.params[0], TypeVariable)

    def test_overeager_unification_of_constraint_options(self):
        # See issue #17
        self.assertEqual(F(A, _).is_subtype(F(_, A)), True)
        x = TypeVariable()
        c = EliminationConstraint(x, [F(A, _), F(_, A)])
        self.assertEqual(len(c.alternatives), 2)
        c = EliminationConstraint(x, [F(_, _), F(_, _)])
        self.assertEqual(len(c.alternatives), 1)

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
            a ** b ** c [c << {b, _}, b << {c, a}, a << {b, c}]).instance()
        self.assertEqual(len(f.params[1].params[1].constraints()), 3)

    def test_reach_all_operators(self):
        f = TypeSchema(lambda a, b, c:
            a ** b ** c [c << {b, _}, b << {a, _}, a << {A, _}]).instance()
        self.assertEqual(f.params[1].params[1].operators(), {A})

    def test_curried_function_signature_same_as_uncurried(self):
        # See issue #53
        self.assertEqual(
            A1 ** A2 ** A,
            (A1, A2) ** A
        )

    def test_print_schematic_types_with_parameter_name(self):
        f = TypeSchema(lambda foo: foo)
        self.assertEqual(str(f), "foo")

    def test_print_subtype_bounds(self):
        A = TypeOperator('A')
        x, y = TypeVariable(), TypeVariable()
        (A ** A).apply(x)
        self.assertEqual(
            x.text(with_constraints=True, labels={x: "x"}),
            "x [x <= A]")
        (y ** y ** y).apply(A)
        self.assertEqual(
            y.text(with_constraints=True, labels={y: "y"}),
            "y [y >= A]")

    def test_print_subtype_constraints(self):
        A = TypeOperator('A')
        f = TypeSchema(lambda x: x [x <= A])
        g = TypeSchema(lambda x: x [A <= x])
        self.assertEqual(str(f), "x [x <= A]")
        self.assertEqual(str(g), "x [A <= x]")

    def test_print_elimination_constraints(self):
        A = TypeOperator('A')
        B = TypeOperator('B')
        f = TypeSchema(lambda x: x [x << (A, B)])
        self.assertEqual(str(f), "x [x << (A, B)]")

    def test_print_subtype_bounds_from_elimination_constraint(self):
        A = TypeOperator('A')
        f = TypeSchema(lambda x: x [x << A])  # TODO: since A has no subtypes,
        # could bind already?
        g = TypeSchema(lambda x: x [A << x])
        self.assertEqual(str(f), "x [x <= A]")
        self.assertEqual(str(g), "A")  # this subtype bound should be fixed

    def test_inclusive_subtype(self):
        A = TypeOperator('A')
        F = TypeOperator('F', params=2)
        self.assertEqual(
            set(F(A, A).successors(Direction.DOWN, include_bottom=True)),
            {F(Bottom, A), F(A, Bottom)}
        )
        self.assertEqual(
            set(F(Bottom, A).successors(Direction.DOWN, include_bottom=True)),
            {F(Bottom, Bottom)}
        )
        self.assertEqual(
            set(F(Bottom, Bottom).successors(Direction.DOWN,
                include_bottom=True)),
            {Bottom()}
        )
        self.assertEqual(
            set(F(Top, Top).successors(Direction.DOWN, include_bottom=True)),
            {F(Bottom, Top), F(Top, Bottom)}
        )

    def test_concretization(self):
        F = TypeOperator('F', params=1)
        G = TypeSchema(lambda x: F(x))
        self.assertEqual(F(F(_)).concretize(True), F(F(Top)))
        self.assertRaises(UnexpectedVariableError, G.concretize)

    def test_top_type_is_usable_as_a_function(self):
        # cf. <https://github.com/quangis/transformation-algebra/issues/100>
        F = TypeOperator()
        self.apply(Top, F, result=Top)

    def test_top_and_bottom(self):
        F = TypeOperator()
        self.assertTrue(F().match(Top(), subtype=True))
        self.assertTrue(Bottom().match(F(), subtype=True))
        self.assertFalse(F().match(Bottom(), subtype=True))
        self.assertFalse(Top().match(F(), subtype=True))
        self.assertTrue(Top().match(Top(), subtype=True))
        self.assertTrue(Bottom().match(Bottom(), subtype=True))

    def test_top_is_not_contained(self):
        F = TypeOperator()
        G = TypeOperator(params=1)
        self.assertFalse(Top() in G(F()))
        self.assertFalse(Bottom() in G(F()))
        self.assertTrue(F() in G(F()))

    def test_top_and_bottom_apply_in_compound_types(self):
        # cf. <https://github.com/quangis/transformation-algebra/issues/108>
        A = TypeOperator()
        F = TypeOperator(params=1)

        f = A ** A
        self.apply(f, Bottom, result=A)
        self.apply(f, Top, result=TypeMismatch)

        g = F(A) ** A
        self.apply(g, F(Bottom), result=A)
        self.apply(g, F(Top), result=TypeMismatch)

    def test_top_and_bottom_apply(self):
        # cf. <https://github.com/quangis/transformation-algebra/issues/107>
        A = TypeOperator()
        f = TypeSchema(lambda x: x ** x ** x)
        self.apply(f, Top, A, result=Top)
        self.apply(f, Bottom, A, result=A)
        self.apply(f, A, Top, result=Top)
        self.apply(f, A, Bottom, result=A)

    @unittest.skip("see issue #79")
    def test_intersection_type_order_is_irrelevant(self):
        A, B, C = (TypeOperator() for _ in range(3))
        self.assertMatch(A & B & C, C & B & A, subtype=False)

    @unittest.skip("see issue #79")
    def test_intersection_type_subtype(self):
        A, B, C = (TypeOperator() for _ in range(3))
        F = TypeOperator(params=1)
        self.assertMatch(A & B & C, B, subtype=True)
        self.assertMatch(A & B & C, A & B, subtype=True)
        self.assertMatch(A & B & C, A & C, subtype=True)
        self.assertMatch(F(A & B & C), F(A), subtype=True)
        self.assertNotMatch(A, A & B, subtype=True)
        self.assertNotMatch(F(A), F(A & B), subtype=True)
        self.assertNotMatch(F(A) & F(B), F(A & B), subtype=True)

    @unittest.skip("see issue #79")
    def test_empty_intersection(self):
        A, B, C = (TypeOperator() for _ in range(3))
        F = TypeOperator(params=1)
        G = TypeOperator(params=1)
        self.assertMatch(F(A) & G(A), Bottom)

    def test_wildcard_status_lost_once_subtype_constraint_is_present(self):
        A = TypeOperator()
        w = _.instance()
        self.assertTrue(w.wildcard)
        self.assertEqual(w.upper, None)
        (A ** A).apply(w)
        self.assertIsInstance(w, TypeVariable)
        self.assertNotEqual(w.upper, None)
        self.assertFalse(w.wildcard)


if __name__ == '__main__':
    unittest.main()
