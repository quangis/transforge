import unittest

from transformation_algebra import error
from transformation_algebra.type import \
    Type, TypeSchema, TypeVar, _

τ = Type.declare

Any = Type.declare('Any')
Ord = Type.declare('Ord', supertype=Any)
Bool = Type.declare('Bool', supertype=Ord)
Str = Type.declare('Str', supertype=Ord)
Int = Type.declare('Int', supertype=Ord)
UInt = Type.declare('UInt', supertype=Int)
T = Type.declare('T', 1)
Set = Type.declare('Set', 1)
Map = Type.declare('Map', 2)


class TestType(unittest.TestCase):

    def apply(self, f, x, result=None):
        """
        Test the application of an argument to a function.
        """
        f = f.instance()
        x = x.instance()

        if isinstance(result, type) and issubclass(result, Exception):
            self.assertRaises(result, lambda x: f.apply(x), x)
        elif result:
            actual = f.apply(x)
            expected = result.instance()
            self.assertEqual(actual, expected)
        else:
            f.apply(x)

    def test_apply_non_function(self):
        self.apply(Int, Int, error.FunctionApplicationError)

    def test_basic_match(self):
        f = Int ** Str
        self.apply(f, Int, Str)

    def test_basic_mismatch(self):
        f = Int ** Str
        self.apply(f, Str, error.SubtypeMismatch)

    def test_basic_sub_match(self):
        f = Any ** Str
        self.apply(f, Int, Str)

    def test_basic_sub_mismatch(self):
        f = Int ** Str
        self.apply(f, Any, error.SubtypeMismatch)

    def test_compound_match(self):
        f = T(Int) ** Str
        self.apply(f, T(Int), Str)

    def test_compound_mismatch(self):
        f = T(Int) ** Str
        self.apply(f, T(Str), error.SubtypeMismatch)

    def test_compound_sub_match(self):
        f = T(Any) ** Str
        self.apply(f, T(Int), Str)

    def test_compound_sub_mismatch(self):
        f = T(Int) ** Str
        self.apply(f, T(Any), error.SubtypeMismatch)

    def test_variable(self):
        wrap = TypeSchema(lambda α: α ** T(α))
        self.apply(wrap, Int, T(Int))

    def test_compose(self):
        compose = TypeSchema(lambda x, y, z: (y ** z) ** (x ** y) ** (x ** z))
        self.apply(
            compose.apply(Int ** Str), Str ** Int,
            Str ** Str)

    def test_compose_subtype(self):
        compose = TypeSchema(lambda x, y, z: (y ** z) ** (x ** y) ** (x ** z))
        self.apply(
            compose.apply(Int ** Str), Str ** UInt,
            Str ** Str)

    def test_variable_subtype_mismatch(self):
        f = TypeSchema(lambda x: (x ** Int) ** x)
        self.apply(f, Int ** Any, error.SubtypeMismatch)

    def test_functions_as_arguments1(self):
        swap = TypeSchema(lambda α, β, γ: (α ** β ** γ) ** (β ** α ** γ))
        f = TypeSchema(lambda x: Bool ** x ** x)
        self.apply(swap.apply(f).apply(UInt), Bool, UInt)

    def test_functions_as_arguments2(self):
        id = TypeSchema(lambda x: x ** x)
        f = Int ** Int
        x = UInt
        self.apply(id.apply(f), x, Int)

    def test_order_of_subtype_application(self):
        """
        This test is inspired by Traytel et al (2011).
        """
        leq = TypeSchema(lambda α: α ** α ** Bool)
        self.apply(leq.apply(UInt), Int(), Bool())
        self.apply(leq.apply(Int), UInt, Bool)
        self.apply(leq.apply(Int), Bool, error.SubtypeMismatch)

    def test_order_of_subtype_application_with_constraints(self):
        leq = TypeSchema(lambda α: α ** α ** Bool | α @ [Ord, Bool])
        self.apply(leq.apply(Int), UInt, Bool)
        self.apply(leq, Any, error.ConstraintViolation)

    def test_violation_of_constraints(self):
        sum = TypeSchema(lambda α: α ** α | α @ [Int, Set(Int)])
        self.apply(sum, Set(UInt), Set(UInt))
        self.apply(sum, Bool, error.ConstraintViolation)

    def test_preservation_of_basic_subtypes_in_constraints(self):
        f = TypeSchema(lambda x: x ** x | x @ [Any])
        self.apply(f, Int, Int)

    def test_unification_of_compound_types_in_constraints(self):
        f = TypeSchema(lambda xs, x: xs ** x | xs @ [Set(x), T(x)])
        self.apply(f, T(Int), Int)

    def test_non_unification_of_base_types(self):
        """
        We cannot unify with base types from constraints, because they might
        also be subtypes. So in this case, we know that x is a Map, but we
        don't know that its parameters are exactly Str and Int: that might be
        too loose a bound.
        """
        f = TypeSchema(lambda x: x ** x | x @ [Map(Str, Int)])
        result = f.apply(TypeVar())
        self.assertEqual(result.operator, Map)

    def test_multiple_bounds1(self):
        """
        This works because UInt ** UInt is acceptable for Int ** UInt.
        """
        f = TypeSchema(lambda x: (x ** x) ** x)
        self.apply(f, Int ** UInt, UInt)

    def test_multiple_bounds2(self):
        """
        This doesn't work because the upper bound UInt cannot be reconciled
        with the lower bound Int.
        """
        f = TypeSchema(lambda x: (x ** x) ** x)
        self.apply(f, UInt ** Int, error.SubtypeMismatch)

    def test_constrain_wildcard(self):
        f = TypeSchema(lambda x: x ** x | x @ [_])
        self.apply(f, Int, Int)

    def test_constrain_free_variable(self):
        f = TypeSchema(lambda x, y, z: x ** x | y @ [x, z])
        g = TypeSchema(lambda x, y, z: x ** x | x @ [y, z])
        self.assertRaises(error.ConstrainFreeVariable, TypeSchema.instance, f)
        self.assertRaises(error.ConstrainFreeVariable, TypeSchema.instance, g)

    def test_global_subtype_resolution(self):
        f = TypeSchema(lambda x: x ** (x ** x) ** x)
        self.apply(f.apply(UInt), Int ** UInt, UInt)
        self.apply(f.apply(Int), Int ** UInt, Int)

    def test_interdependent_types(self):
        f = TypeSchema(lambda α, β: α ** β | α @ [Set(β), Map(_, β)])
        self.apply(f, Set(Int), Int)
        self.apply(f, Int, error.ConstraintViolation)

    def test_subtyping_of_concrete_functions(self):
        self.assertTrue(Int ** Int <= UInt ** Int)
        self.assertTrue(Int ** Int <= Int ** Any)
        self.assertFalse(Int ** Int <= Any ** Int)
        self.assertFalse(Int ** Int <= Int ** UInt)

    def test_subtyping_of_variables(self):
        x = TypeVar()
        self.assertEqual(x < x, False)
        self.assertEqual(x <= x, True)

    def test_subtyping_of_variable_functions(self):
        x = TypeVar()
        self.assertEqual(x ** Int <= UInt ** Int, None)
        self.assertEqual(Int ** x <= Int ** Any, None)
        self.assertEqual(Int ** Int <= x ** Int, None)
        self.assertEqual(Int ** Int <= Int ** x, None)

    def test_subtyping_of_wildcard_functions(self):
        self.assertTrue(_ ** Int <= UInt ** Int)
        self.assertTrue(Int ** _ <= Int ** Any)
        self.assertTrue(Int ** Int <= _ ** Int)
        self.assertTrue(Int ** Int <= Int ** _)
        self.assertFalse(_ ** Any <= UInt ** Int)
        self.assertFalse(UInt ** _ <= Int ** Any)

    def test_constrained_to_base_type(self):
        # Addresses issue #2, which caused an infinite loop
        A = Type.declare('A')
        f = TypeSchema(lambda x: x ** x | x @ A)
        g = TypeSchema(lambda x, y: (x ** y) ** y)
        self.apply(g, f)

    def test_constrained_to_compound_type(self):
        # Same as before, making sure that compound types also work
        A = Type.declare('A')
        F = Type.declare('F', params=1)
        f = TypeSchema(lambda x: x ** x | x @ F(A))
        g = TypeSchema(lambda x, y: (x ** y) ** y)
        self.apply(g, f)

    def test_unification_of_constraint_with_variables(self):
        # See issue #13
        A, B, C, R2, R3 = τ('A'), τ('B'), τ('C'), τ('R2', 2), τ('R3', 3)
        actual = TypeSchema(lambda x:
            R3(A, x, C) | R2(C, B) @ [R2(A, x), R2(C, x)])
        expected = R3(A, B, C)
        self.assertEqual(actual.instance(), expected.instance())

    def test_timely_constraint_check(self):
        # See issue #13
        A, B, F = Type.declare('A'), Type.declare('B'), Type.declare('F', 2)
        f = TypeSchema(lambda r, x: r ** x | r @ [F(A, x), F(B, x)])
        actual = f.apply(F(A, B))
        expected = B
        self.assertEqual(actual.instance(), expected.instance())

    def test_unification_of_constraint_options(self):
        # See issue #11
        A = Type.declare('A')
        F = Type.declare('F', params=2)
        actual = TypeSchema(lambda x: x | F(A, A) @ [F(A, x), F(A, x)])
        expected = A
        self.assertEqual(actual.instance(), expected.instance())

    def test_overeager_unification_of_constraint_options(self):
        # See issue #17
        A = Type.declare('A')
        F = Type.declare('F', params=2)
        self.assertEqual(F(A, _) <= F(_, A), True)
        x = TypeVar()
        c = x @ [F(A, _), F(_, A)]
        self.assertEqual(len(c.alternatives), 2)
        c = x @ [F(_, _), F(_, _)]
        self.assertEqual(len(c.alternatives), 1)

    def test_unification_of_constraint_option_subtypes(self):
        # See issue #16
        A = Type.declare('A')
        F = Type.declare('F', params=2)
        B = Type.declare('B', supertype=A)
        f = TypeSchema(lambda x, y: F(x, y) | F(y, A) @ [F(A, x), F(B, x)])
        actual = f.instance().params[0]
        expected = A.instance()
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
