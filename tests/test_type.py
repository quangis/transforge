import unittest

from transformation_algebra import error
from transformation_algebra.type import TypeOperator, TypeSchema, TypeVar, _

Any = TypeOperator('Any')
Ord = TypeOperator('Ord', supertype=Any)
Bool = TypeOperator('Bool', supertype=Ord)
Str = TypeOperator('Str', supertype=Ord)
Int = TypeOperator('Int', supertype=Ord)
UInt = TypeOperator('UInt', supertype=Int)
T = TypeOperator('T', 1)
Set = TypeOperator('Set', 1)
Map = TypeOperator('Map', 2)


class TestType(unittest.TestCase):

    def apply(self, f, x, result=None):
        """
        Test the application of an argument to a function.
        """
        f = f.instance()
        x = x.instance()

        if isinstance(result, type) and issubclass(result, Exception):
            self.assertRaises(result, lambda x: f.apply(x), x)
        else:
            actual = f.apply(x)
            expected = result.instance()
            self.assertEqual(actual, expected)

    def test_apply_non_function(self):
        self.apply(Int, Int, error.NonFunctionApplication)

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
            compose.instance().apply(Int ** Str), Str ** Int,
            Str ** Str)

    def test_compose_subtype(self):
        compose = TypeSchema(lambda x, y, z: (y ** z) ** (x ** y) ** (x ** z))
        self.apply(
            compose.instance().apply(Int ** Str), Str ** UInt,
            Str ** Str)

    def test_variable_subtype_mismatch(self):
        f = TypeSchema(lambda x: (x ** Int) ** x)
        self.apply(f, Int ** Any, error.SubtypeMismatch)

    def test_functions_as_arguments1(self):
        swap = TypeSchema(lambda α, β, γ: (α ** β ** γ) ** (β ** α ** γ))
        f = TypeSchema(lambda x: Bool ** x ** x)
        self.apply(swap.instance().apply(f.instance()).apply(UInt()), Bool, UInt)

    def test_functions_as_arguments2(self):
        id = TypeSchema(lambda x: x ** x)
        f = Int ** Int
        x = UInt
        self.apply(id.instance().apply(f.instance()), x, Int)

    def test_order_of_subtype_application(self):
        """
        This test is inspired by Traytel et al (2011).
        """
        leq = TypeSchema(lambda α: α ** α ** Bool)
        self.apply(leq.instance().apply(UInt()), Int(), Bool())
        self.apply(leq.instance().apply(Int()), UInt, Bool)
        self.apply(leq.instance().apply(Int()), Bool, error.SubtypeMismatch)

    def test_order_of_subtype_application_with_constraints(self):
        leq = TypeSchema(lambda α: α ** α ** Bool | α @ [Ord, Bool])
        self.apply(leq.instance().apply(Int()), UInt, Bool)
        self.apply(leq, Any, error.ViolatedConstraint)

    def test_violation_of_constraints(self):
        sum = TypeSchema(lambda α: α ** α | α @ [Int, Set(Int)])
        self.apply(sum, Set(UInt), Set(UInt))
        self.apply(sum, Bool, error.ViolatedConstraint)

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
        result = f.instance().apply(TypeVar())
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

    def test_global_subtype_resolution(self):
        f = TypeSchema(lambda x: x ** (x ** x) ** x)
        self.apply(f.instance().apply(UInt()), Int ** UInt, UInt)
        self.apply(f.instance().apply(Int()), Int ** UInt, Int)

    def test_interdependent_types(self):
        f = TypeSchema(lambda α, β: α ** β | α @ [Set(β), Map(_, β)])
        self.apply(f, Set(Int), Int)
        self.apply(f, Int, error.ViolatedConstraint)

    def test_subtyping_of_concrete_functions(self):
        self.assertTrue(Int ** Int <= UInt ** Int)
        self.assertTrue(Int ** Int <= Int ** Any)
        self.assertFalse(Int ** Int <= Any ** Int)
        self.assertFalse(Int ** Int <= Int ** UInt)

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


if __name__ == '__main__':
    unittest.main()
