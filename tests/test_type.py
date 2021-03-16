import unittest

from transformation_algebra import error
from transformation_algebra.type import Operator, Schema, _, VariableTerm

Any = Operator('Any')
Ord = Operator('Ord', supertype=Any)
Bool = Operator('Bool', supertype=Ord)
Str = Operator('Str', supertype=Ord)
Int = Operator('Int', supertype=Ord)
UInt = Operator('UInt', supertype=Int)
T = Operator('T', 1)
Set = Operator('Set', 1)
Map = Operator('Map', 2)


class TestType(unittest.TestCase):

    def apply(self, f, x, result=None):
        """
        Test the application of an argument to a function.
        """
        f = f.instance()
        x = x.instance()

        if isinstance(result, type) and issubclass(result, Exception):
            self.assertRaises(result, f, x)
        else:
            actual = f(x).plain()
            expected = result.plain()
            self.assertEqual(actual, expected)

    def test_apply_non_function(self):
        self.apply(Int.instance(), Int, error.NonFunctionApplication)

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
        wrap = Schema(lambda α: α ** T(α))
        self.apply(wrap, Int, T(Int))

    def test_compose(self):
        compose = Schema(lambda x, y, z: (y ** z) ** (x ** y) ** (x ** z))
        self.apply(
            compose(Int ** Str), Str ** Int,
            Str ** Str)

    def test_compose_subtype(self):
        compose = Schema(lambda x, y, z: (y ** z) ** (x ** y) ** (x ** z))
        self.apply(
            compose(Int ** Str), Str ** UInt,
            Str ** Str)

    def test_variable_subtype_mismatch(self):
        f = Schema(lambda x: (x ** Int) ** x)
        self.apply(f, Int ** Any, error.SubtypeMismatch)

    def test_functions_as_arguments1(self):
        swap = Schema(lambda α, β, γ: (α ** β ** γ) ** (β ** α ** γ))
        f = Schema(lambda x: Bool ** x ** x)
        self.apply(swap(f, UInt), Bool, UInt)

    def test_functions_as_arguments2(self):
        id = Schema(lambda x: x ** x)
        f = Int ** Int
        x = UInt
        self.apply(id(f), x, Int)

    def test_order_of_subtype_application(self):
        """
        This test is inspired by Traytel et al (2011).
        """
        leq = Schema(lambda α: α ** α ** Bool)
        self.apply(leq(UInt), Int, Bool)
        self.apply(leq(Int), UInt, Bool)
        self.apply(leq(Int), Bool, error.SubtypeMismatch)

    def test_order_of_subtype_application_with_constraints(self):
        leq = Schema(lambda α: α ** α ** Bool | α @ [Ord, Bool])
        self.apply(leq(Int), UInt, Bool)
        self.apply(leq, Any, error.ViolatedConstraint)

    def test_violation_of_constraints(self):
        sum = Schema(lambda α: α ** α | α @ [Int, Set(Int)])
        self.apply(sum, Set(UInt), Set(UInt))
        self.apply(sum, Bool, error.ViolatedConstraint)

    def test_preservation_of_basic_subtypes_in_constraints(self):
        f = Schema(lambda x: x ** x | x @ [Any])
        self.apply(f, Int, Int)

    def test_unification_of_compound_types_in_constraints(self):
        f = Schema(lambda xs, x: xs ** x | xs @ [Set(x), T(x)])
        self.apply(f, T(Int), Int)

    def test_non_unification_of_base_types(self):
        """
        We cannot unify with base types from constraints, because they might
        also be subtypes. So in this case, we know that x is a Map, but we
        don't know that its parameters are exactly Str and Int: that might be
        too loose a bound.
        """
        f = Schema(lambda x: x ** x | x @ [Map(Str, Int)])
        result = f(_).plain().follow()  # TODO make follow unnecessary
        self.assertEqual(result.operator, Map)

    def test_multiple_bounds1(self):
        """
        This works because UInt ** UInt is acceptable for Int ** UInt.
        """
        f = Schema(lambda x: (x ** x) ** x)
        self.apply(f, Int ** UInt, UInt)

    def test_multiple_bounds2(self):
        """
        This doesn't work because the upper bound UInt cannot be reconciled
        with the lower bound Int.
        """
        f = Schema(lambda x: (x ** x) ** x)
        self.apply(f, UInt ** Int, error.SubtypeMismatch)

    def test_global_subtype_resolution(self):
        f = Schema(lambda x: x ** (x ** x) ** x)
        self.apply(f(UInt), Int ** UInt, UInt)
        self.apply(f(Int), Int ** UInt, Int)

    def test_subtyping_of_concrete_functions(self):
        self.assertTrue(Int ** Int <= UInt ** Int)
        self.assertTrue(Int ** Int <= Int ** Any)
        self.assertFalse(Int ** Int <= Any ** Int)
        self.assertFalse(Int ** Int <= Int ** UInt)

    def test_subtyping_of_variable_functions(self):
        x = VariableTerm()
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
