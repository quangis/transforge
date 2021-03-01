import unittest

from quangis import error
from quangis.transformation.type import Operator, Σ, instance

Any = Operator('Any')
Bool = Operator('Bool', supertype=Any)
Str = Operator('Str', supertype=Any)
Int = Operator('Int', supertype=Any)
UInt = Operator('UInt', supertype=Int)
T = Operator('T', 1)


class TestType(unittest.TestCase):

    def apply(self, f, x, result=None):
        """
        Test the application of an argument to a function.
        """

        if isinstance(result, type) and issubclass(result, Exception):
            self.assertRaises(result, f, x)
        else:
            self.assertEqual(
                f(x).plain.specify(),
                instance(result).plain.specify())

    def test_apply_non_function(self):
        self.apply(Σ(Int), Σ(Int), error.NonFunctionApplication)

    def test_basic_match(self):
        f = Σ(Int ** Str)
        self.apply(f, Int, Str)

    def test_basic_mismatch(self):
        f = Σ(Int ** Str)
        self.apply(f, Str, error.SubtypeMismatch)

    def test_basic_sub_match(self):
        f = Σ(Any ** Str)
        self.apply(f, Int, Str)

    def test_basic_sub_mismatch(self):
        f = Σ(Int ** Str)
        self.apply(f, Any, error.SubtypeMismatch)

    def test_compound_match(self):
        f = Σ(T(Int) ** Str)
        self.apply(f, T(Int), Str)

    def test_compound_mismatch(self):
        f = Σ(T(Int) ** Str)
        self.apply(f, T(Str), error.SubtypeMismatch)

    def test_compound_sub_match(self):
        f = Σ(T(Any) ** Str)
        self.apply(f, T(Int), Str)

    def test_compound_sub_mismatch(self):
        f = Σ(T(Int) ** Str)
        self.apply(f, T(Any), error.SubtypeMismatch)

    def test_variable(self):
        wrap = Σ(lambda α: α ** T(α))
        self.apply(wrap, Int, T(Int))

    def test_compose(self):
        compose = Σ(lambda x, y, z: (y ** z) ** (x ** y) ** (x ** z))
        self.apply(
            compose(Int ** Str), Str ** Int,
            Str ** Str)

    def test_compose_subtype(self):
        compose = Σ(lambda x, y, z: (y ** z) ** (x ** y) ** (x ** z))
        self.apply(
            compose(Int ** Str), Str ** UInt,
            Str ** Str)

    def test_variable_subtype_match(self):
        f = Σ(lambda x: (x ** Any) ** x)
        self.apply(f, Int ** Int, Int)

    def test_variable_subtype_mismatch(self):
        f = Σ(lambda x: (x ** Int) ** x)
        self.apply(f, Int ** Any, error.SubtypeMismatch)

#    def test_simple_constraints_passed(self):
#        self.apply(
#            (var.x ** var.x, var.x.subtype(Int, Str)),
#            Int,
#            result=Int
#        )
#
#    def test_simple_constraints_subtype_passed(self):
#        self.apply(
#            (var.x ** var.x, var.x.subtype(Any)),
#            Int,
#            result=Int
#        )
#
#    def test_simple_constraints_subtype_violated(self):
#        self.apply(
#            (var.x ** var.x, var.x.subtype(Int, Str)),
#            Any,
#            result=error.ViolatedConstraint
#        )

    def test_weird(self):
        swap = Σ(lambda α, β, γ: (α ** β ** γ) ** (β ** α ** γ))
        f = Σ(Int ** Int ** Int)
        x = UInt
        self.apply(swap(f, x), x, Int)

    def test_functions_as_arguments(self):
        id = Σ(lambda x: x ** x)
        f = Σ(Int ** Int)
        x = UInt
        self.apply(id(f), x, Int)

    def test_order_of_subtype_application(self):
        """
        This test is inspired by Traytel et al (2011).
        """
        leq = Σ(lambda α: α ** α ** Bool)
        self.apply(leq(UInt), Int, Bool)
        self.apply(leq(Int), UInt, Bool)
        self.apply(leq(Int), Bool, error.SubtypeMismatch)


if __name__ == '__main__':
    unittest.main()
