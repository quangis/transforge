import unittest

from quangis import error
from quangis.transformation.type import Operator, Σ, instance

Any = Operator('Any')
Bool = Operator('Bool')
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
            self.assertEqual(f(x).plain, instance(result).plain)

    def test_apply_non_function(self):
        self.apply(
            Σ(Int),
            Σ(Int),
            error.NonFunctionApplication)

    def test_basic_match(self):
        self.apply(
            Σ(Int ** Str),
            Σ(Int),
            Str)

    def test_basic_mismatch(self):
        self.apply(
            Σ(Int ** Str),
            Σ(Str),
            error.SubtypeMismatch)

    def test_basic_sub_match(self):
        self.apply(
            Σ(Any ** Str),
            Σ(Int),
            Str)

    def test_basic_sub_mismatch(self):
        self.apply(
            Σ(Int ** Str),
            Σ(Any),
            error.SubtypeMismatch)

    def test_compound_match(self):
        self.apply(
            Σ(T(Int) ** Str),
            Σ(T(Int)),
            Str)

    def test_compound_mismatch(self):
        self.apply(
            Σ(T(Int) ** Str),
            Σ(T(Str)),
            error.SubtypeMismatch)

    def test_compound_sub_match(self):
        self.apply(
            Σ(T(Any) ** Str),
            Σ(T(Int)),
            Str)

    def test_compound_sub_mismatch(self):
        self.apply(
            Σ(T(Int) ** Str),
            Σ(T(Any)),
            error.SubtypeMismatch)

    def test_variable(self):
        self.apply(
            Σ(lambda α: α ** T(α)),
            Σ(Int),
            T(Int))
#
#    def test_compose(self):
#        self.apply(
#            ((var.y ** var.z) ** (var.x ** var.y) ** (var.x ** var.z)),
#            Int ** Str,
#            Str ** Int,
#            result=Str ** Str)
#
#    def test_simple_subtype_match(self):
#        self.apply(
#            (var.any ** Any, var.any.subtype(Any)),
#            Int,
#            result=Any)
#
#    def test_simple_subtype_mismatch(self):
#        self.apply(Int ** Any, Any, result=error.TypeMismatch)
#
#    def test_complex_subtype_match(self):
#        self.apply(
#            ((var.any ** var.any) ** Any, var.any.subtype(Any)),
#            Int ** Int, result=Any)
#
#    def test_complex_subtype_mismatch(self):
#        self.apply((Int ** Any) ** Any, Any ** Any, result=error.TypeMismatch)
#
#    def test_variable_subtype_match(self):
#        self.apply(
#            ((var.x ** var.y) ** var.x, var.y.subtype(Any)),
#            (Int ** Any), result=Int)
#
#    def test_variable_subtype_mismatch(self):
#        self.apply(
#            ((var.x ** var.y) ** var.x, var.y.subtype(Int)),
#            (Int ** Any), result=error.ViolatedConstraint)
#
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
#
#    def test_compose_something(self):
#        self.apply(
#            (var.x ** var.x, (var.x ** var.x).subtype(Any ** var._)),
#            Int,
#            result=Int)
#

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
