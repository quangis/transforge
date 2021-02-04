import unittest

from quangis import error
from quangis.transformation.type import TypeOperator, Variables, Definition

var = Variables()
Any = TypeOperator.Any()
Int = TypeOperator.Int(supertype=Any)
Str = TypeOperator.Str(supertype=Any)
T = TypeOperator.T


class TestType(unittest.TestCase):

    @staticmethod
    def define(x) -> Definition:
        """
        Obtain a test definition.
        """
        if isinstance(x, tuple):
            return Definition('test', *x)
        else:
            return Definition('test', x)

    def apply(self, f, *xs, result=None):
        """
        Test if the application of one (or more) arguments to a function gives
        the correct result.
        """

        f = self.define(f).instance()
        for i in range(0, len(xs)-1):
            x = self.define(xs[i]).instance()
            f = f.apply(x)
        x = self.define(xs[-1]).instance()

        if isinstance(result, type) and issubclass(result, Exception):
            self.assertRaises(result, f.apply, x)
        else:
            self.assertEqual(f.apply(x), result)

    def test_apply_non_function(self):
        self.apply(Int, Int, result=error.NonFunctionApplication)

    def test_simple_function_match(self):
        self.apply(Int ** Str, Int, result=Str)

    def test_simple_function_mismatch(self):
        self.apply(Int ** Str, Str, result=error.TypeMismatch)

    def test_complex_function_match(self):
        self.apply(T(Int) ** T(Str), T(Int), result=T(Str))

    def test_variable(self):
        self.apply(var.x ** T(var.x), Int, result=T(Int))

    def test_compose(self):
        self.apply(
            ((var.y ** var.z) ** (var.x ** var.y) ** (var.x ** var.z)),
            Int ** Str,
            Str ** Int,
            result=Str ** Str)

    def test_simple_subtype_match(self):
        self.apply(Any ** Any, Int, result=Any)

    def test_simple_subtype_mismatch(self):
        self.apply(Int ** Any, Any, result=error.TypeMismatch)

    def test_complex_subtype_match(self):
        self.apply((Any ** Any) ** Any, Int ** Int, result=Any)

    def test_complex_subtype_mismatch(self):
        self.apply((Int ** Any) ** Any, Any ** Any, result=error.TypeMismatch)

    def test_variable_subtype_match(self):
        self.apply((var.x ** Any) ** var.x, Int ** Int, result=Int)

    def test_variable_subtype_mismatch(self):
        self.apply((var.x ** Int) ** var.x, Int ** Any, result=error.TypeMismatch)

    def test_simple_constraints_passed(self):
        self.apply(
            (var.x ** var.x, var.x.limit(Int, Str)),
            Int,
            result=Int
        )

    def test_simple_constraints_subtype_passed(self):
        self.apply(
            (var.x ** var.x, var.x.limit(Any)),
            Int,
            result=Int
        )

    def test_simple_constraints_subtype_violated(self):
        self.apply(
            (var.x ** var.x, var.x.limit(Int, Str)),
            Any,
            result=error.ViolatedConstraint
        )

    # fails
    def test_compose_constraint_unification(self):
        self.apply(
            ((var.y ** var.z) ** (var.x ** var.y) ** (var.x ** var.z)),
            (var.x, var.x.limit(Int ** Str, Str ** Int)),
            Str ** Int,
            result=Str ** Str)

    # fails
    def test_compose_constraint_subtype(self):
        self.apply(
            ((var.y ** var.z) ** (var.x ** var.y) ** (var.x ** var.z)),
            (var.x, var.x.limit(Int ** Str, Str ** Int)),
            Str ** Any,
            result=Int ** Any)

    def test_compose_something(self):
        self.apply(
            (var.x ** var.x, (var.x ** var.x).limit(Any ** var.z)),
            Int,
            result=Int)

    def test_recursive_constraint(self):
        self.assertRaises(error.RecursiveType, var.x.limit, var.x ** var.x)

    def test_constraint_bound_on_subject_side(self):
        self.apply(
            (var.x ** var.y, var.x.limit(var.y)),
            Int,
            result=Int)

    def test_constraint_bound_on_typeclass_side(self):
        self.apply(
            (var.x ** var.y, var.y.limit(var.x)),
            Int,
            result=Int)

    def test_constraint_bound_on_subject_side_multiple(self):
        self.apply(
            (var.x ** var.y, var.x.limit(var.y, Int)),
            Str,
            result=Str)

    # fails
    def test_constraint_bound_on_typeclass_side_multiple(self):
        self.apply(
            (var.x ** var.y, var.y.limit(var.x, Int)),
            Int,
            result=Int)

    def test_overloaded_function(self):
        self.apply(
            (Str ** Int, Int ** Str),
            Int,
            result=Str)

    # If you have a function that takes Any ** Any but you get an Int ** Int,
    # that is fine. But what if it affects a binding somewhere down the stream?


if __name__ == '__main__':
    unittest.main()
