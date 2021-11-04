import unittest

from transformation_algebra.type import TypeOperator, SubtypeMismatch
from transformation_algebra.expr import Operator, Expr
from transformation_algebra.lang import Language


class TestAlgebra(unittest.TestCase):
    def test_string_schematic_type(self):
        """
        Test that schematic types are printed with the names of their schematic
        variables.
        """
        A = TypeOperator()
        f = Operator(type=lambda x: x | x << A)
        lang = Language(scope=locals())

        self.assertEqual(str(f.type), "x | x << [A]")

    def test_parse_inline_typing(self):
        A = TypeOperator()
        x = Operator(type=A)
        f = Operator(type=A ** A)
        lang = Language(scope=locals())

        lang.parse("f x : A")

    def test_parse_sources(self):
        A = TypeOperator()
        B = TypeOperator()
        x = Operator(type=A)
        f = Operator(type=lambda x: x ** x)
        lang = Language(scope=locals())

        lang.parse("f 1 : A")
        self.assertRaises(SubtypeMismatch, lang.parse, "1 : B; f 1 : A")

    def test_parse_anonymous_source(self):
        A = TypeOperator()
        lang = Language(scope=locals())
        self.assertTrue(
            lang.parse("~A").match(~A)
        )


if __name__ == '__main__':
    unittest.main()
