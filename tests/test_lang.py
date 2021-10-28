import unittest

from transformation_algebra import error
from transformation_algebra.type import Type
from transformation_algebra.expr import Operator
from transformation_algebra.lang import Language


class TestAlgebra(unittest.TestCase):
    def test_string_schematic_type(self):
        """
        Test that schematic types are printed with the names of their schematic
        variables.
        """
        A = Type.declare()
        f = Operator(type=lambda x: x | x @ A)
        lang = Language(scope=locals())

        self.assertEqual(str(lang.f), "f : x | x @ [A]")

    def test_parse_inline_typing(self):
        A = Type.declare()
        x = Operator(type=A)
        f = Operator(type=A ** A)
        lang = Language(scope=locals())

        lang.parse("f x : A")

    def test_parse_sources(self):
        A = Type.declare()
        B = Type.declare()
        x = Operator(type=A)
        f = Operator(type=lambda x: x ** x)
        lang = Language(scope=locals())

        lang.parse("f 1 : A")
        self.assertRaises(error.SubtypeMismatch, lang.parse, "1 : B; f 1 : A")


if __name__ == '__main__':
    unittest.main()
