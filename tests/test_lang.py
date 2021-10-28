import unittest

from transformation_algebra.type import Type
from transformation_algebra.expr import Operator
from transformation_algebra.lang import Language


class TestAlgebra(unittest.TestCase):
    def test_string_schematic_type(self):
        """
        Test that schematic types are printed with the names of their schematic
        variables.
        """
        lang = Language()
        A = Type.declare()
        f = Operator(type=lambda x: x | x @ A)
        lang.add_scope(locals())

        self.assertEqual(str(lang.f), "f : x | x @ [A]")

    def test_parse_inline_typing(self):
        lang = Language()
        A = Type.declare()
        x = Operator(type=A)
        f = Operator(type=A ** A)
        lang.add_scope(locals())

        lang.parse("f x : A")


if __name__ == '__main__':
    unittest.main()
