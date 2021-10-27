import unittest

from transformation_algebra.type import Type
from transformation_algebra.expr import Operator
from transformation_algebra.alg import TransformationAlgebra


class TestAlgebra(unittest.TestCase):
    def test_string_schematic_type(self):
        """
        Test that schematic types are printed with the names of their schematic
        variables.
        """
        lang = TransformationAlgebra()
        lang.A = Type.declare()
        lang.f = Operator(type=lambda x: x | x @ lang.A)
        self.assertEqual(str(lang.f), "f : x | x @ [A]")


if __name__ == '__main__':
    unittest.main()
