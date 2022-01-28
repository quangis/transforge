import unittest

from transformation_algebra.type import TypeOperator, TypeAlias, \
    SubtypeMismatch, _
from transformation_algebra.expr import Operator, Expr
from transformation_algebra.lang import Language


class TestAlgebra(unittest.TestCase):

    def test_taxonomy(self):
        A = TypeOperator()
        B = TypeOperator(supertype=A)
        F = TypeOperator(params=2)
        AA = TypeAlias(F(A, A))
        lang = Language(scope=locals())

        actual = lang.taxonomy
        expected = {
            A(): {B()},
            B(): set(),
            F(A, A): {F(A, B), F(B, A)},
            F(A, B): {F(B, B)},
            F(B, A): {F(B, B)},
            F(B, B): set()
        }

        self.assertEqual(expected, actual)

    def test_complex_taxonomy(self):
        A = TypeOperator()
        B = TypeOperator(supertype=A)
        C = TypeOperator(supertype=A)
        F = TypeOperator(params=2)
        AA = TypeAlias(F(A, A))
        lang = Language(scope=locals())

        actual = lang.taxonomy
        expected = {
            A(): {B(), C()},
            B(): set(),
            C(): set(),
            F(A, A): {F(A, B), F(A, C), F(B, A), F(C, A)},
            F(A, B): {F(B, B), F(C, B)},
            F(A, C): {F(B, C), F(C, C)},
            F(B, A): {F(B, B), F(B, C)},
            F(C, A): {F(C, B), F(C, C)},
            F(B, B): set(),
            F(B, C): set(),
            F(C, B): set(),
            F(C, C): set(),
        }

        self.assertEqual(expected, actual)

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

    def test_type_synonyms(self):
        A = TypeOperator()
        F = TypeOperator(params=1)
        FA = TypeAlias(F(A))
        f = Operator(type=lambda x: x ** F(x))
        lang = Language(scope=locals())

        lang.parse("f(1 : A) : FA")

    def test_type_synonyms_no_variables(self):
        F = TypeOperator(params=1)
        self.assertRaises(RuntimeError, TypeAlias, F(_))

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
        F = TypeOperator(params=1)
        f = Operator(type=A ** F(A) ** A)
        lang = Language(scope=locals())
        self.assertTrue(
            lang.parse("~A").match(~A)
        )
        self.assertTrue(
            lang.parse("f ~A ~F(A)").match(f(~A, ~F(A)))
        )


if __name__ == '__main__':
    unittest.main()
