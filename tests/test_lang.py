import unittest

from transformation_algebra.type import TypeOperator, SubtypeMismatch, _
from transformation_algebra.expr import Operator, Expr
from transformation_algebra.lang import Language


class TestAlgebra(unittest.TestCase):

    def test_taxonomy(self):
        A = TypeOperator()
        B = TypeOperator(supertype=A)
        F = TypeOperator(params=2)
        FAA = F(A, A)
        lang = Language(scope=locals())

        # Note that types are subtypes of just one parent, so even though
        # F(B, B) should reasonably be child of both F(A, B) and F(B, A), we
        # just note F(A, A) for now.
        actual = lang.taxonomy()
        expected = {
            A(): None,
            B(): A(),
            F(A, A): None,
            F(A, B): F(A, A),
            F(B, A): F(A, A),
            F(B, B): F(A, A)
        }

        for sub, sup in actual.items():
            self.assertIn(sub, expected)
            self.assertEqual(actual[sub], expected[sub])

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
        FA = F(A)
        f = Operator(type=lambda x: x ** F(x))
        lang = Language(scope=locals())

        lang.parse("f(1 : A) : FA")

    def test_type_synonyms_no_variables(self):
        A, F = TypeOperator(), TypeOperator(params=1)
        FA, FV = F(A), F(_)
        lang = Language(scope=locals())
        self.assertIn(FA, lang.synonyms.values())
        self.assertNotIn(FV, lang.synonyms.values())

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
