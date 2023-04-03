import unittest

from transforge.type import TypeOperator, TypeAlias, \
    _, Top, TypeVariable, TypeParameterError, UnexpectedVariableError
from transforge.expr import Operator, Source
from transforge.lang import Language, TypeAnnotationError, \
    UndefinedTokenError, MissingInputError
from collections import defaultdict


class TestAlgebra(unittest.TestCase):

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

        lang.parse("f(1 : A) : FA", Source())

    def test_concretizable_parsed_wildcard(self):
        # Wildcards sometimes need to be concretized to catch-all `Top` types,
        # specifically in queries. Sometimes, they should just be left as
        # variables, specifically in full expressions.
        # We could do this automatically (make `parse_type` do the
        # concretization while `parse_expr` wouldn't), but I think being
        # explicit probably makes for fewer surprises
        # cf <https://github.com/quangis/transformation-algebra/issues/108>
        lang = Language()
        self.assertEqual(type(lang.parse_type("_")), TypeVariable)
        self.assertEqual(lang.parse_type("_").concretize(replace=True), Top())

    def test_type_synonyms_no_variables(self):
        F = TypeOperator(params=1)
        self.assertRaises(UnexpectedVariableError, TypeAlias, F(_))

    def test_parse_sources(self):
        A = TypeOperator()
        B = TypeOperator()
        x = Operator(type=A)
        f = Operator(type=lambda x: x ** x)
        lang = Language(scope=locals())

        lang.parse("f 1 : A", Source())
        self.assertRaises(TypeAnnotationError, lang.parse, "1 : B; f 1 : A",
            Source())

    def test_parse_anonymous_source(self):
        A = TypeOperator()
        F = TypeOperator(params=1)
        f = Operator(type=A ** F(A) ** A)
        lang = Language(scope=locals())
        expr = lang.parse("- : A")
        expr.fix()
        self.assertTrue(expr.match(Source(A)))
        expr2 = lang.parse("f (- : A) (- : F(A))")
        expr2.fix()
        self.assertTrue(expr2.match(f(Source(A), Source(F(A)))))

    def test_parse_tuple(self):
        A = TypeOperator()
        F = TypeOperator(params=2)
        lang = Language(scope=locals())
        self.assertTrue(lang.parse_type("A * A").match(A * A))
        self.assertTrue(lang.parse_type("(A * A) * A").match((A * A) * A))
        self.assertTrue(lang.parse_type("A * (A * A)").match(A * (A * A)))
        self.assertTrue(lang.parse_type("A * A * A").match(A * (A * A)))
        # for now, associativity doesn't match but won't matter later because
        # it's a non-associative operator anyway

        self.assertTrue(lang.parse_type("F(A * A, A)").match(F(A * A, A)))
        self.assertTrue(lang.parse_type("F(A, A * A)").match(F(A, A * A)))
        self.assertTrue(lang.parse_type("F((A * A), (A))").match(F(A * A, A)))
        self.assertTrue(lang.parse_type("F((A), (A * A))").match(F(A, A * A)))

        self.assertTrue(lang.parse("- : (A * A)").match(Source(A * A)))
        self.assertRaises(UndefinedTokenError, lang.parse, "- : A * A")

    def test_parameterized_type_alias(self):
        # See issue #73
        A = TypeOperator()
        B = TypeOperator(supertype=A)
        F = TypeOperator(params=2)
        G = TypeAlias(lambda x: F(x, B), A)
        lang = Language(scope=locals())
        self.assertTrue(
            lang.parse("- : G(B)").match(Source(F(B, B)))
        )
        self.assertRaises(TypeParameterError, lang.parse, "- : G")

    def test_canon(self):
        A = TypeOperator()
        B = TypeOperator(supertype=A)
        F = TypeOperator(params=2)
        G = TypeAlias(F(A, B))
        lang = Language(scope=locals(), canon={Top, A, G})
        self.assertEqual(lang.canon, {Top(), A(), B(), F(A, B), F(B, B),
            F(A, Top), F(B, Top), F(Top, B), F(Top, Top)})

    def test_canon_parameterized_operator(self):
        A = TypeOperator()
        F = TypeOperator(params=2)
        self.assertRaises(ValueError, Language, scope=locals(), canon={F})

    def test_canon_parameterized_alias(self):
        F = TypeAlias(lambda x: x)
        self.assertRaises(ValueError, Language, scope=locals(), canon={F})

    def test_subtypes_and_supertypes(self):
        # Make sure that direct subtypes and supertypes are available for
        # canonical nodes, and mirroring eachother

        A = TypeOperator()
        B = TypeOperator(supertype=A)
        F = TypeOperator(params=2)
        lang = Language(scope=locals(), canon={Top, A, F(A, B)})

        tree = {
            Top: [A, F(Top, Top)],
            A: [B],
            F(Top, Top): [F(Top, B), F(A, Top)],
            F(Top, B): [F(A, B)],
            F(A, Top): [F(A, B), F(B, Top)],
            F(B, Top): [F(B, B)],
            F(A, B): [F(B, B)],
        }

        subtypes = defaultdict(set)
        supertypes = defaultdict(set)
        for parent, children in tree.items():
            parent = parent.instance()
            for child in children:
                child = child.instance()
                subtypes[parent].add(child)
                supertypes[child].add(parent)

        for s, expected in subtypes.items():
            with self.subTest(subtype_of=s):
                self.assertEqual(set(lang.subtypes(s)), expected)

        for s, expected in supertypes.items():
            with self.subTest(supertype_of=s):
                self.assertEqual(set(lang.supertypes(s)), expected)

    def test_wildcard_parsing_does_not_cause_subtyping_errors(self):
        # Test that a parsed `_` does not cause subtype errors once it is
        # applied to a function.
        # cf. <https://github.com/quangis/transformation-algebra/issues/108>
        # The test uses a compound type because the issue does not arise with a
        # simple type. My guess is that is due to interactions with the `:`
        # operator that causes a subtype unification.
        A = TypeOperator()
        F = TypeOperator(params=1)
        f = Operator(type=F(A) ** A)
        lang = Language(scope=locals())
        lang.parse("f (-: F(_))")

    def test_missing_input_default_sources(self):
        # Test that parsing can be done either with explicit arguments given or 
        # with default sources substituting for inputs
        A = TypeOperator()
        f = Operator(type=A ** A)
        lang = Language(scope=locals())
        self.assertRaises(MissingInputError, lang.parse, "f 1")
        self.assertTrue(
            lang.parse("f 1", defaults=True).match(f(Source()), strict=False)
        )


if __name__ == '__main__':
    unittest.main()
