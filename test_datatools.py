"""Unit tests for :mod:`datatools`.

Besides ordinary cases, the tests deliberately cover input that
generated tests often miss:

* non-Latin scripts (Chinese, Arabic, Hindi, Thai) and right-to-left text,
* emoji, combining marks, zero-width characters, NUL and lone surrogates,
* very long strings (up to one million code points),
* ``+inf``, ``-inf``, ``nan``, negative zero, subnormals and float limits,
* integers too large for a float,
* ``bool``, ``None`` and ``bytes`` passed where text or numbers belong,
* seeded random (property-style) checks.

Run with ``python -m unittest -v``.
"""

import doctest
import math
import random
import unittest

import datatools
from datatools import (
    clamp, count_characters, count_words, describe, format_bytes,
    is_palindrome, levenshtein, mean, median, normalize_text, parse_number,
    run_length_decode, run_length_encode, safe_divide, slugify, truncate,
    variance,
)

INF = math.inf
NAN = math.nan

CHINESE = "\u4f60\u597d\uff0c\u4e16\u754c"                # ni hao, shi jie
ARABIC = "\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627"
HINDI = ("\u0928\u092e\u0938\u094d\u0924\u0947 "
         "\u0926\u0941\u0928\u093f\u092f\u093e")
THAI = ("\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35"
        "\u0e0a\u0e32\u0e27\u0e42\u0e25\u0e01")
EMOJI = "\U0001F600\U0001F44D\U0001F3FD"   # grin, thumbs-up + skin tone
COMBINING = "e\u0301"                                     # e + acute
PRECOMPOSED = "\u00e9"
ZERO_WIDTH = "a\u200bb"
LONE_SURROGATE = "\ud800"
NOT_TEXT = (None, 42, 3.14, b"bytes", ["a"], True)
NOT_NUMBER = (None, "1", b"1", [1], True, 1 + 2j)


def load_tests(loader, tests, ignore):
    """Add the doctests embedded in the module docstrings to the suite."""
    tests.addTests(doctest.DocTestSuite(datatools))
    return tests


class TestNormalizeText(unittest.TestCase):
    """Tests for :func:`normalize_text`."""

    def test_nfc_composes_combining_sequence(self):
        self.assertEqual(normalize_text(COMBINING), PRECOMPOSED)

    def test_nfd_decomposes(self):
        self.assertEqual(normalize_text(PRECOMPOSED, form="NFD"), COMBINING)

    def test_nfkc_folds_full_width_characters(self):
        self.assertEqual(
            normalize_text("\uff21\uff22\uff23\uff11", form="NFKC"), "ABC1")
        self.assertEqual(normalize_text("\uff21", form="NFC"), "\uff21")

    def test_collapses_unicode_whitespace(self):
        text = "  a \t\n b\u00a0\u00a0c\u3000d  "
        self.assertEqual(normalize_text(text), "a b c d")

    def test_whitespace_kept_when_requested(self):
        self.assertEqual(normalize_text(" a  b ", collapse_whitespace=False),
                         " a  b ")

    def test_scripts_survive_unchanged(self):
        for text in (CHINESE, ARABIC, HINDI, THAI, EMOJI):
            with self.subTest(text=text):
                self.assertEqual(normalize_text(text, form="NFC"),
                                 normalize_text(normalize_text(text)))

    def test_empty_and_whitespace_only(self):
        self.assertEqual(normalize_text(""), "")
        self.assertEqual(normalize_text(" \t\n\u3000"), "")

    def test_nul_and_lone_surrogate_do_not_crash(self):
        self.assertEqual(normalize_text("a\x00b"), "a\x00b")
        self.assertEqual(normalize_text(LONE_SURROGATE), LONE_SURROGATE)

    def test_zero_width_space_is_not_whitespace(self):
        self.assertEqual(normalize_text(ZERO_WIDTH), ZERO_WIDTH)

    def test_very_long_input(self):
        result = normalize_text("a " * 200000)
        self.assertEqual(len(result), 2 * 200000 - 1)

    def test_idempotent_on_random_text(self):
        rng = random.Random(1)
        alphabet = "ab \u00e9\u0301\u4f60\u0645\u3000\t"
        for _ in range(200):
            text = "".join(rng.choice(alphabet) for _ in range(30))
            once = normalize_text(text)
            self.assertEqual(normalize_text(once), once)

    def test_invalid_form(self):
        for form in ("nfc", "", "NFX", None):
            with self.subTest(form=form):
                with self.assertRaises(ValueError):
                    normalize_text("a", form=form)

    def test_wrong_type(self):
        for value in NOT_TEXT:
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    normalize_text(value)


class TestCounting(unittest.TestCase):
    """Tests for :func:`count_words` and :func:`count_characters`."""

    def test_basic_words(self):
        self.assertEqual(count_words("one two  three"), 3)
        self.assertEqual(count_words(""), 0)
        self.assertEqual(count_words("   \n\t"), 0)

    def test_apostrophe_and_underscore(self):
        self.assertEqual(count_words("don't"), 2)
        self.assertEqual(count_words("snake_case"), 1)

    def test_chinese_with_and_without_spaces(self):
        self.assertEqual(count_words("\u4f60\u597d \u4e16\u754c"), 2)
        self.assertEqual(count_words("\u4f60\u597d\u4e16\u754c"), 1)

    def test_arabic_hindi_thai(self):
        self.assertEqual(count_words(ARABIC), 2)
        self.assertEqual(count_words(HINDI), 2)
        self.assertEqual(count_words(THAI), 1)

    def test_combining_marks_do_not_split_words(self):
        self.assertEqual(count_words("\u0928\u092e\u0938\u094d\u0924\u0947"),
                         1)
        self.assertEqual(count_words(COMBINING), 1)
        self.assertEqual(count_words("cafe\u0301 au lait"), 3)

    def test_chinese_punctuation_separates(self):
        self.assertEqual(count_words(CHINESE), 2)

    def test_non_ascii_digits_are_word_characters(self):
        self.assertEqual(count_words("123 \u0663\u0664\u0665"), 2)

    def test_emoji_and_punctuation_are_separators(self):
        self.assertEqual(count_words("\U0001F600 \U0001F600 !?"), 0)
        self.assertEqual(count_words("a\U0001F600b"), 2)

    def test_very_long_input(self):
        self.assertEqual(count_words("word " * 100000), 100000)

    def test_character_counts_code_points(self):
        self.assertEqual(count_characters(CHINESE), 5)
        self.assertEqual(count_characters(COMBINING), 2)
        self.assertEqual(count_characters(PRECOMPOSED), 1)
        self.assertEqual(count_characters(EMOJI), 3)

    def test_character_whitespace_option(self):
        self.assertEqual(count_characters("a b\u3000c\n"), 6)
        self.assertEqual(count_characters("a b\u3000c\n",
                                          ignore_whitespace=True), 3)

    def test_character_very_long_input(self):
        self.assertEqual(count_characters("x" * 1000000), 1000000)

    def test_wrong_type(self):
        for value in NOT_TEXT:
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    count_words(value)
                with self.assertRaises(TypeError):
                    count_characters(value)


class TestTruncate(unittest.TestCase):
    """Tests for :func:`truncate`."""

    def test_short_text_unchanged(self):
        self.assertEqual(truncate("short", 10), "short")
        self.assertEqual(truncate("exact", 5), "exact")

    def test_cut_with_default_suffix(self):
        self.assertEqual(truncate("Hello, world", 8), "Hello...")

    def test_limit_not_larger_than_suffix(self):
        self.assertEqual(truncate("abcdef", 3), "...")
        self.assertEqual(truncate("abcdef", 2), "..")
        self.assertEqual(truncate("abcdef", 0), "")

    def test_empty_suffix(self):
        self.assertEqual(truncate("abcdef", 3, suffix=""), "abc")

    def test_chinese_and_arabic(self):
        self.assertEqual(truncate("\u4f60\u597d\u4e16\u754c\u4f60\u597d", 4,
                                  suffix="\u2026"),
                         "\u4f60\u597d\u4e16\u2026")
        self.assertEqual(len(truncate(ARABIC, 5)), 5)

    def test_emoji_suffix(self):
        self.assertEqual(truncate("abcdef", 4, suffix="\U0001F600"),
                         "abc\U0001F600")

    def test_huge_and_very_long_values(self):
        self.assertEqual(truncate("short", 10 ** 100), "short")
        self.assertEqual(len(truncate("x" * 1000000, 500000)), 500000)

    def test_result_never_exceeds_limit(self):
        rng = random.Random(2)
        for _ in range(300):
            text = "".join(rng.choice("ab\u4f60") for _ in range(40))
            limit = rng.randrange(0, 50)
            with self.subTest(text=text, limit=limit):
                self.assertLessEqual(len(truncate(text, limit)), limit)

    def test_negative_limit(self):
        with self.assertRaises(ValueError):
            truncate("abc", -1)

    def test_wrong_types(self):
        for limit in (3.0, "3", None, True, INF):
            with self.subTest(limit=limit):
                with self.assertRaises(TypeError):
                    truncate("abc", limit)
        with self.assertRaises(TypeError):
            truncate(None, 3)
        with self.assertRaises(TypeError):
            truncate("abc", 3, suffix=None)


class TestSlugify(unittest.TestCase):
    """Tests for :func:`slugify`."""

    def test_basic(self):
        self.assertEqual(slugify("Hello, World!"), "hello-world")

    def test_separators_collapse_and_strip(self):
        self.assertEqual(slugify("  a   b--c_d  "), "a-b-c-d")
        self.assertEqual(slugify("---"), "")

    def test_accents_removed_in_ascii_mode(self):
        self.assertEqual(slugify("Caf\u00e9 M\u00fcnster"), "cafe-munster")
        self.assertEqual(slugify(COMBINING + "a"), "ea")

    def test_non_latin_vanishes_in_ascii_mode(self):
        for text in (CHINESE, ARABIC, HINDI, THAI):
            with self.subTest(text=text):
                self.assertEqual(slugify(text), "")

    def test_chinese_kept_in_unicode_mode(self):
        self.assertEqual(slugify("\u4f60\u597d \u4e16\u754c",
                                 allow_unicode=True),
                         "\u4f60\u597d-\u4e16\u754c")

    def test_arabic_kept_with_diacritics(self):
        word = "\u0645\u064e\u0631\u0652\u062d\u064e\u0628\u064b\u0627"
        self.assertEqual(slugify(word, allow_unicode=True), word)

    def test_hindi_vowel_signs_survive(self):
        self.assertEqual(slugify(HINDI, allow_unicode=True),
                         HINDI.replace(" ", "-"))

    def test_emoji_and_symbols_dropped(self):
        self.assertEqual(slugify("I \u2764 Python \U0001F600",
                                 allow_unicode=True), "i-python")

    def test_full_width_letters(self):
        self.assertEqual(slugify("\uff28\uff25\uff2c\uff2c\uff2f"), "hello")
        self.assertEqual(slugify("\uff28\uff25\uff2c\uff2c\uff2f",
                                 allow_unicode=True), "hello")

    def test_only_symbols_gives_empty_string(self):
        self.assertEqual(slugify("!?@#"), "")
        self.assertEqual(slugify(""), "")

    def test_very_long_input(self):
        self.assertEqual(len(slugify("a" * 1000000)), 1000000)

    def test_output_contains_only_allowed_characters(self):
        rng = random.Random(3)
        alphabet = "aB !-_\u4f60\u0645\U0001F600\u00e9"
        for _ in range(200):
            text = "".join(rng.choice(alphabet) for _ in range(25))
            result = slugify(text, allow_unicode=True)
            with self.subTest(text=text):
                self.assertNotIn("--", result)
                self.assertFalse(result.startswith("-"))
                self.assertFalse(result.endswith("-"))
                self.assertNotIn(" ", result)

    def test_wrong_type(self):
        for value in NOT_TEXT:
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    slugify(value)


class TestIsPalindrome(unittest.TestCase):
    """Tests for :func:`is_palindrome`."""

    def test_classic(self):
        self.assertTrue(is_palindrome("A man, a plan, a canal: Panama"))
        self.assertFalse(is_palindrome("hello"))

    def test_trivial(self):
        self.assertTrue(is_palindrome(""))
        self.assertTrue(is_palindrome("x"))
        self.assertTrue(is_palindrome("!?"))
        self.assertTrue(is_palindrome("12321"))

    def test_case_folding_and_full_width(self):
        self.assertTrue(is_palindrome("Madam"))
        self.assertTrue(is_palindrome("\uff21\uff42\uff22\uff41"))

    def test_chinese(self):
        self.assertTrue(is_palindrome("\u4e0a\u6d77\u81ea\u6765\u6c34"
                                      "\u6765\u81ea\u6d77\u4e0a"))
        self.assertFalse(is_palindrome(CHINESE))

    def test_arabic_right_to_left(self):
        self.assertTrue(is_palindrome("\u0645\u0644\u0645"))
        self.assertFalse(is_palindrome("\u0645\u0644\u0643"))

    def test_emoji_are_ignored(self):
        self.assertTrue(is_palindrome("a\U0001F600a"))

    def test_very_long_input(self):
        self.assertTrue(is_palindrome("ab" * 50000 + "ba" * 50000))
        self.assertFalse(is_palindrome("a" * 50000 + "b" + "a" * 49999 + "c"))

    def test_wrong_type(self):
        for value in NOT_TEXT:
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    is_palindrome(value)


class TestLevenshtein(unittest.TestCase):
    """Tests for :func:`levenshtein`."""

    def test_known_values(self):
        self.assertEqual(levenshtein("kitten", "sitting"), 3)
        self.assertEqual(levenshtein("flaw", "lawn"), 2)
        self.assertEqual(levenshtein("abc", "abc"), 0)

    def test_empty_strings(self):
        self.assertEqual(levenshtein("", ""), 0)
        self.assertEqual(levenshtein("", "abc"), 3)
        self.assertEqual(levenshtein("abcd", ""), 4)

    def test_chinese_arabic_emoji(self):
        self.assertEqual(levenshtein("\u4f60\u597d\u4e16\u754c",
                                     "\u4f60\u597d\u5730\u7403"), 2)
        self.assertEqual(levenshtein(ARABIC, ARABIC + "\u0627"), 1)
        self.assertEqual(levenshtein("\U0001F600", "\U0001F601"), 1)

    def test_canonically_equal_strings_differ_unless_normalized(self):
        self.assertEqual(levenshtein(PRECOMPOSED, COMBINING), 2)
        self.assertEqual(levenshtein(normalize_text(PRECOMPOSED),
                                     normalize_text(COMBINING)), 0)

    def test_very_long_strings(self):
        self.assertEqual(levenshtein("a" * 1500, "b" * 1500), 1500)
        self.assertEqual(levenshtein("a" * 1000000, ""), 1000000)
        self.assertEqual(levenshtein("a" * 2000, "a" * 1999), 1)

    def test_symmetry_and_bounds_on_random_strings(self):
        rng = random.Random(4)
        for _ in range(200):
            a, b = ("".join(rng.choice("abc\u4f60")
                            for _ in range(rng.randrange(15)))
                    for _ in range(2))
            dist = levenshtein(a, b)
            with self.subTest(a=a, b=b):
                self.assertEqual(dist, levenshtein(b, a))
                self.assertLessEqual(dist, max(len(a), len(b)))
                self.assertGreaterEqual(dist, abs(len(a) - len(b)))

    def test_triangle_inequality(self):
        rng = random.Random(5)
        for _ in range(100):
            a, b, c = ("".join(rng.choice("ab") for _ in range(8))
                       for _ in range(3))
            self.assertLessEqual(levenshtein(a, c),
                                 levenshtein(a, b) + levenshtein(b, c))

    def test_wrong_type(self):
        for value in NOT_TEXT:
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    levenshtein(value, "a")
                with self.assertRaises(TypeError):
                    levenshtein("a", value)


class TestRunLength(unittest.TestCase):
    """Tests for :func:`run_length_encode` and :func:`run_length_decode`."""

    def test_encode_known(self):
        self.assertEqual(run_length_encode("aaabcc"),
                         [("a", 3), ("b", 1), ("c", 2)])
        self.assertEqual(run_length_encode(""), [])

    def test_digits_are_unambiguous(self):
        self.assertEqual(run_length_encode("111"), [("1", 3)])
        self.assertEqual(run_length_decode([("1", 3)]), "111")

    def test_scripts_and_emoji(self):
        text = "\u4f60\u4f60\u597d\u0645\u0645\U0001F600\U0001F600"
        pairs = run_length_encode(text)
        self.assertEqual(pairs[0], ("\u4f60", 2))
        self.assertEqual(run_length_decode(pairs), text)

    def test_very_long_run(self):
        self.assertEqual(run_length_encode("x" * 1000000),
                         [("x", 1000000)])
        self.assertEqual(len(run_length_decode([("x", 1000000)])), 1000000)

    def test_round_trip_on_random_text(self):
        rng = random.Random(6)
        alphabet = "aab1\u4f60\u0645\U0001F600\u0301 "
        for _ in range(300):
            text = "".join(rng.choice(alphabet) for _ in range(40))
            with self.subTest(text=text):
                self.assertEqual(run_length_decode(run_length_encode(text)),
                                 text)

    def test_decode_accepts_lists_and_generators(self):
        self.assertEqual(run_length_decode([["a", 2]]), "aa")
        self.assertEqual(run_length_decode(iter([("a", 2), ("b", 1)])), "aab")
        self.assertEqual(run_length_decode([]), "")

    def test_decode_invalid_values(self):
        for pairs in ([("ab", 2)], [("", 2)], [("a", 0)], [("a", -1)]):
            with self.subTest(pairs=pairs):
                with self.assertRaises(ValueError):
                    run_length_decode(pairs)

    def test_decode_invalid_types(self):
        bad = ([("a", 2.0)], [("a", True)], [("a", "2")], [(1, 2)],
               ["ab"], [("a",)], [("a", 1, 2)], [None], [5])
        for pairs in bad:
            with self.subTest(pairs=pairs):
                with self.assertRaises(TypeError):
                    run_length_decode(pairs)

    def test_encode_wrong_type(self):
        for value in NOT_TEXT:
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    run_length_encode(value)


class TestSafeDivide(unittest.TestCase):
    """Tests for :func:`safe_divide`."""

    def test_normal(self):
        self.assertEqual(safe_divide(6, 3), 2.0)
        self.assertEqual(safe_divide(-1, 4), -0.25)

    def test_zero_without_default(self):
        for denominator in (0, 0.0, -0.0):
            with self.subTest(denominator=denominator):
                with self.assertRaises(ZeroDivisionError):
                    safe_divide(1, denominator)

    def test_zero_with_default(self):
        self.assertEqual(safe_divide(1, 0, default=0.0), 0.0)
        self.assertEqual(safe_divide(0, 0, default=-1), -1)
        self.assertTrue(math.isnan(safe_divide(1, -0.0, default=NAN)))

    def test_infinities(self):
        self.assertEqual(safe_divide(INF, 2), INF)
        self.assertEqual(safe_divide(-INF, 2), -INF)
        self.assertEqual(safe_divide(1, INF), 0.0)
        self.assertEqual(safe_divide(1, -INF), -0.0)
        self.assertTrue(math.isnan(safe_divide(INF, INF)))

    def test_nan_propagates(self):
        self.assertTrue(math.isnan(safe_divide(NAN, 2)))
        self.assertTrue(math.isnan(safe_divide(2, NAN)))

    def test_negative_zero_numerator_keeps_sign(self):
        self.assertEqual(math.copysign(1, safe_divide(-0.0, 5)), -1.0)

    def test_huge_integers(self):
        self.assertEqual(safe_divide(10 ** 400, 10 ** 399), 10.0)
        self.assertEqual(safe_divide(10 ** 400, 1), INF)
        self.assertEqual(safe_divide(-10 ** 400, 1), -INF)
        self.assertEqual(safe_divide(10 ** 400, -1), -INF)

    def test_float_overflow_gives_infinity(self):
        self.assertEqual(safe_divide(1e308, 1e-308), INF)

    def test_subnormal(self):
        self.assertEqual(safe_divide(5e-324, 2), 0.0)

    def test_wrong_types(self):
        for value in NOT_NUMBER:
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    safe_divide(value, 1)
                with self.assertRaises(TypeError):
                    safe_divide(1, value)


class TestClamp(unittest.TestCase):
    """Tests for :func:`clamp`."""

    def test_inside_below_above(self):
        self.assertEqual(clamp(5, 0, 10), 5)
        self.assertEqual(clamp(-3, 0, 10), 0)
        self.assertEqual(clamp(15, 0, 10), 10)

    def test_bounds_inclusive_and_equal(self):
        self.assertEqual(clamp(0, 0, 10), 0)
        self.assertEqual(clamp(10, 0, 10), 10)
        self.assertEqual(clamp(99, 4, 4), 4)

    def test_infinite_values_and_bounds(self):
        self.assertEqual(clamp(INF, 0, 10), 10)
        self.assertEqual(clamp(-INF, 0, 10), 0)
        self.assertEqual(clamp(5, -INF, INF), 5)
        self.assertEqual(clamp(INF, -INF, INF), INF)
        self.assertEqual(clamp(-INF, -INF, INF), -INF)

    def test_keeps_type(self):
        self.assertIsInstance(clamp(5, 0, 10), int)
        self.assertIsInstance(clamp(5.5, 0, 10), float)

    def test_huge_integers(self):
        self.assertEqual(clamp(10 ** 400, 0, 10 ** 500), 10 ** 400)

    def test_nan_rejected_everywhere(self):
        for args in ((NAN, 0, 1), (0, NAN, 1), (0, 0, NAN)):
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    clamp(*args)

    def test_low_above_high(self):
        with self.assertRaises(ValueError):
            clamp(1, 10, 0)

    def test_wrong_types(self):
        for value in NOT_NUMBER:
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    clamp(value, 0, 1)
                with self.assertRaises(TypeError):
                    clamp(0, value, 1)
                with self.assertRaises(TypeError):
                    clamp(0, 0, value)

    def test_result_always_within_bounds(self):
        rng = random.Random(7)
        for _ in range(300):
            low = rng.uniform(-100, 100)
            high = low + rng.uniform(0, 100)
            value = rng.choice([rng.uniform(-1e6, 1e6), INF, -INF])
            self.assertTrue(low <= clamp(value, low, high) <= high)


class TestMean(unittest.TestCase):
    """Tests for :func:`mean`."""

    def test_basic(self):
        self.assertEqual(mean([1, 2, 3]), 2.0)
        self.assertEqual(mean([5]), 5.0)
        self.assertEqual(mean([-1, 1]), 0.0)

    def test_accurate_summation(self):
        self.assertEqual(mean([0.1] * 10), 0.1)

    def test_accepts_iterators_and_ranges(self):
        self.assertEqual(mean(iter([2, 4])), 3.0)
        self.assertEqual(mean(range(1000000)), 499999.5)

    def test_infinities(self):
        self.assertEqual(mean([1, INF]), INF)
        self.assertEqual(mean([-INF, 1]), -INF)
        self.assertEqual(mean([INF, INF]), INF)
        self.assertEqual(mean([-INF, -INF]), -INF)

    def test_opposite_infinities_undefined(self):
        with self.assertRaises(ValueError):
            mean([INF, -INF])

    def test_nan_rejected(self):
        with self.assertRaises(ValueError):
            mean([1, NAN])

    def test_overflowing_sum(self):
        self.assertEqual(mean([1e308, 1e308]), 1e308)
        self.assertEqual(mean([-1e308, 1e308]), 0.0)

    def test_subnormals(self):
        self.assertEqual(mean([5e-324] * 3), 5e-324)

    def test_huge_integers(self):
        self.assertEqual(mean([10 ** 400, 10 ** 400]), INF)
        self.assertEqual(mean([-10 ** 400, -10 ** 400]), -INF)
        self.assertEqual(mean([10 ** 400, -10 ** 400]), 0.0)
        self.assertEqual(mean([10 ** 400, 10 ** 400, 10 ** 400]),
                         INF)

    def test_huge_integer_mixed_with_float_is_rejected(self):
        with self.assertRaises(ValueError):
            mean([10 ** 400, 1.5])

    def test_empty(self):
        for empty in ([], (), iter([])):
            with self.subTest(empty=empty):
                with self.assertRaises(ValueError):
                    mean(empty)

    def test_wrong_types(self):
        for value in ("123", b"123"):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    mean(value)
        for item in (None, "1", True, 1j):
            with self.subTest(item=item):
                with self.assertRaises(TypeError):
                    mean([1, item])
        with self.assertRaises(TypeError):
            mean(5)

    def test_between_min_and_max(self):
        rng = random.Random(8)
        for _ in range(200):
            size = rng.randrange(1, 30)
            data = [rng.uniform(-1e3, 1e3) for _ in range(size)]
            self.assertTrue(min(data) - 1e-9 <= mean(data) <= max(data) + 1e-9)


class TestMedian(unittest.TestCase):
    """Tests for :func:`median`."""

    def test_odd_and_even(self):
        self.assertEqual(median([3, 1, 2]), 2)
        self.assertEqual(median([1, 2, 3, 4]), 2.5)
        self.assertEqual(median([7]), 7)

    def test_duplicates_and_negatives(self):
        self.assertEqual(median([2, 2, 2, 2]), 2)
        self.assertEqual(median([-5, -1, -3]), -3)

    def test_infinities(self):
        self.assertEqual(median([INF, 1, 2]), 2)
        self.assertEqual(median([1, 2, INF, INF]), INF)
        self.assertEqual(median([-INF, 5]), -INF)
        self.assertEqual(median([-INF, -INF]), -INF)
        self.assertEqual(median([INF, INF]), INF)

    def test_middle_pair_of_opposite_infinities_undefined(self):
        with self.assertRaises(ValueError):
            median([-INF, INF])

    def test_nan_rejected(self):
        with self.assertRaises(ValueError):
            median([1, NAN, 3])

    def test_overflowing_pair(self):
        self.assertEqual(median([1e308, 1.7e308]), 1.35e308)

    def test_mixed_int_and_float_pair(self):
        self.assertEqual(median([1, 2.5]), 1.75)
        self.assertEqual(median([1.5, 2.5]), 2.0)

    def test_huge_integers(self):
        self.assertEqual(median([10 ** 400]), 10 ** 400)
        self.assertEqual(median([10 ** 400, 10 ** 400 + 2]), INF)
        self.assertEqual(median([-10 ** 400, -10 ** 400 - 2]), -INF)
        self.assertEqual(median([10 ** 400, -10 ** 400]), 0.0)
        with self.assertRaises(ValueError):
            median([1.5, 10 ** 400])

    def test_empty(self):
        with self.assertRaises(ValueError):
            median([])

    def test_wrong_types(self):
        with self.assertRaises(TypeError):
            median("123")
        with self.assertRaises(TypeError):
            median([1, None])
        with self.assertRaises(TypeError):
            median([1, True])

    def test_does_not_modify_input(self):
        data = [3, 1, 2]
        median(data)
        self.assertEqual(data, [3, 1, 2])

    def test_large_input(self):
        data = list(range(1000001))
        random.Random(9).shuffle(data)
        self.assertEqual(median(data), 500000)

    def test_between_min_and_max(self):
        rng = random.Random(10)
        for _ in range(200):
            size = rng.randrange(1, 20)
            data = [rng.randrange(-50, 50) for _ in range(size)]
            self.assertTrue(min(data) <= median(data) <= max(data))


class TestVariance(unittest.TestCase):
    """Tests for :func:`variance`."""

    def test_known_values(self):
        self.assertEqual(variance([1, 2, 3, 4]), 1.25)
        self.assertEqual(variance([5]), 0.0)
        self.assertEqual(variance([2, 2, 2]), 0.0)

    def test_shift_invariance(self):
        base = [1.0, 2.0, 4.0, 8.0]
        shifted = [x + 1e6 for x in base]
        self.assertAlmostEqual(variance(base), variance(shifted), places=6)

    def test_infinite_rejected(self):
        for data in ([1, INF], [-INF, 1], [INF, -INF]):
            with self.subTest(data=data):
                with self.assertRaises(ValueError):
                    variance(data)

    def test_nan_and_empty_rejected(self):
        with self.assertRaises(ValueError):
            variance([1, NAN])
        with self.assertRaises(ValueError):
            variance([])

    def test_overflow_gives_infinity(self):
        self.assertEqual(variance([1e308, -1e308]), INF)

    def test_huge_integers_rejected(self):
        with self.assertRaises(ValueError):
            variance([10 ** 400, 0])
        with self.assertRaises(ValueError):
            describe([10 ** 400])

    def test_never_negative(self):
        rng = random.Random(11)
        for _ in range(200):
            size = rng.randrange(1, 20)
            data = [rng.uniform(-1e3, 1e3) for _ in range(size)]
            self.assertGreaterEqual(variance(data), 0.0)

    def test_wrong_types(self):
        with self.assertRaises(TypeError):
            variance("1234")
        with self.assertRaises(TypeError):
            variance([1, "2"])


class TestDescribe(unittest.TestCase):
    """Tests for :func:`describe`."""

    def test_keys_and_values(self):
        result = describe([1, 2, 3, 4])
        self.assertEqual(sorted(result), ["count", "max", "mean", "median",
                                          "min", "variance"])
        self.assertEqual(result["count"], 4)
        self.assertEqual(result["min"], 1)
        self.assertEqual(result["max"], 4)
        self.assertEqual(result["mean"], 2.5)
        self.assertEqual(result["median"], 2.5)
        self.assertEqual(result["variance"], 1.25)

    def test_infinity_gives_nan_variance(self):
        result = describe([1, 2, INF])
        self.assertEqual(result["max"], INF)
        self.assertEqual(result["mean"], INF)
        self.assertTrue(math.isnan(result["variance"]))

    def test_negative_infinity(self):
        self.assertEqual(describe([-INF, 0])["min"], -INF)

    def test_mixed_infinities_rejected(self):
        with self.assertRaises(ValueError):
            describe([INF, -INF])

    def test_nan_empty_and_types(self):
        with self.assertRaises(ValueError):
            describe([NAN])
        with self.assertRaises(ValueError):
            describe([])
        with self.assertRaises(TypeError):
            describe("abc")

    def test_generator_is_consumed_once(self):
        result = describe(x for x in (1, 2, 3))
        self.assertEqual(result["count"], 3)


class TestFormatBytes(unittest.TestCase):
    """Tests for :func:`format_bytes`."""

    def test_units(self):
        cases = {0: "0 B", 1: "1 B", 1023: "1023 B", 1024: "1.0 KiB",
                 1536: "1.5 KiB", 1048576: "1.0 MiB", 1024 ** 3: "1.0 GiB",
                 1024 ** 4: "1.0 TiB", 1024 ** 5: "1.0 PiB",
                 1024 ** 6: "1.0 EiB", 1024 ** 7: "1.0 ZiB",
                 1024 ** 8: "1.0 YiB"}
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(format_bytes(value), expected)

    def test_rounding_carries_to_next_unit(self):
        self.assertEqual(format_bytes(1048575), "1.0 MiB")

    def test_precision(self):
        self.assertEqual(format_bytes(1536, precision=0), "2 KiB")
        self.assertEqual(format_bytes(1536, precision=3), "1.500 KiB")

    def test_floats(self):
        self.assertEqual(format_bytes(2048.0), "2.0 KiB")
        self.assertEqual(format_bytes(0.0), "0 B")
        self.assertEqual(format_bytes(-0.0), "0 B")

    def test_beyond_largest_unit(self):
        self.assertTrue(format_bytes(10 ** 30).endswith(" YiB"))
        self.assertTrue(format_bytes(1e300).endswith(" YiB"))
        self.assertTrue(format_bytes(1.7e308).endswith(" YiB"))

    def test_too_large_for_float(self):
        with self.assertRaises(ValueError):
            format_bytes(10 ** 400)

    def test_non_finite_and_negative(self):
        for value in (INF, -INF, NAN, -1, -0.5):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    format_bytes(value)

    def test_precision_range(self):
        for precision in (-1, 11, 10 ** 9):
            with self.subTest(precision=precision):
                with self.assertRaises(ValueError):
                    format_bytes(1024, precision=precision)

    def test_wrong_types(self):
        for value in NOT_NUMBER:
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    format_bytes(value)
        for precision in (1.0, "1", None, True):
            with self.subTest(precision=precision):
                with self.assertRaises(TypeError):
                    format_bytes(1024, precision=precision)


class TestParseNumber(unittest.TestCase):
    """Tests for :func:`parse_number`."""

    def test_integers_and_floats(self):
        self.assertEqual(parse_number("42"), 42)
        self.assertIsInstance(parse_number("42"), int)
        self.assertEqual(parse_number("-3.5"), -3.5)
        self.assertEqual(parse_number("+7"), 7)
        self.assertEqual(parse_number("1e3"), 1000.0)
        self.assertEqual(parse_number(".5"), 0.5)

    def test_whitespace_and_separators(self):
        self.assertEqual(parse_number("  7 \n"), 7)
        self.assertEqual(parse_number("1,234,567"), 1234567)
        self.assertEqual(parse_number("-1,234.5"), -1234.5)
        self.assertEqual(parse_number("\uff11\uff0c\uff12\uff13\uff14"), 1234)
        self.assertEqual(parse_number("\u3000 12 \u00a0"), 12)

    def test_infinity_and_nan_words(self):
        self.assertEqual(parse_number("inf"), INF)
        self.assertEqual(parse_number("-inf"), -INF)
        self.assertEqual(parse_number("Infinity"), INF)
        self.assertTrue(math.isnan(parse_number("nan")))

    def test_exponent_overflow_gives_infinity(self):
        self.assertEqual(parse_number("1e400"), INF)
        self.assertEqual(parse_number("-1e400"), -INF)

    def test_arabic_indic_and_persian_digits(self):
        self.assertEqual(parse_number("\u0664\u0662"), 42)
        self.assertEqual(parse_number("\u06f4\u06f2"), 42)
        self.assertEqual(parse_number("-\u0661\u0662\u0663"), -123)

    def test_devanagari_and_thai_digits(self):
        self.assertEqual(parse_number("\u0967\u0968"), 12)
        self.assertEqual(parse_number("\u0e51\u0e52"), 12)

    def test_full_width_digits_and_unicode_minus(self):
        self.assertEqual(parse_number("\uff11\uff12\uff13"), 123)
        self.assertEqual(parse_number("\u22125"), -5)

    def test_invalid_text(self):
        invalid = ["", "   ", "abc", "1_000", "1 000", "--1", "0x10",
                   "\u4e09", "\u0661\u0662\u066b\u0665", "1.2.3", "e5",
                   "1,", ",1", "1,5", "12,34", "1,,234", "1,2345",
                   "\u0000", "\U0001F600", "12abc"]
        for text in invalid:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    parse_number(text)

    def test_very_long_digit_string_does_not_crash(self):
        result = parse_number("9" * 5000)
        self.assertTrue(isinstance(result, int) or result == INF)

    def test_very_long_invalid_string(self):
        with self.assertRaises(ValueError):
            parse_number("1" * 100000 + "x")

    def test_negative_zero(self):
        self.assertEqual(math.copysign(1, parse_number("-0.0")), -1.0)
        self.assertEqual(parse_number("-0"), 0)

    def test_round_trip_with_integers(self):
        rng = random.Random(12)
        for _ in range(200):
            value = rng.randrange(-10 ** 12, 10 ** 12)
            self.assertEqual(parse_number(f"{value:,}"), value)

    def test_wrong_type(self):
        for value in NOT_TEXT:
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    parse_number(value)


class TestPublicApi(unittest.TestCase):
    """Checks that the public interface is complete and documented."""

    def test_all_names_exist_and_are_documented(self):
        self.assertGreaterEqual(len(datatools.__all__), 4)
        for name in datatools.__all__:
            with self.subTest(name=name):
                func = getattr(datatools, name)
                self.assertTrue(callable(func))
                self.assertTrue(func.__doc__)
                self.assertIn(":returns:", func.__doc__)


if __name__ == "__main__":
    unittest.main()