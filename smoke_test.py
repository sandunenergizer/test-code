"""Smoke test for :mod:`datatools`.

A smoke test answers one question quickly: "does the module load and do
all public functions still work on ordinary input?" It runs in well
under a second, needs no test framework, and exits with status 0 on
success or 1 on the first failure, so it can gate a deployment script.

Run with ``python smoke_test.py``.
"""

import math
import sys

import datatools


def check(name, condition):
    """Print one result line and abort the run if *condition* is false.

    :param str name: Short description of the check.
    :param bool condition: Whether the check passed.
    """
    print(("PASS  " if condition else "FAIL  ") + name)
    if not condition:
        sys.exit(1)


def main():
    """Exercise every public function once and exit non-zero on failure."""
    check("module exposes __all__ and a version",
          bool(datatools.__all__) and bool(datatools.__version__))
    check("every public name is callable",
          all(callable(getattr(datatools, n)) for n in datatools.__all__))

    check("normalize_text composes accents",
          datatools.normalize_text("e\u0301") == "\u00e9")
    check("count_words counts Latin and Chinese",
          datatools.count_words("hello \u4f60\u597d world") == 3)
    check("count_characters counts code points",
          datatools.count_characters("a\U0001F600") == 2)
    check("truncate shortens long text",
          datatools.truncate("Hello, world", 8) == "Hello...")
    check("slugify builds a slug",
          datatools.slugify("Caf\u00e9 au Lait!") == "cafe-au-lait")
    check("is_palindrome accepts a palindrome",
          datatools.is_palindrome("Was it a car or a cat I saw?"))
    check("levenshtein computes edit distance",
          datatools.levenshtein("kitten", "sitting") == 3)
    check("run-length coding round-trips",
          datatools.run_length_decode(
              datatools.run_length_encode("aaabcc")) == "aaabcc")

    check("safe_divide handles zero with default",
          datatools.safe_divide(1, 0, default=0.0) == 0.0)
    check("clamp limits a value", datatools.clamp(15, 0, 10) == 10)
    check("mean averages numbers", datatools.mean([1, 2, 3]) == 2.0)
    check("median finds the middle", datatools.median([3, 1, 2]) == 2)
    check("variance of a constant is zero",
          datatools.variance([4, 4, 4]) == 0.0)
    check("describe summarizes data",
          datatools.describe([1, 2, 3])["count"] == 3)
    check("format_bytes uses binary prefixes",
          datatools.format_bytes(1536) == "1.5 KiB")
    check("parse_number reads Arabic-Indic digits",
          datatools.parse_number("\u0664\u0662") == 42)
    check("parse_number reads infinity",
          datatools.parse_number("-inf") == -math.inf)

    print("All smoke checks passed.")


if __name__ == "__main__":
    main()