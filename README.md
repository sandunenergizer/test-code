# datatools

Small, dependency-free Python helpers for **Unicode text** and **IEEE 754
numbers** (infinity, NaN, overflow), with a unit test suite, a smoke test
and reStructuredText documentation.

* `datatools.py` – the module under test (about 700 lines, 17 public functions)
* `test_datatools.py` – 174 unit tests, including the module's doctests
* `smoke_test.py` – a fast end-to-end check of every public function
* Requires Python 3.7+ and nothing else

## Run

```
python -m unittest -v        # unit tests + doctests
python smoke_test.py         # smoke test, exit code 0 = success
python -m pydoc datatools    # read the API documentation
```

## Functions

| Group  | Functions |
|--------|-----------|
| Text   | `normalize_text`, `count_words`, `count_characters`, `truncate`, `slugify`, `is_palindrome`, `levenshtein`, `run_length_encode`, `run_length_decode` |
| Number | `safe_divide`, `clamp`, `mean`, `median`, `variance`, `describe`, `format_bytes`, `parse_number` |

Rules shared by all functions: a wrong *type* raises `TypeError`, an unusable
*value* raises `ValueError`, `bool` is never accepted as a number, and strings
are handled as Unicode code points.

## Standards followed

* **PEP 8** – 79-character lines, no trailing whitespace, naming conventions
* **PEP 257 / PEP 287** – every module, function and test class has a docstring
  written in reStructuredText (`:param:`, `:returns:`, `:raises:`,
  `.. code-block::`), so Sphinx can build HTML documentation from it
* **PEP 484 style** – `from __future__ import annotations` is used; the API is
  described in the docstrings
* Doctest examples in the docstrings are executed as part of the test suite,
  so the documentation cannot drift from the code

## Test strategy: cases that are usually missed

| Area | Examples tested |
|------|-----------------|
| Scripts | Chinese, Arabic (right-to-left, with diacritics), Hindi and Thai (combining vowel signs), full-width letters and digits, Arabic-Indic, Persian, Devanagari and Thai digits |
| Odd code points | emoji with skin-tone modifiers, combining accents (`e` + U+0301 vs U+00E9), zero-width space, ideographic space U+3000, no-break space, NUL, a lone surrogate |
| Size | strings of 1,000,000 code points, 200,000 collapsed whitespace runs, a 1,000,000-element mean and median, 5,000-digit numbers, 100,000-character invalid input |
| Floating point | `+inf`, `-inf`, `nan`, `-0.0`, subnormals (`5e-324`), float overflow (`1e308 + 1e308`), `1e400` |
| Integers | values far beyond the float range (`10**400`) in every numeric function |
| Wrong input | `None`, `bytes`, `bool`, `complex`, strings where numbers belong and numbers where strings belong |
| Properties | seeded random checks: round-trips, symmetry and triangle inequality of the edit distance, idempotent normalization, results within bounds |

## Defects found by the tests

Writing the edge-case tests exposed real defects in the first draft, which
were then fixed:

1. `mean`, `median` and `variance` crashed with `OverflowError` for integers
   larger than a float can hold.
2. `count_words` split Hindi and Thai words in the middle, because combining
   marks are not "word characters" in Python's `\w`.
3. `parse_number("1,")` was accepted, and `"1,5"` silently became fifteen even
   though many locales write that as one and a half. Commas are now accepted
   only as correctly placed thousands separators.

## Known limitations

* Strings are measured in code points, not grapheme clusters, so an emoji with
  a skin tone counts as two characters and `truncate` can cut between a base
  letter and its accent.
* `count_words` does no dictionary-based segmentation, so Chinese and Thai text
  without spaces counts as one word per uninterrupted run.
* `levenshtein` takes time proportional to the product of the two lengths.
* Line coverage was measured with the standard library `trace` module and
  reached every executable line of `datatools.py`; it does not prove every
  combination of inputs is correct.

## Tools used

Code, tests and documentation were generated with Claude Sonnet 5.5
(Anthropic) in the Claude chat app, which also ran the test suite and fixed the
defects listed above.