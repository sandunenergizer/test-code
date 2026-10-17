"""Text and number utilities with robust Unicode and IEEE 754 handling.

The module offers small, dependency-free helpers that are easy to get
subtly wrong: counting words in scripts, comparing strings that contain
combining characters, and computing statistics over values that may be
infinite or not-a-number.

Typical usage::

    >>> from datatools import slugify, mean
    >>> slugify("Hello, World!")
    'hello-world'
    >>> mean([1, 2, 3])
    2.0

Design rules shared by every function:

* Arguments of the wrong *type* raise :exc:`TypeError`.
* Arguments of the right type but with an unusable *value* raise
  :exc:`ValueError`.
* Strings are processed as sequences of Unicode code points, not bytes
  and not user-perceived characters (grapheme clusters).
* ``bool`` is deliberately rejected wherever a number is expected,
  because ``True`` silently behaving as ``1`` hides programming errors.
"""

from __future__ import annotations

import itertools
import math
import re
import unicodedata

__all__ = [
    "normalize_text",
    "count_words",
    "count_characters",
    "truncate",
    "slugify",
    "is_palindrome",
    "levenshtein",
    "run_length_encode",
    "run_length_decode",
    "safe_divide",
    "clamp",
    "mean",
    "median",
    "variance",
    "describe",
    "format_bytes",
    "parse_number",
]

__version__ = "1.0.0"

_WHITESPACE_RE = re.compile(r"\s+")
_GROUPED_NUMBER_RE = re.compile(
    r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?:[eE][+-]?\d+)?")
_SEPARATOR_RE = re.compile(r"[-_\s]+")
_NORMAL_FORMS = ("NFC", "NFD", "NFKC", "NFKD")
_BYTE_UNITS = ("B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB")


# ---------------------------------------------------------------------------
# Argument validation helpers
# ---------------------------------------------------------------------------

def _require_str(value, name):
    """Raise :exc:`TypeError` unless *value* is a :class:`str`.

    :param value: The object to check.
    :param str name: Argument name used in the error message.
    :raises TypeError: If *value* is not a string.
    """
    if not isinstance(value, str):
        raise TypeError(f"{name} must be str, not {type(value).__name__}")


def _is_real(value):
    """Return ``True`` for ``int`` and ``float`` values, but not ``bool``."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_infinite(value):
    """Return ``True`` for ``+inf`` and ``-inf``; huge ints are finite."""
    return isinstance(value, float) and math.isinf(value)


def _require_real(value, name):
    """Raise :exc:`TypeError` unless *value* is a real number.

    :param value: The object to check.
    :param str name: Argument name used in the error message.
    :raises TypeError: If *value* is not an ``int`` or ``float``.
    """
    if not _is_real(value):
        raise TypeError(
            f"{name} must be int or float, not {type(value).__name__}")


def _validate_numbers(values):
    """Convert *values* to a non-empty list of numbers without NaN.

    :param values: An iterable of ``int`` or ``float`` objects.
    :returns: The values as a new list.
    :rtype: list
    :raises TypeError: If *values* is a string or contains non-numbers.
    :raises ValueError: If *values* is empty or contains NaN.
    """
    if isinstance(values, (str, bytes)):
        raise TypeError("values must be an iterable of numbers")
    data = list(values)
    if not data:
        raise ValueError("values must not be empty")
    for item in data:
        _require_real(item, "element")
        if isinstance(item, float) and math.isnan(item):
            raise ValueError("values must not contain NaN")
    return data


# ---------------------------------------------------------------------------
# Text functions
# ---------------------------------------------------------------------------

def normalize_text(text, form="NFC", collapse_whitespace=True):
    """Normalize Unicode text and optionally collapse whitespace.

    Visually identical strings can have different code point sequences,
    for example ``"\\u00e9"`` versus ``"e\\u0301"``. Normalizing first
    makes comparisons reliable.

    :param str text: The text to normalize.
    :param str form: One of ``"NFC"``, ``"NFD"``, ``"NFKC"`` or ``"NFKD"``.
    :param bool collapse_whitespace: If true, every run of Unicode
        whitespace (including U+00A0 and U+3000) becomes one space and
        leading and trailing whitespace is removed.
    :returns: The normalized text.
    :rtype: str
    :raises TypeError: If *text* is not a string.
    :raises ValueError: If *form* is not a valid normalization form.

    .. code-block:: python

        >>> normalize_text("e\\u0301") == "\\u00e9"
        True
        >>> normalize_text("  a \\t\\n b  ")
        'a b'
    """
    _require_str(text, "text")
    if form not in _NORMAL_FORMS:
        raise ValueError(f"form must be one of {_NORMAL_FORMS}, got {form!r}")
    result = unicodedata.normalize(form, text)
    if collapse_whitespace:
        result = _WHITESPACE_RE.sub(" ", result).strip()
    return result


def count_words(text):
    """Count runs of letters, digits, combining marks and underscores.

    A "word" is a maximal run of code points whose Unicode category is
    a letter, number or mark, or that are underscores. Combining marks
    are included so that scripts such as Hindi and Thai are not split in
    the middle of a word. Punctuation, symbols and emoji separate words,
    so ``"don't"`` counts as two words. Scripts written without spaces,
    such as Chinese, count as one word per uninterrupted run; no
    dictionary-based segmentation is done.

    :param str text: The text to inspect.
    :returns: The number of words.
    :rtype: int
    :raises TypeError: If *text* is not a string.

    .. code-block:: python

        >>> count_words("one two  three")
        3
        >>> count_words("\\u4f60\\u597d \\u4e16\\u754c")
        2
    """
    _require_str(text, "text")
    count = 0
    inside_word = False
    for char in text:
        word_char = char == "_" or unicodedata.category(char)[0] in "LNM"
        if word_char and not inside_word:
            count += 1
        inside_word = word_char
    return count


def count_characters(text, ignore_whitespace=False):
    """Count Unicode code points in *text*.

    Note that one user-perceived character may consist of several code
    points, for example a base letter plus a combining accent, or an
    emoji with a skin-tone modifier.

    :param str text: The text to inspect.
    :param bool ignore_whitespace: If true, whitespace is not counted.
    :returns: The number of code points.
    :rtype: int
    :raises TypeError: If *text* is not a string.

    .. code-block:: python

        >>> count_characters("a b")
        3
        >>> count_characters("a b", ignore_whitespace=True)
        2
    """
    _require_str(text, "text")
    if ignore_whitespace:
        return sum(1 for char in text if not char.isspace())
    return len(text)


def truncate(text, max_length, suffix="..."):
    """Shorten *text* to at most *max_length* code points.

    If the text is cut, *suffix* is appended and counts toward the
    limit. If *max_length* is not larger than the suffix, only the
    first *max_length* code points of the suffix are returned.

    :param str text: The text to shorten.
    :param int max_length: Maximum length of the result, at least 0.
    :param str suffix: Marker appended to shortened text.
    :returns: The original or shortened text.
    :rtype: str
    :raises TypeError: If an argument has the wrong type.
    :raises ValueError: If *max_length* is negative.

    .. code-block:: python

        >>> truncate("Hello, world", 8)
        'Hello...'
        >>> truncate("short", 10)
        'short'
    """
    _require_str(text, "text")
    _require_str(suffix, "suffix")
    if isinstance(max_length, bool) or not isinstance(max_length, int):
        raise TypeError("max_length must be int")
    if max_length < 0:
        raise ValueError("max_length must not be negative")
    if len(text) <= max_length:
        return text
    if max_length <= len(suffix):
        return suffix[:max_length]
    return text[:max_length - len(suffix)] + suffix


def slugify(text, allow_unicode=False):
    """Convert *text* into a lowercase, hyphen-separated URL slug.

    Letters, numbers and combining marks are kept. Punctuation, symbols
    and emoji are dropped. Whitespace, hyphens and underscores become a
    single hyphen.

    :param str text: The text to convert.
    :param bool allow_unicode: If false, text is reduced to ASCII, so
        accents are removed and scripts such as Chinese vanish. If true,
        letters of every script are kept.
    :returns: The slug, possibly an empty string.
    :rtype: str
    :raises TypeError: If *text* is not a string.

    .. code-block:: python

        >>> slugify("Caf\\u00e9 M\\u00fcnster, 2024!")
        'cafe-munster-2024'
        >>> slugify("\\u4f60\\u597d \\u4e16\\u754c", allow_unicode=True) == (
        ...     "\\u4f60\\u597d-\\u4e16\\u754c")
        True
    """
    _require_str(text, "text")
    if allow_unicode:
        value = unicodedata.normalize("NFKC", text)
    else:
        value = unicodedata.normalize("NFKD", text)
        value = value.encode("ascii", "ignore").decode("ascii")
    kept = "".join(
        char for char in value
        if unicodedata.category(char)[0] in "LNM" or char in "-_"
        or char.isspace()
    )
    return _SEPARATOR_RE.sub("-", kept.casefold()).strip("-")


def is_palindrome(text):
    """Return whether *text* reads the same in both directions.

    The text is compatibility-normalized, case-folded, and reduced to
    letters and digits first, so spaces, punctuation and emoji are
    ignored. The empty string is a palindrome.

    :param str text: The text to test.
    :returns: ``True`` if *text* is a palindrome.
    :rtype: bool
    :raises TypeError: If *text* is not a string.

    .. code-block:: python

        >>> is_palindrome("A man, a plan, a canal: Panama")
        True
        >>> is_palindrome("hello")
        False
    """
    _require_str(text, "text")
    folded = unicodedata.normalize("NFKC", text).casefold()
    cleaned = [char for char in folded if char.isalnum()]
    return cleaned == cleaned[::-1]


def levenshtein(first, second):
    """Return the edit distance between two strings.

    The distance is the minimum number of single code point insertions,
    deletions and substitutions that turn *first* into *second*. Inputs
    are compared as given, so normalize them first if canonically
    equivalent strings must count as equal. Time is proportional to the
    product of the lengths; memory is proportional to the shorter one.

    :param str first: The first string.
    :param str second: The second string.
    :returns: The edit distance, at least 0.
    :rtype: int
    :raises TypeError: If either argument is not a string.

    .. code-block:: python

        >>> levenshtein("kitten", "sitting")
        3
    """
    _require_str(first, "first")
    _require_str(second, "second")
    if first == second:
        return 0
    if len(first) < len(second):
        first, second = second, first
    if not second:
        return len(first)
    previous = list(range(len(second) + 1))
    for row, char_a in enumerate(first, start=1):
        current = [row]
        for col, char_b in enumerate(second, start=1):
            cost = 0 if char_a == char_b else 1
            current.append(min(previous[col] + 1,
                               current[col - 1] + 1,
                               previous[col - 1] + cost))
        previous = current
    return previous[-1]


def run_length_encode(text):
    """Compress *text* into a list of ``(character, count)`` pairs.

    Pairs are returned instead of a string such as ``"3a2b"`` so that
    texts containing digits stay unambiguous.

    :param str text: The text to encode.
    :returns: One pair per run of identical code points.
    :rtype: list
    :raises TypeError: If *text* is not a string.

    .. code-block:: python

        >>> run_length_encode("aaabcc")
        [('a', 3), ('b', 1), ('c', 2)]
    """
    _require_str(text, "text")
    return [(char, sum(1 for _ in group))
            for char, group in itertools.groupby(text)]


def run_length_decode(pairs):
    """Rebuild text from the output of :func:`run_length_encode`.

    :param pairs: An iterable of ``(character, count)`` pairs, where
        each character is a single-code-point string and each count is
        an ``int`` of at least 1.
    :returns: The decoded text.
    :rtype: str
    :raises TypeError: If a pair or its parts have the wrong type.
    :raises ValueError: If a character is not exactly one code point or
        a count is below 1.

    .. code-block:: python

        >>> run_length_decode([("a", 3), ("b", 1)])
        'aaab'
    """
    parts = []
    for pair in pairs:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise TypeError("each item must be a (character, count) pair")
        char, count = pair
        _require_str(char, "character")
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError("count must be int")
        if len(char) != 1:
            raise ValueError("character must be exactly one code point")
        if count < 1:
            raise ValueError("count must be at least 1")
        parts.append(char * count)
    return "".join(parts)


# ---------------------------------------------------------------------------
# Number functions
# ---------------------------------------------------------------------------

def safe_divide(numerator, denominator, default=None):
    """Divide two numbers, handling zero and overflow explicitly.

    Infinity and NaN follow IEEE 754 rules, so ``inf / inf`` is NaN.
    Dividing huge integers whose quotient does not fit in a float
    returns a signed infinity instead of raising :exc:`OverflowError`.

    :param numerator: The dividend, an ``int`` or ``float``.
    :param denominator: The divisor, an ``int`` or ``float``.
    :param default: Value returned when *denominator* is zero. If it is
        ``None``, :exc:`ZeroDivisionError` is raised instead.
    :returns: The quotient or *default*.
    :raises TypeError: If an operand is not a real number.
    :raises ZeroDivisionError: If *denominator* is zero and no default
        was given.

    .. code-block:: python

        >>> safe_divide(6, 3)
        2.0
        >>> safe_divide(1, 0, default=0.0)
        0.0
    """
    _require_real(numerator, "numerator")
    _require_real(denominator, "denominator")
    if denominator == 0:
        if default is None:
            raise ZeroDivisionError("denominator must not be zero")
        return default
    try:
        return numerator / denominator
    except OverflowError:
        negative = (numerator < 0) != (denominator < 0)
        return -math.inf if negative else math.inf


def clamp(value, low, high):
    """Limit *value* to the closed interval from *low* to *high*.

    Infinite bounds are allowed; NaN is rejected everywhere because it
    has no meaningful order.

    :param value: The number to limit.
    :param low: The lower bound.
    :param high: The upper bound, not smaller than *low*.
    :returns: *low*, *high* or *value*, keeping its original type.
    :raises TypeError: If an argument is not a real number.
    :raises ValueError: If an argument is NaN or *low* exceeds *high*.

    .. code-block:: python

        >>> clamp(15, 0, 10)
        10
        >>> clamp(-3, 0, 10)
        0
    """
    for name, number in (("value", value), ("low", low), ("high", high)):
        _require_real(number, name)
        if isinstance(number, float) and math.isnan(number):
            raise ValueError(f"{name} must not be NaN")
    if low > high:
        raise ValueError("low must not be greater than high")
    return min(max(value, low), high)


def mean(values):
    """Return the arithmetic mean of *values*.

    A mix of ``+inf`` and ``-inf`` has no mean and raises an error. Sums
    that would overflow are computed by dividing each term first.

    :param values: A non-empty iterable of ``int`` or ``float``.
    :returns: The mean as a float, possibly infinite.
    :rtype: float
    :raises TypeError: If *values* is a string or has non-numbers.
    :raises ValueError: If *values* is empty, contains NaN, contains
        both positive and negative infinity, or mixes floats with
        integers too large for a float.

    .. code-block:: python

        >>> mean([1, 2, 3, 4])
        2.5
        >>> mean([1, float("inf")])
        inf
    """
    data = _validate_numbers(values)
    has_positive = any(item == math.inf for item in data)
    has_negative = any(item == -math.inf for item in data)
    if has_positive and has_negative:
        raise ValueError("mean of +inf and -inf is undefined")
    if has_positive:
        return math.inf
    if has_negative:
        return -math.inf
    count = len(data)
    if all(isinstance(item, int) for item in data):
        total = sum(data)
        try:
            return total / count
        except OverflowError:
            return math.inf if total > 0 else -math.inf
    try:
        return math.fsum(data) / count
    except OverflowError:
        pass
    try:
        return math.fsum(item / count for item in data)
    except OverflowError as exc:
        raise ValueError("values are too large to average") from exc


def median(values):
    """Return the median of *values*.

    For an even count the two middle values are averaged. The middle
    pair ``-inf`` and ``+inf`` has no average and raises an error.

    :param values: A non-empty iterable of ``int`` or ``float``.
    :returns: The median. An odd count returns an original element.
    :raises TypeError: If *values* is a string or has non-numbers.
    :raises ValueError: If *values* is empty, contains NaN, the two
        middle values are ``-inf`` and ``+inf``, or they mix a float
        with an integer too large for a float.

    .. code-block:: python

        >>> median([3, 1, 2])
        2
        >>> median([1, 2, 3, 4])
        2.5
    """
    ordered = sorted(_validate_numbers(values))
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    low, high = ordered[middle - 1], ordered[middle]
    if low == high:
        return low
    if _is_infinite(low) and _is_infinite(high):
        raise ValueError("median of -inf and +inf is undefined")
    if _is_infinite(low):
        return low
    if _is_infinite(high):
        return high
    if isinstance(low, int) and isinstance(high, int):
        total = low + high
        try:
            return total / 2
        except OverflowError:
            return math.inf if total > 0 else -math.inf
    try:
        total = low + high
        if math.isinf(total):
            raise OverflowError
        return total / 2
    except OverflowError:
        pass
    try:
        return low / 2 + high / 2
    except OverflowError as exc:
        raise ValueError("values are too large to average") from exc


def variance(values):
    """Return the population variance of *values*.

    :param values: A non-empty iterable of finite ``int`` or ``float``.
    :returns: The variance, or ``inf`` if it exceeds the float range.
    :rtype: float
    :raises TypeError: If *values* is a string or has non-numbers.
    :raises ValueError: If *values* is empty, contains NaN, contains an
        infinite value, or contains an integer too large for a float.

    .. code-block:: python

        >>> variance([1, 2, 3, 4])
        1.25
    """
    data = _validate_numbers(values)
    if any(_is_infinite(item) for item in data):
        raise ValueError("variance of infinite values is undefined")
    try:
        data = [float(item) for item in data]
    except OverflowError as exc:
        raise ValueError("values are too large for a float") from exc
    centre = mean(data)
    try:
        squares = ((item - centre) ** 2 for item in data)
        return math.fsum(squares) / len(data)
    except OverflowError:
        return math.inf


def describe(values):
    """Summarize *values* in a dictionary.

    :param values: A non-empty iterable of ``int`` or ``float``.
    :returns: A dictionary with the keys ``count``, ``min``, ``max``,
        ``mean``, ``median`` and ``variance``. The variance is ``nan``
        if any value is infinite.
    :rtype: dict
    :raises TypeError: If *values* is a string or has non-numbers.
    :raises ValueError: If *values* is empty, contains NaN, mixes
        positive and negative infinity, or contains an integer too
        large for a float.

    .. code-block:: python

        >>> describe([1, 2, 3])["median"]
        2
    """
    data = _validate_numbers(values)
    finite = not any(_is_infinite(item) for item in data)
    return {
        "count": len(data),
        "min": min(data),
        "max": max(data),
        "mean": mean(data),
        "median": median(data),
        "variance": variance(data) if finite else math.nan,
    }


def format_bytes(num_bytes, precision=1):
    """Format a byte count using binary prefixes such as ``KiB``.

    :param num_bytes: A finite, non-negative ``int`` or ``float``.
    :param int precision: Decimal places for units above bytes, 0 to 10.
    :returns: A string such as ``"1.5 KiB"``. Values beyond the largest
        unit are still expressed in ``YiB``.
    :rtype: str
    :raises TypeError: If an argument has the wrong type.
    :raises ValueError: If *num_bytes* is negative, NaN, infinite or too
        large for a float, or *precision* is out of range.

    .. code-block:: python

        >>> format_bytes(1536)
        '1.5 KiB'
        >>> format_bytes(512)
        '512 B'
    """
    _require_real(num_bytes, "num_bytes")
    if isinstance(precision, bool) or not isinstance(precision, int):
        raise TypeError("precision must be int")
    if not 0 <= precision <= 10:
        raise ValueError("precision must be between 0 and 10")
    try:
        value = float(num_bytes)
    except OverflowError as exc:
        raise ValueError("num_bytes is too large") from exc
    if math.isnan(value) or math.isinf(value):
        raise ValueError("num_bytes must be finite")
    if value < 0:
        raise ValueError("num_bytes must not be negative")
    unit = 0
    while unit < len(_BYTE_UNITS) - 1 and round(value, precision) >= 1024:
        value /= 1024
        unit += 1
    if unit == 0:
        return f"{int(value)} B"
    return f"{value:.{precision}f} {_BYTE_UNITS[unit]}"


def parse_number(text):
    """Parse a number from text written in many scripts and styles.

    Accepted: surrounding whitespace, digits of any Unicode decimal
    script such as Arabic-Indic, full-width digits, the Unicode minus
    sign, exponents, the words ``inf``, ``-inf`` and ``nan``, and comma
    thousands separators in groups of three such as ``"1,234.5"``.
    Ambiguous commas such as ``"1,5"`` (a decimal comma in many
    locales), underscores, spaces inside the number, hexadecimal
    notation and non-decimal numerals such as Chinese ``"\\u4e09"`` are
    rejected.

    :param str text: The text to parse.
    :returns: An ``int`` if the text is an integer, otherwise a float.
    :raises TypeError: If *text* is not a string.
    :raises ValueError: If *text* is not a valid number.

    .. code-block:: python

        >>> parse_number("1,234")
        1234
        >>> parse_number("\\u0664\\u0662")
        42
    """
    _require_str(text, "text")
    cleaned = unicodedata.normalize("NFKC", text).strip()
    cleaned = cleaned.replace("\u2212", "-")
    if "," in cleaned:
        if not _GROUPED_NUMBER_RE.fullmatch(cleaned):
            raise ValueError(f"not a number: {text!r}")
        cleaned = cleaned.replace(",", "")
    if not cleaned or "_" in cleaned:
        raise ValueError(f"not a number: {text!r}")
    try:
        return int(cleaned)
    except ValueError:
        pass
    try:
        return float(cleaned)
    except ValueError as exc:
        raise ValueError(f"not a number: {text!r}") from exc