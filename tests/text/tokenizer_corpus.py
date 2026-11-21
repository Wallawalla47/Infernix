"""The adversarial corpus of the tokenizer golden tests (tests/models/*/generate_tokenizer_golden.py): whitespace
families (CRLF runs, tabs, NBSP, ideographic and zero-width spaces), combining marks, ZWJ emoji, CJK and other
scripts, number forms, apostrophes, paths and URLs, code, markup that looks like special tokens, and seeded random
mixtures of those. Deterministic: a family's extra strings join the alphabet of the random mixtures."""

from __future__ import annotations

import random

WHITESPACE = [
    " ", "  ", "   ", "\t", "\t\t", " \t", "\t ", "\n", "\n\n", "\n\n\n", "\r", "\r\n", "\r\n\r\n", "\n\r",
    " \n", "\n ", " \r\n", "\r\n ", "\u00a0", "\u00a0\u00a0", "\u2002", "\u2003", "\u3000", "\u200b",
    "\u200d", "\u200c", "\u2028", "\u2029", "\x0b", "\x0c", "\x85", "\ufeff", "\u00ad", " \u00a0 ",
]

WORDS = [
    "hello", "Hello", "HELLO", "don't", "DON'T", "I'm", "it's", "we'll", "they've", "you'd", "she're",
    "O'Brien", "rock'n'roll", "it\u2019s", "l'\u00e9t\u00e9", "naïve", "café", "e\u0301", "a\u0308\u0323",
    "résumé", "Ünïcödé", "ǅ", "ß", "İstanbul", "東京", "日本語のテキスト", "한국어", "ひらがなカタカナ",
    "Привет", "мир", "مرحبا", "שלום", "हिंदी", "क्षत्रिय", "ไทย", "ជំរាបសួរ", "ᚠᚢᚦ", "😀", "👨\u200d👩\u200d👧\u200d👦",
    "🇯🇵", "👍🏽", "❤\ufe0f", "x²", "½", "Ⅷ", "٣٤", "１２３", "①②", "123", "0", "007", "3.14159", "1,234,567",
    "1e-9", "0x1F", "a1b2c3", "_", "__init__", "snake_case", "camelCase", "kebab-case", "C:\\Users\\Ian\\file.txt",
    "\\\\server\\share", "/usr/local/bin", "~/.config", "https://example.com/a?b=c&d=e#f", "user@example.com",
    "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>", "<tool_call>", "</tool_call>",
    "<function=get>", "<parameter=x>", "<|vision_start|>", "<|image_pad|>", "<|fim_prefix|>", "<s>", "</s>",
    "!!!", "...", "?!", "->", "=>", "::", "&&", "||", "{}", "[]", "()", '""', "'", "''", "`x`", "***", "---",
    "```python\nprint(1)\n```", "def f(x):\n    return x", "if (a && b) { c(); }", "SELECT * FROM t;",
    "# heading", "- item", "1. item", "| a | b |", "\\n", "\\r\\n", "\\u00e9", "%20", "&amp;", "\u0000",
    "\x01", "\x1b[0m", "\x7f", "\u0080", "\ufffd", "\U0001f600\U0001f601", "\U00010348", "\U0001d400",
]


def corpus(extra: list[str] = (), *, size: int = 2200) -> list[str]:
    """The unique nonempty strings: the fixed families, then ``extra``, then random mixtures up to ``size``."""

    rng = random.Random(20261026)
    out: list[str] = []
    out.extend(WHITESPACE)
    out.extend(WORDS)
    for space in WHITESPACE:
        for word in WORDS[::7]:
            out.extend((space + word, word + space, space + word + space, word + space + word))
    for count in (2, 3, 5, 8, 20):
        out.append("\r\n" * count)
        out.append("\n" * count)
        out.append(" " * count)
        out.append("\t" * count)
        out.append("\r\n\r\n".join(["line"] * count))
        out.append("0123456789" * count)
        out.append(("a" + "\u0301") * count)
    out.extend([" " * 100, "x" + " " * 100 + "y", "\n" * 50 + "z", "é" * 64, "😀" * 33, "a" * 300, "ab" * 150])
    out.extend(extra)
    alphabet = WORDS + WHITESPACE + list(extra)
    while len(out) < size:
        out.append("".join(rng.choice(alphabet) for _ in range(rng.randint(2, 7))))
    seen: set[str] = set()
    unique: list[str] = []
    for text in out:
        if text not in seen and text:
            seen.add(text)
            unique.append(text)
    return unique
