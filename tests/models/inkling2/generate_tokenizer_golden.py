"""Golden token ids for the Inkling 2 tokenizer differential test (text/tokenizer.cpp's Inkling pipeline).

Reads the tokenizer resources of an Inkling 2 checkpoint directory or artifact (tokenizer.json, tokenizer_config.json,
generation_config.json), encodes a corpus with Hugging Face `tokenizers` (add_special_tokens off, as the frontend does
for chat text) and writes
  <golden>      the corpus and its ids, with the tokenizer.json SHA-256 they belong to
  <resources>/  the three resource files, which the C++ test (test_tokenizer_golden_real.cpp) loads

The corpus is the shared one (tests/text/tokenizer_corpus.py) plus what Inkling's Split pattern distinguishes: the
upper/lower letter-run classes and their backtracking (camel and title case, Lt, Lm, Lo and marks between them), the
case-insensitive contractions (U+017F folds to s), digit runs past three, punctuation followed by slashes and
newlines, and whitespace before punctuation and words. With --sweep the corpus is instead every codepoint (surrogates
excepted) in blocks of 64, each block joined bare, by spaces, by letters of both cases, by digits and by apostrophes:
a qualification run, too large to commit.

  py -3.11 generate_tokenizer_golden.py [--sweep] <checkpoint dir | artifact> <golden.json> <resources dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(REPOSITORY / "tests" / "text"))

import tokenizers  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from tokenizer_corpus import corpus  # noqa: E402

RESOURCES = ("tokenizer.json", "tokenizer_config.json", "generation_config.json")

INKLING = [
    # contractions, Unicode case folding (U+017F LATIN SMALL LETTER LONG S folds to s)
    "'s", "'S", "'ſ", "it'ſ", "IT'S", "We'RE", "x'Ve", "y'LL", "z'lL", "w'Ll", "q'd'", "''s", "a'ſt",
    "Ǆ'ſ", " 'ſ", "he'K", "she'İ", "THEY'VE", "o'cLoCk", "it's'", "'re'", "abc'", "'",
    # letter-run classes and their backtracking
    "McDonald", "iPhone", "HTMLParser", "ABCdef", "abcDEF", "aBcDeF", "XMLHttpRequest", "IOError", "ID's",
    "ǅungla", "ǄUNGLA", "ǆ", "ᾼᾼ", "ʰello", "ʰ", "ǈ", "Aʰa", "ʰA",
    "Σίσυφος", "ΣΊΣΥΦΟΣ", "Áb",
    "́a", "́", "́́", "x⃝", "ॐ", "ǅA", "Aǅ", "ꙮ", "ⅷ",
    "ⓐ", "\U0001d400\U0001d41a", "Ⓐ", "Aあa", "あA", "aあB", "क्ष",
    "אָ", "اَل", "ÀàÁ", "İstanbulİ", "ẞß",
    # numbers
    "1234", "12345", "1234567", "١٢٣٤", "１２３４", "²³⁴⁵",
    "½¼", "Ⅻ12", "1a2", "12ab", "a12", "1,2345", "v1.2.3", "2026-11-21",
    # punctuation runs, slashes and newlines
    "...//", "a.b/c", "!/\n", " ///", "-->\r\n", "?/\r\n/", "*/", " */\n", "</div>\n", "]]>\r\n\r\n", "/*", "//",
    "https://a.b/c/", ";\n\n", "\"\n", "'/'", " .", "  .", "\t.", "　.", " 　x", ".́", "́.",
    # prefixes before words
    "\tword", " word", "　word", "#word", "@user", "$var", "(Hello", "​word", "﻿word", "­word",
    " word", "\nword", "\r\nword", " \nword", "\u0085word", "-Word", "_word", "©word",
    # Inkling's own markers
    "<|message_user|>", "<|message_model|>", "<|content_text|>", "<|content_thinking|>", "<|end_message|>",
    "<|content_model_end_sampling|>", "<|content_invoke_tool_json|>", "<|unused_200054|>", "<|begin_of_text|>",
]


def sweep() -> list[str]:
    points = [c for c in range(0x110000) if not 0xD800 <= c <= 0xDFFF]
    out: list[str] = []
    for first in range(0, len(points), 64):
        chars = [chr(c) for c in points[first:first + 64]]
        out.append("".join(chars))
        for joiner in (" ", "a", "A", "1", "'s", "ſ", "́"):
            out.append(joiner.join(chars) + joiner)
    return out


def read_resources(source: Path) -> dict[str, bytes]:
    if source.is_dir():
        return {name: (source / name).read_bytes() for name in RESOURCES}
    from tools.artifact.reader import Artifact

    with Artifact.open(source) as artifact:
        return {name: artifact.read_object(f"resource/text/{name}") for name in RESOURCES}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("source", type=Path)
    parser.add_argument("golden", type=Path)
    parser.add_argument("resources", type=Path)
    args = parser.parse_args()
    resources = read_resources(args.source)
    args.resources.mkdir(parents=True, exist_ok=True)
    for name, payload in resources.items():
        (args.resources / name).write_bytes(payload)

    tokenizer = Tokenizer.from_str(resources["tokenizer.json"].decode("utf-8"))
    texts = sweep() if args.sweep else corpus(INKLING, size=2600)
    encoded = tokenizer.encode_batch(texts, add_special_tokens=False)
    cases = [{"text": text, "ids": item.ids} for text, item in zip(texts, encoded)]
    golden = {
        "tokenizer_json_bytes": len(resources["tokenizer.json"]),
        "tokenizer_json_sha256": hashlib.sha256(resources["tokenizer.json"]).hexdigest(),
        "tokenizers_version": tokenizers.__version__,
        "cases": cases,
    }
    # One case per line, so a regenerated file diffs by case.
    head = json.dumps({key: value for key, value in golden.items() if key != "cases"}, ensure_ascii=False, indent=0)
    lines = [json.dumps(case, ensure_ascii=False, separators=(",", ":")) for case in cases]
    args.golden.write_text(head[:-2] + ',\n"cases": [\n' + ",\n".join(lines) + "\n]\n}\n", encoding="utf-8")
    print(f"{len(cases)} cases, tokenizer {golden['tokenizer_json_sha256'][:16]}, tokenizers {tokenizers.__version__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
