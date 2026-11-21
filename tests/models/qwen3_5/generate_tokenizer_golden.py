"""Golden token ids for the Qwen tokenizer differential test (candidates LE-10).

Reads the tokenizer resources of one artifact (resource/text/tokenizer.json, tokenizer_config.json and
generation_config.json), encodes an adversarial corpus with Hugging Face `tokenizers` (add_special_tokens
off, as the Engine's frontend does for chat text), and writes
  <golden>      the corpus and its ids, with the tokenizer.json SHA-256 they belong to
  <resources>/  the three resource files, which the C++ test (test_tokenizer_golden_real.cpp) loads
The corpus is the shared deterministic one (tests/text/tokenizer_corpus.py).

  py -3.11 generate_tokenizer_golden.py <artifact> <golden.json> <resources dir>
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY))

from tokenizers import Tokenizer  # noqa: E402
import tokenizers  # noqa: E402

from tools.artifact.reader import Artifact  # noqa: E402
sys.path.insert(0, str(REPOSITORY / "tests" / "text"))
from tokenizer_corpus import corpus  # noqa: E402

RESOURCES = ("tokenizer.json", "tokenizer_config.json", "generation_config.json")

# Contractions under Oniguruma's Unicode case folding: U+017F LATIN SMALL LETTER LONG S folds to s, so "'ſ"
# is a contraction piece and the letters after it a piece of their own; the Kelvin sign folds to k, which no
# contraction has.
QWEN_EXTRA = ["'ſ", "it'ſ", "x'ſy", "IT'ſS", "'ſt", "he'K", "WE'RE", "'Ll", "y'ſſ"]


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__)
        return 2
    artifact_path, golden_path, resources_dir = sys.argv[1:]
    resources: dict[str, bytes] = {}
    with Artifact.open(artifact_path) as artifact:
        for name in RESOURCES:
            resources[name] = artifact.read_object(f"resource/text/{name}")
    target = Path(resources_dir)
    target.mkdir(parents=True, exist_ok=True)
    for name, payload in resources.items():
        (target / name).write_bytes(payload)

    tokenizer = Tokenizer.from_str(resources["tokenizer.json"].decode("utf-8"))
    cases = []
    for text in corpus(QWEN_EXTRA):
        cases.append({"text": text, "ids": tokenizer.encode(text, add_special_tokens=False).ids})
    golden = {
        "tokenizer_json_bytes": len(resources["tokenizer.json"]),
        "tokenizer_json_sha256": hashlib.sha256(resources["tokenizer.json"]).hexdigest(),
        "tokenizers_version": tokenizers.__version__,
        "cases": cases,
    }
    # One case per line, so a regenerated file diffs by case.
    head = json.dumps({key: value for key, value in golden.items() if key != "cases"}, ensure_ascii=False, indent=0)
    lines = [json.dumps(case, ensure_ascii=False, separators=(",", ":")) for case in cases]
    Path(golden_path).write_text(head[:-2] + ',\n"cases": [\n' + ",\n".join(lines) + "\n]\n}\n", encoding="utf-8")
    print(f"{len(cases)} cases, tokenizer {golden['tokenizer_json_sha256'][:16]}, tokenizers {tokenizers.__version__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
