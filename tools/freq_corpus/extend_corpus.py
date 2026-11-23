"""The conversations of a larger prepared corpus that a measured one lacks, for one stratum, in two parts of a given size, so
a ranking can be extended (the tool's counts add up) and its convergence read at the first part.

    python tools/freq_corpus/extend_corpus.py --measured train.jsonl --larger larger/train.jsonl --stratum code \
        --tokenizer tokenizer.json --first-tokens 810000 --out-a part_a.jsonl --out-b part_b.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--measured", type=Path, required=True)
    parser.add_argument("--larger", type=Path, required=True)
    parser.add_argument("--stratum", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--first-tokens", type=int, required=True, help="assistant tokens of part a")
    parser.add_argument("--out-a", type=Path, required=True)
    parser.add_argument("--out-b", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    with open(args.measured, encoding="utf-8") as measured:
        seen = {json.loads(line)["id"] for line in measured}
    tokens = {"a": 0, "b": 0}
    records = {"a": 0, "b": 0}
    with open(args.larger, encoding="utf-8") as larger, open(args.out_a, "w", encoding="utf-8", newline="\n") as a, \
            open(args.out_b, "w", encoding="utf-8", newline="\n") as b:
        for line in larger:
            record = json.loads(line)
            if record["stratum"] != args.stratum or record["id"] in seen:
                continue
            part = "a" if tokens["a"] < args.first_tokens else "b"
            for message in record["messages"]:
                if message["role"] == "assistant":
                    tokens[part] += len(tokenizer.encode(message["content"], add_special_tokens=False).ids)
            records[part] += 1
            (a if part == "a" else b).write(line)
    print(json.dumps({"conversations": records, "assistant_tokens": tokens}))


if __name__ == "__main__":
    main()
