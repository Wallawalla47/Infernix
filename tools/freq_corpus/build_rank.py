"""The rank order of a model's vocabulary from its measured counts, and how well it covers held-out text.

    python tools/freq_corpus/build_rank.py --train ranking.train.counts.i64 [more.counts.i64 ...] \
        --heldout ranking.heldout.counts.i64 --groundtruth groundtruth.counts.i64 --tokenizer tokenizer.json --out OUT_DIR

Each stratum's argmax counts are normalised to frequencies and mixed by WEIGHTS (English text and code first). Tokens are
ordered by that mix, ties and never-predicted tokens by their count in the corpus text, then by id; the tokenizer's special
tokens come first, so a shortlist of any length can stop a turn or open a tool call. Several --train files are added
(a measurement extended later). Writes rank.order.i32 (int32 LE [vocab]) and coverage.report.json: the share of the held-out
split's argmax positions each shortlist length holds, per stratum and under the same mix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

STRATA = ("en_chat", "zh_chat", "other_chat", "code", "math")
WEIGHTS = {"en_chat": 0.50, "zh_chat": 0.20, "other_chat": 0.12, "code": 0.15, "math": 0.03}
GRID = (1024, 2048, 4096, 8192, 16384, 32768, 49152, 65536, 98304, 131072, 163840)


def rows_of(path: Path, vocab: int, rows: int) -> np.ndarray:
    data = np.fromfile(path, dtype="<i8")
    if data.size != vocab * rows:
        raise ValueError(f"{path}: {data.size} counts, expected {rows} rows of {vocab}")
    return data.reshape(rows, vocab)


def special_ids(tokenizer: Path) -> np.ndarray:
    added = json.loads(tokenizer.read_text(encoding="utf-8"))["added_tokens"]
    return np.asarray(sorted(int(token["id"]) for token in added if token.get("special")), dtype=np.int64)


def mixed(counts: np.ndarray) -> np.ndarray:
    """WEIGHTS' mix of the strata rows' frequencies ([1 + strata, vocab] counts)."""
    out = np.zeros(counts.shape[1], dtype=np.float64)
    for i, name in enumerate(STRATA):
        total = int(counts[i + 1].sum())
        if total > 0:
            out += WEIGHTS[name] * counts[i + 1] / total
    return out


def order_of(primary: np.ndarray, secondary: np.ndarray, first: np.ndarray) -> np.ndarray:
    """Descending (primary, secondary), then ascending id; `first` ahead of everything in that same order."""
    order = np.lexsort((np.arange(primary.size), -secondary, -primary))
    forced = np.zeros(primary.size, dtype=bool)
    forced[first] = True
    return np.concatenate((order[forced[order]], order[~forced[order]]))


def held(order: np.ndarray, counts: np.ndarray, rows: int) -> float:
    return float(counts[order[:rows]].sum()) / max(int(counts.sum()), 1)


def mixed_coverage(order: np.ndarray, counts: np.ndarray, rows: int) -> float:
    """The held-out share of a shortlist under WEIGHTS ([1 + strata, vocab] counts)."""
    return sum(WEIGHTS[name] * held(order, counts[i + 1], rows) for i, name in enumerate(STRATA))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--groundtruth", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    corpus = np.fromfile(args.groundtruth, dtype="<i8")
    vocab = corpus.size
    train = sum(rows_of(path, vocab, 1 + len(STRATA)) for path in args.train)
    heldout = rows_of(args.heldout, vocab, 1 + len(STRATA))
    special = special_ids(args.tokenizer)
    if special.size == 0 or special[-1] >= vocab:
        raise ValueError("the tokenizer's special tokens must lie in the vocabulary")

    order = order_of(mixed(train), corpus, special)
    if not np.array_equal(np.sort(order), np.arange(vocab)):
        raise ValueError("the rank order is not a permutation of the vocabulary")

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "rank.order.i32"
    order.astype("<i4").tofile(path)
    report = {
        "artifact_type": "token_rank_coverage",
        "schema_version": 2,
        "vocab": vocab,
        "rank_by": "the strata's argmax frequencies under stratum_weights, then corpus-text counts, then id; special tokens first",
        "stratum_weights": WEIGHTS,
        "special_tokens": int(special.size),
        "rank_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "train_positions": {name: int(train[i + 1].sum()) for i, name in enumerate(STRATA)},
        "heldout_positions": {name: int(heldout[i + 1].sum()) for i, name in enumerate(STRATA)},
        "tokens_the_model_predicted_in_train": int(np.count_nonzero(train[0])),
        "heldout_coverage": {
            "weighted": {str(rows): mixed_coverage(order, heldout, rows) for rows in GRID},
            **{name: {str(rows): held(order, heldout[i + 1], rows) for rows in GRID} for i, name in enumerate(STRATA)},
        },
    }
    (args.out / "coverage.report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("train_positions", "heldout_positions", "tokens_the_model_predicted_in_train")}))
    for rows in GRID:
        key = str(rows)
        print(f"{rows:7d}: weighted {report['heldout_coverage']['weighted'][key] * 100:6.2f} %  " +
              "  ".join(f"{name} {report['heldout_coverage'][name][key] * 100:.2f}" for name in STRATA))


if __name__ == "__main__":
    main()
