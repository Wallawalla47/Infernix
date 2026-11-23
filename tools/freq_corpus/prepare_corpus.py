"""Builds the chat corpus of a token-frequency ranking from public datasets (docs/maintainer/inkling2_mini-design.md §13.6).

    python tools/freq_corpus/prepare_corpus.py --raw RAW_DIR --tokenizer tokenizer.json --out OUT_DIR

RAW_DIR holds the parquet shards and their MANIFEST.json. Each conversation goes to one stratum (en_chat, zh_chat,
other_chat, code, math) and one split (a tenth, by the hash of its id, is held out). Within a stratum the conversations
are taken in the order of that hash until the stratum's assistant tokens reach its budget, so a run is deterministic.
The default budgets keep the proportions of NInfer's Qwen ranking (42 / 35 / 15 / 6 / 2 %).

Writes train.jsonl and heldout.jsonl ({"id", "stratum", "source", "messages"}), the assistant tokens of EVERY usable
conversation as groundtruth.counts.i64 (int64 LE [vocab], the tie-break for tokens the model's own counts never reach)
and corpus.manifest.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

STRATA = ("en_chat", "zh_chat", "other_chat", "code", "math")
SHARES = {"en_chat": 0.42, "zh_chat": 0.35, "other_chat": 0.15, "code": 0.06, "math": 0.02}
CALCULATOR = re.compile(r"<<[^<>]*>>")


def language_stratum(language: str) -> str:
    low = language.lower()
    if low in ("english", "en"):
        return "en_chat"
    if "chinese" in low or low == "zh" or low.startswith("zh-"):
        return "zh_chat"
    return "other_chat"


def turns(*pairs: str) -> list[dict]:
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": text} for i, text in enumerate(pairs)]


def well_formed(messages: list[dict]) -> list[dict] | None:
    """User and assistant alternating from a user turn, ending on an assistant turn, no empty message."""
    messages = list(messages)
    if messages and messages[-1]["role"] != "assistant":
        messages.pop()
    if len(messages) < 2 or len(messages) % 2:
        return None
    for i, message in enumerate(messages):
        if message["role"] != ("user" if i % 2 == 0 else "assistant") or not message["content"].strip():
            return None
    return messages


def rows(path: Path, columns: list[str]):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=4096, columns=columns):
        yield from batch.to_pylist()


def wildchat(folder: Path):
    # Only the messages' role and text (their timestamps need a timezone database to convert).
    columns = ["conversation_hash", "conversation.list.element.content", "conversation.list.element.role",
               "language", "toxic", "redacted"]
    for path in sorted(folder.glob("*.parquet")):
        for row in rows(path, columns):
            if row["toxic"] or row["redacted"] or row["language"] in (None, "Nolang"):
                continue
            messages = [{"role": m["role"], "content": m["content"] or ""} for m in row["conversation"]]
            yield "wildchat-" + row["conversation_hash"], language_stratum(row["language"]), "wildchat", messages


def aya(folder: Path):
    for path in sorted(folder.glob("*.parquet")):
        for i, row in enumerate(rows(path, ["inputs", "targets", "language"])):
            yield f"aya-{path.stem}-{i}", language_stratum(row["language"] or ""), "aya", turns(row["inputs"] or "", row["targets"] or "")


def oasst(folder: Path):
    """Each message tree's best path: from the root prompt, the lowest-ranked (best) reply at every step."""
    columns = ["message_id", "parent_id", "text", "role", "lang", "review_result", "deleted", "rank", "message_tree_id"]
    children, roots = defaultdict(list), []
    for path in sorted(folder.glob("*.parquet")):
        for row in rows(path, columns):
            if row["deleted"] or row["review_result"] is False:
                continue
            if row["parent_id"] is None:
                roots.append(row)
            else:
                children[row["parent_id"]].append(row)
    for root in roots:
        messages, node = [], root
        while node is not None:
            messages.append({"role": "user" if node["role"] == "prompter" else "assistant", "content": node["text"] or ""})
            replies = children.get(node["message_id"], [])
            node = min(replies, key=lambda r: (r["rank"] is None, r["rank"] or 0, r["message_id"])) if replies else None
        yield "oasst-" + root["message_id"], language_stratum(root["lang"] or ""), "oasst2", messages


def pairs(folder: Path, prefix: str, stratum: str, question: str, answer: str, clean=lambda text: text):
    for path in sorted(folder.glob("*.parquet")):
        for i, row in enumerate(rows(path, [question, answer])):
            yield f"{prefix}-{path.stem}-{i}", stratum, prefix, turns(row[question] or "", clean(row[answer] or ""))


def gsm8k_answer(text: str) -> str:
    return CALCULATOR.sub("", text).replace("####", "Answer:")


def conversations(raw: Path):
    yield from wildchat(raw / "allenai__WildChat-1M")
    yield from aya(raw / "CohereLabs__aya_dataset")
    yield from oasst(raw / "OpenAssistant__oasst2")
    yield from pairs(raw / "nvidia__OpenCodeInstruct", "opencode", "code", "input", "output")
    yield from pairs(raw / "nvidia__OpenMathInstruct-2", "openmath", "math", "problem", "generated_solution")
    yield from pairs(raw / "openai__gsm8k", "gsm8k", "math", "question", "answer", gsm8k_answer)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-tokens", type=int, default=12_000_000, help="assistant tokens of the train split")
    parser.add_argument("--heldout-fraction", type=int, default=10, help="one conversation in this many is held out")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    vocab = tokenizer.get_vocab_size(with_added_tokens=True)

    by_stratum: dict[str, list] = defaultdict(list)
    skipped = 0
    for identifier, stratum, source, messages in conversations(args.raw):
        messages = well_formed(messages)
        if messages is None:
            skipped += 1
            continue
        digest = hashlib.sha256(identifier.encode("utf-8")).digest()
        by_stratum[stratum].append((digest, identifier, source, messages))

    counts = np.zeros(vocab, dtype=np.int64)
    budgets = {s: int(args.train_tokens * SHARES[s]) for s in STRATA}
    taken: Counter = Counter()
    sources: Counter = Counter()
    conversations_taken: Counter = Counter()
    with open(args.out / "train.jsonl", "w", encoding="utf-8", newline="\n") as train, \
            open(args.out / "heldout.jsonl", "w", encoding="utf-8", newline="\n") as heldout:
        for stratum in STRATA:
            items = sorted(by_stratum[stratum], key=lambda item: item[0])
            for start in range(0, len(items), 2048):
                chunk = items[start : start + 2048]
                texts, owners = [], []
                for index, item in enumerate(chunk):
                    for message in item[3]:
                        if message["role"] == "assistant":
                            texts.append(message["content"])
                            owners.append(index)
                lengths = [0] * len(chunk)
                for owner, encoding in zip(owners, tokenizer.encode_batch(texts, add_special_tokens=False)):
                    ids = np.asarray(encoding.ids, dtype=np.int64)
                    np.add.at(counts, ids, 1)
                    lengths[owner] += len(ids)
                for (digest, identifier, source, messages), length in zip(chunk, lengths):
                    held = int.from_bytes(digest[:8], "little") % args.heldout_fraction == 0
                    split = "heldout" if held else "train"
                    limit = budgets[stratum] if split == "train" else budgets[stratum] // (args.heldout_fraction - 1)
                    if taken[(stratum, split)] >= limit:
                        continue
                    taken[(stratum, split)] += length
                    sources[(stratum, split, source)] += length
                    conversations_taken[(stratum, split)] += 1
                    record = {"id": identifier, "stratum": stratum, "source": source, "messages": messages}
                    (heldout if held else train).write(json.dumps(record, ensure_ascii=False) + "\n")
            print(stratum, "usable", len(items), "train", taken[(stratum, "train")], "heldout", taken[(stratum, "heldout")], flush=True)

    counts.tofile(args.out / "groundtruth.counts.i64")
    report = {
        "artifact_type": "freq_corpus",
        "tokenizer": str(args.tokenizer),
        "vocab": vocab,
        "skipped_malformed": skipped,
        "budgets": budgets,
        "usable_conversations": {s: len(by_stratum[s]) for s in STRATA},
        "assistant_tokens": {f"{s}/{split}": n for (s, split), n in sorted(taken.items())},
        "conversations": {f"{s}/{split}": n for (s, split), n in sorted(conversations_taken.items())},
        "by_source": {f"{s}/{split}/{source}": n for (s, split, source), n in sorted(sources.items())},
        "groundtruth_tokens": int(counts.sum()),
        "sources": json.loads((args.raw / "MANIFEST.json").read_text(encoding="utf-8")),
    }
    (args.out / "corpus.manifest.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("vocab", "skipped_malformed", "assistant_tokens", "groundtruth_tokens")}, indent=1))


if __name__ == "__main__":
    main()
