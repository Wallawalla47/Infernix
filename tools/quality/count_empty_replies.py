"""Count replies that end empty in `--request-log-jsonl` files (candidates LE-04).

Qwen3.8 reports (Qwen3.8 issue #216, Flash-Next HF discussion #50) describe turns that stop with no
visible content: `finish_reason: stop` after an empty answer, or a stop inside the reasoning block that
never closed. The request log keeps each request's generated token ids, so this script classifies every
`request_done` record without the text:

  tool_call           the reply holds at least one structured tool call
  empty_content       a stop (stop_token or stop_string), no tool call, and nothing but whitespace (or
                      end-of-turn tokens) after the last reasoning close
  unclosed_reasoning  thinking was on, a stop, no tool call, and no reasoning close at all
  content             any other stop with visible content
  output_limit        the output budget ended the reply
  other               every other finish reason (context_capacity, cancelled)

Rates are grouped by model, sampling temperature, requested reasoning effort and request label, with
the counts, so a group of one request is not mistaken for a rate. Rotated files (FILE.1 ..) are read in
the order given.

  py -3.11 -I tools/quality/count_empty_replies.py LOG [LOG ...]
      [--tokenizer-json tokenizer.json] [--json]

`--tokenizer-json` (an artifact's resource/text/tokenizer.json) decodes the tokens after the close, so a
reply of only newlines counts as empty; without it only an end-of-turn token or nothing counts.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

STOP_REASONS = ("stop_token", "stop_string")
THINK_CLOSE = "</think>"
END_OF_TURN = ("<|im_end|>", "<|endoftext|>")
TOOL_CALL_OPEN = "<tool_call>"
DEFAULT_IDS = {"</think>": 248069, "<|im_end|>": 248046, "<|endoftext|>": 248044, "<tool_call>": 248058}


def byte_decoder() -> dict[str, int]:
    """The inverse of the byte-level BPE alphabet (GPT-2's bytes_to_unicode)."""
    printable = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    code = printable[:]
    extra = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            code.append(256 + extra)
            extra += 1
    return {chr(c): b for c, b in zip(code, printable)}


class Vocabulary:
    def __init__(self, path: Path | None) -> None:
        self.ids = dict(DEFAULT_IDS)
        self.tokens: dict[int, str] = {}
        self.decoder = byte_decoder()
        if path is None:
            return
        document = json.loads(path.read_text(encoding="utf-8"))
        for added in document.get("added_tokens", ()):
            if added["content"] in self.ids:
                self.ids[added["content"]] = added["id"]
        self.tokens = {identifier: token for token, identifier in document["model"]["vocab"].items()}

    def text(self, ids: list[int]) -> str:
        data = bytearray()
        for identifier in ids:
            token = self.tokens.get(identifier)
            if token is None:
                return "?"
            data.extend(self.decoder.get(char, ord("?")) for char in token)
        return data.decode("utf-8", errors="replace")


def classify(record: dict, vocabulary: Vocabulary) -> str:
    result = record["result"]
    finish = result.get("finish_reason")
    if result.get("tool_call_count", 0) > 0:
        return "tool_call"
    if finish == "output_limit":
        return "output_limit"
    if finish not in STOP_REASONS:
        return "other"
    tokens = result.get("generated_token_ids") or []
    close = vocabulary.ids[THINK_CLOSE]
    ends = {vocabulary.ids[name] for name in END_OF_TURN}
    thinking = bool(record["request"].get("enable_thinking"))
    if close in tokens:
        tail = tokens[len(tokens) - tokens[::-1].index(close):]
    elif thinking:
        return "unclosed_reasoning"
    else:
        tail = tokens
    tail = [identifier for identifier in tail if identifier not in ends]
    if not tail:
        return "empty_content"
    if vocabulary.tokens and not vocabulary.text(tail).strip():
        return "empty_content"
    return "content"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--tokenizer-json", type=Path, default=None)
    parser.add_argument("--json", action="store_true", help="print the groups as JSON")
    args = parser.parse_args()

    vocabulary = Vocabulary(args.tokenizer_json)
    groups: dict[tuple, Counter] = defaultdict(Counter)
    total = Counter()
    for path in args.logs:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("event") != "request_done":
                    continue
                request = record["request"]
                key = (
                    request.get("model"),
                    request["sampling"]["temperature"],
                    request.get("requested_reasoning_effort"),
                    request.get("label"),
                )
                category = classify(record, vocabulary)
                groups[key][category] += 1
                total[category] += 1

    rows = []
    for key in sorted(groups, key=str):
        counts = groups[key]
        requests = sum(counts.values())
        rows.append(
            {
                "model": key[0],
                "temperature": key[1],
                "reasoning_effort": key[2],
                "label": key[3],
                "requests": requests,
                "counts": dict(counts),
                "empty": counts["empty_content"] + counts["unclosed_reasoning"],
            }
        )
    if args.json:
        json.dump({"groups": rows, "total": dict(total)}, sys.stdout, indent=1)
        print()
        return 0
    print(f"{'model':44} {'temp':>5} {'effort':>8} {'label':>18} {'requests':>9} {'empty':>6}  counts")
    for row in rows:
        print(
            f"{str(row['model'])[:44]:44} {row['temperature']:>5} {str(row['reasoning_effort']):>8} "
            f"{str(row['label'])[:18]:>18} {row['requests']:>9} {row['empty']:>6}  {row['counts']}"
        )
    requests = sum(total.values())
    empty = total["empty_content"] + total["unclosed_reasoning"]
    print(f"\n{requests} requests, {empty} ended empty ({100.0 * empty / requests:.1f} %)" if requests else "no request_done records")
    return 0


if __name__ == "__main__":
    sys.exit(main())
