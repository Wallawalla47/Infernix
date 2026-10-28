"""Writes a long-prompt --messages file for the KV dump (candidates §23.2 PB-01, step 1).

  text   one user message: the files' text, each under a `=== path ===` header, then a question;
  agent  an agent trace: a system prompt and a task, then for each file an assistant read_file call
         and its tool result (the file's text), ending on the last tool result, so the prompt ends
         where the assistant's next turn begins, as an agentic client's request does.

Files are taken in the order given (directories walked in sorted order, keeping source and text
files) until at least --chars characters are in; about 4 characters a token for code and English
text, so the default 160,000 gives a prompt of roughly 35-40K tokens (the dump's JSON records the
exact count).

  python -I tools/kv_lossless_count/make_messages.py text|agent OUT.json PATH... [--chars N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SUFFIXES = {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".py", ".md", ".txt", ".json", ".cmake", ".rs", ".go", ".js", ".ts"}


def files(paths: list[Path]) -> list[Path]:
    out = []
    for path in paths:
        if path.is_dir():
            out.extend(p for p in sorted(path.rglob("*")) if p.is_file() and p.suffix.lower() in SUFFIXES)
        else:
            out.append(path)
    return out


def take(paths: list[Path], chars: int) -> list[tuple[str, str]]:
    taken, total = [], 0
    for path in files(paths):
        text = path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
        if not text.strip():
            continue
        taken.append((path.as_posix(), text))
        total += len(text)
        if total >= chars:
            return taken
    raise SystemExit(f"the inputs hold only {total} characters, fewer than --chars {chars}")


def text_messages(taken: list[tuple[str, str]]) -> list[dict]:
    body = "".join(f"=== {path} ===\n{text}\n\n" for path, text in taken)
    question = "Summarize what the files above do, in three sentences."
    return [{"role": "user", "content": body + question}]


def agent_messages(taken: list[tuple[str, str]]) -> dict:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file of the repository.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string", "description": "Repository-relative path."}},
                    "required": ["path"],
                },
            },
        }
    ]
    messages: list[dict] = [
        {"role": "system", "content": "You are a coding agent working in a local repository. Use the tools to read code before you answer."},
        {"role": "user", "content": "Read the files that implement this feature and explain how they fit together, then propose one improvement."},
    ]
    for index, (path, text) in enumerate(taken):
        call_id = f"call_{index}"
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "read_file", "arguments": {"path": path}}}],
            }
        )
        messages.append({"role": "tool", "tool_call_id": call_id, "content": text})
    return {"tools": tools, "messages": messages}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kind", choices=("text", "agent"))
    parser.add_argument("out", type=Path)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--chars", type=int, default=160_000)
    args = parser.parse_args()
    taken = take(args.paths, args.chars)
    document = text_messages(taken) if args.kind == "text" else agent_messages(taken)
    args.out.write_text(json.dumps(document, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{args.out}: {len(taken)} files, {sum(len(text) for _, text in taken)} characters")
    return 0


if __name__ == "__main__":
    sys.exit(main())
