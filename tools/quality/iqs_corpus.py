"""Build the IQS-1 certification corpus (eval/corpora/iqs-1; docs/maintainer/model-quality.md, section 6).

Deterministic: fixed sources, fixed slices, fixed item order, no randomness. The main set is 32 documents of
8 domains (4 each), each scored as one 4,096-token window; the long set is 4 documents scored as 32,768-token
windows. Text documents are tokenized raw by each artifact; chat documents (JSON) are rendered with each
model's own chat template by infernix-certify, so chat, reasoning and tool-call formatting is the model's own.

Sources: the repository's existing corpora (eval/corpora/perplexity-1m, decode-4m) and the downloaded public
datasets under --sources (one folder each: gsm8k, math500, oasst2, hermes, gutenberg), read as data only.

Usage: python -I tools/quality/iqs_corpus.py --sources DIR --existing eval/corpora --out eval/corpora/iqs-1
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

MAIN_WINDOW, LONG_WINDOW = 4096, 32768
MAIN_STARTS = [256, 1024, 2048, 3840]
LONG_STARTS = [8192, 16384, 24576, 32640]
DECODE_TOKENS = 64
# Characters per document: several times what a window needs at the densest tokenization seen (about
# 1.5 characters per token for Chinese, 3 for code, 4 for English), so no artifact's tokenizer runs short.
MAIN_CHARS = {"default": 40000, "chinese": 16000}
LONG_CHARS = {"default": 240000, "chinese": 110000}

GUTENBERG = [  # Project Gutenberg ebook id, title, author (public domain in the US)
    (22367, "Die Verwandlung", "Franz Kafka"),
    (12108, "Der Tod in Venedig", "Thomas Mann"),
    (2499, "Siddhartha: eine indische Dichtung", "Hermann Hesse"),
    (5323, "Effi Briest", "Theodor Fontane"),
]


def at_line(text: str, offset: int) -> int:
    """The first line start at or after offset."""
    if offset <= 0:
        return 0
    found = text.find("\n", offset)
    return len(text) if found < 0 else found + 1


def slice_text(text: str, start: int, chars: int) -> str:
    begin = at_line(text, start)
    end = at_line(text, begin + chars)
    return text[begin:end]


def gutenberg_body(text: str) -> str:
    start = re.search(r"\*\*\* START OF (THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\n", text)
    end = re.search(r"\*\*\* END OF (THE|THIS) PROJECT GUTENBERG EBOOK", text)
    if not start or not end:
        raise ValueError("no Project Gutenberg start/end markers")
    return text[start.end():end.start()]


def read_parquet(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def gsm8k_items(sources: Path) -> list[tuple[str, str, str]]:
    """(problem, worked solution without calculator annotations, final answer)."""
    out = []
    for row in read_parquet(sources / "gsm8k" / "test-00000-of-00001.parquet"):
        solution, _, answer = row["answer"].partition("####")
        out.append((row["question"].strip(), re.sub(r"<<[^>]*>>", "", solution).strip(), answer.strip()))
    return out


def math500_items(sources: Path) -> list[tuple[str, str, str]]:
    out = []
    for line in (sources / "math500" / "test.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            out.append((item["problem"].strip(), item["solution"].strip(), str(item["answer"]).strip()))
    return out


def math_text(items: list[tuple[str, str, str]], first: int, chars: int) -> tuple[str, int]:
    parts, i, size = [], first, 0
    while size < chars:
        problem, solution, answer = items[i]
        block = f"Problem: {problem}\nSolution: {solution}\nAnswer: {answer}\n\n"
        parts.append(block)
        size += len(block)
        i += 1
    return "".join(parts), i


def math_chat(items: list[tuple[str, str, str]], first: int, chars: int) -> tuple[dict, int]:
    """A tutoring chat whose assistant turns reason through each problem (reasoning kept in every turn)."""
    messages, i, size = [], first, 0
    while size < chars:
        problem, solution, answer = items[i]
        messages.append({"role": "user", "content": problem})
        messages.append({"role": "assistant", "reasoning": solution, "content": f"The answer is {answer}."})
        size += len(problem) + len(solution) + len(answer)
        i += 1
    return {"enable_thinking": True, "preserve_thinking": True, "messages": messages}, i


def oasst_threads(sources: Path) -> list[list[dict]]:
    """English conversation threads: from each root prompt, the best-ranked reply at every step."""
    rows = [r for r in read_parquet(sources / "oasst2" / "validation.parquet") if not r.get("deleted")]
    children: dict[str, list[dict]] = {}
    for row in rows:
        children.setdefault(row["parent_id"], []).append(row)
    threads = []
    roots = sorted((r for r in rows if r["parent_id"] is None and r["lang"] == "en"), key=lambda r: r["message_id"])
    for root in roots:
        thread, node = [], root
        while node is not None:
            thread.append({"role": "user" if node["role"] == "prompter" else "assistant", "content": node["text"].strip()})
            replies = sorted(children.get(node["message_id"], []),
                             key=lambda r: (r["rank"] if r["rank"] is not None else 1e9, r["message_id"]))
            node = replies[0] if replies else None
        while thread and thread[-1]["role"] != "assistant":
            thread.pop()
        if len(thread) >= 2:
            threads.append(thread)
    return threads


def oasst_chat(threads: list[list[dict]], first: int, chars: int) -> tuple[dict, int]:
    messages, i, size = [], first, 0
    while size < chars:
        messages.extend(threads[i])
        size += sum(len(m["content"]) for m in threads[i])
        i += 1
    return {"enable_thinking": False, "messages": messages}, i


def hermes_tool_chat(conversations: list[dict], first: int, chars: int) -> tuple[dict, int]:
    """Function-calling sessions joined into one agentic chat: the union of their tools, each <tool_call> a
    structured tool call and each <tool_response> a tool message."""
    tools: dict[str, dict] = {}
    messages, i, size = [], first, 0
    while size < chars:
        conv = conversations[i]
        i += 1
        for tool in json.loads(conv["tools"]) if isinstance(conv.get("tools"), str) else conv.get("tools", []):
            tools.setdefault(tool["function"]["name"], tool)
        for turn in conv["conversations"]:
            text = turn["value"]
            size += len(text)
            if turn["from"] == "system":
                continue
            if turn["from"] == "human":
                messages.append({"role": "user", "content": text.strip()})
            elif turn["from"] == "gpt":
                calls = []
                for body in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, flags=re.S):
                    call = json.loads(body.replace("'", '"')) if body.lstrip().startswith("{'") else json.loads(body)
                    calls.append({"name": call["name"], "arguments": call.get("arguments", {})})
                content = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.S).strip()
                message = {"role": "assistant", "content": content}
                if calls:
                    message["tool_calls"] = calls
                messages.append(message)
            elif turn["from"] == "tool":
                for body in re.findall(r"<tool_response>\s*(.*?)\s*</tool_response>", text, flags=re.S) or [text]:
                    messages.append({"role": "tool", "content": body.strip()})
    while messages and messages[-1]["role"] != "assistant":
        messages.pop()
    return {"enable_thinking": False, "tools": list(tools.values()), "messages": messages}, i


def hermes_json_chat(conversations: list[dict], first: int, chars: int) -> tuple[dict, int]:
    """JSON-mode sessions joined into one chat: each request carries its own schema."""
    messages = [{"role": "system", "content": "You answer in JSON that follows the schema given with each request."}]
    i, size = first, 0
    while size < chars:
        conv = conversations[i]
        i += 1
        schema = ""
        for turn in conv["conversations"]:
            text = turn["value"].strip()
            size += len(text)
            if turn["from"] == "system":
                found = re.search(r"<schema>\s*(.*?)\s*</schema>", text, flags=re.S)
                schema = found.group(1) if found else text
            elif turn["from"] == "human":
                messages.append({"role": "user", "content": (f"Schema:\n{schema}\n\n" if schema else "") + text})
                schema = ""
            elif turn["from"] == "gpt":
                messages.append({"role": "assistant", "content": text})
    while messages and messages[-1]["role"] != "assistant":
        messages.pop()
    return {"enable_thinking": False, "messages": messages}, i


def build(sources: Path, existing: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    data = out / "data"
    provenance: list[dict] = []
    main_docs: list[dict] = []
    long_docs: list[dict] = []

    def add(collection: list[dict], doc_id: str, domain: str, kind: str, content: str | dict, source: dict) -> None:
        relative = Path(domain) / (doc_id + (".json" if kind == "chat" else ".txt"))
        path = data / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(content, ensure_ascii=False, indent=1) if kind == "chat" else content
        path.write_text(text, encoding="utf-8", newline="\n")
        collection.append({"id": doc_id, "domain": domain, "kind": kind, "path": f"data/{relative.as_posix()}"})
        provenance.append({"id": doc_id, "domain": domain, "kind": kind, "chars": len(text), **source})

    def existing_text(name: str) -> str:
        return (existing / name).read_text(encoding="utf-8")

    for n in range(4):
        add(main_docs, f"en-ref-{n}", "english_reference", "text",
            slice_text(existing_text(f"perplexity-1m/data/wikitext/0{n}.txt"), 0, MAIN_CHARS["default"]),
            {"source": f"eval/corpora/perplexity-1m/data/wikitext/0{n}.txt", "license": "CC-BY-SA-4.0 (WikiText)"})
    for n in range(4):
        add(main_docs, f"en-long-{n}", "english_long_form", "text",
            slice_text(existing_text(f"perplexity-1m/data/pg19/0{n}.txt"), 0, MAIN_CHARS["default"]),
            {"source": f"eval/corpora/perplexity-1m/data/pg19/0{n}.txt", "license": "Apache-2.0 (PG-19)"})
    for n in range(4):
        add(main_docs, f"zh-{n}", "chinese", "text",
            slice_text(existing_text(f"perplexity-1m/data/zhwiki/0{n}.txt"), 0, MAIN_CHARS["chinese"]),
            {"source": f"eval/corpora/perplexity-1m/data/zhwiki/0{n}.txt", "license": "CC-BY-SA-3.0 (Wikipedia)"})
    for n, (book, title, author) in enumerate(GUTENBERG):
        body = gutenberg_body((sources / "gutenberg" / f"pg{book}.txt").read_text(encoding="utf-8"))
        add(main_docs, f"de-{n}", "german", "text", slice_text(body, 2000, MAIN_CHARS["default"]),
            {"source": f"https://www.gutenberg.org/ebooks/{book}", "title": title, "author": author,
             "license": "Public domain in the US (Project Gutenberg)"})
    sglang = existing_text("decode-4m/data/sglang/00.txt")
    code_sources = [("perplexity-1m/data/ninfer/00.txt", 0, "Apache-2.0 (NInfer)"),
                    ("perplexity-1m/data/ninfer/01.txt", 0, "Apache-2.0 (NInfer)"),
                    ("decode-4m/data/sglang/00.txt", 0, "Apache-2.0 (SGLang)"),
                    ("decode-4m/data/sglang/00.txt", 1_000_000, "Apache-2.0 (SGLang)")]
    for n, (name, offset, license_) in enumerate(code_sources):
        text = sglang if "sglang" in name else existing_text(name)
        add(main_docs, f"code-{n}", "code", "text", slice_text(text, offset, MAIN_CHARS["default"]),
            {"source": f"eval/corpora/{name}", "offset": offset, "license": license_})
    gsm, math = gsm8k_items(sources), math500_items(sources)
    cursor_gsm = cursor_math = 0
    for n in range(4):
        if n < 2:
            text, end = math_text(gsm, cursor_gsm, MAIN_CHARS["default"])
            src = {"source": "openai/gsm8k test", "items": [cursor_gsm, end], "license": "MIT"}
            cursor_gsm = end
        else:
            text, end = math_text(math, cursor_math, MAIN_CHARS["default"])
            src = {"source": "HuggingFaceH4/MATH-500 (OpenAI prm800k)", "items": [cursor_math, end], "license": "MIT"}
            cursor_math = end
        add(main_docs, f"math-{n}", "mathematics", "text", text, src)
    threads = oasst_threads(sources)
    cursor = 0
    for n in range(2):
        chat, end = oasst_chat(threads, cursor, MAIN_CHARS["default"])
        add(main_docs, f"chat-{n}", "chat", "chat", chat,
            {"source": "OpenAssistant/oasst2 validation (English threads)", "threads": [cursor, end], "license": "Apache-2.0"})
        cursor = end
    for n, (items, start, name) in enumerate(((gsm, cursor_gsm, "openai/gsm8k test"),
                                               (math, cursor_math, "HuggingFaceH4/MATH-500"))):
        chat, end = math_chat(items, start, MAIN_CHARS["default"])
        add(main_docs, f"chat-{n + 2}", "chat", "chat", chat,
            {"source": f"{name} as a reasoning chat", "items": [start, end], "license": "MIT"})
    calling = json.loads((sources / "hermes" / "func-calling.json").read_text(encoding="utf-8"))
    json_mode = json.loads((sources / "hermes" / "json-mode-agentic.json").read_text(encoding="utf-8"))
    cursor = 0
    for n in range(2):
        chat, end = hermes_tool_chat(calling, cursor, MAIN_CHARS["default"])
        add(main_docs, f"tools-{n}", "tool_calls", "chat", chat,
            {"source": "NousResearch/hermes-function-calling-v1 func-calling.json", "items": [cursor, end],
             "license": "Apache-2.0"})
        cursor = end
    cursor = 0
    for n in range(2):
        chat, end = hermes_json_chat(json_mode, cursor, MAIN_CHARS["default"])
        add(main_docs, f"tools-{n + 2}", "tool_calls", "chat", chat,
            {"source": "NousResearch/hermes-function-calling-v1 json-mode-agentic.json", "items": [cursor, end],
             "license": "Apache-2.0"})
        cursor = end

    long_sources = [("long-en-book", "english_long_form", "decode-4m/data/pg19/00.txt", 0, "default", "Apache-2.0 (PG-19)"),
                    ("long-en-ref", "english_reference", "decode-4m/data/wikitext/00.txt", 0, "default", "CC-BY-SA-4.0 (WikiText)"),
                    ("long-zh", "chinese", "decode-4m/data/zhwiki/00.txt", 0, "chinese", "CC-BY-SA-3.0 (Wikipedia)"),
                    ("long-code", "code", "decode-4m/data/sglang/00.txt", 2_000_000, "default", "Apache-2.0 (SGLang)")]
    for doc_id, domain, name, offset, kind_chars, license_ in long_sources:
        text = sglang if "sglang" in name else existing_text(name)
        add(long_docs, doc_id, domain, "text", slice_text(text, offset, LONG_CHARS[kind_chars]),
            {"source": f"eval/corpora/{name}", "offset": offset, "license": license_, "set": "long"})

    # The task flip audit's fixed items (reported, gated only below 99 % of the reproducible route): GSM8K test
    # items from the end of the split (the corpus uses its start), and MMLU test questions spread over subjects.
    tasks = out / "tasks"
    tasks.mkdir(exist_ok=True)
    gsm_tail = gsm[-250:]
    (tasks / "gsm8k-250.jsonl").write_text(
        "".join(json.dumps({"question": q, "answer": a}, ensure_ascii=False) + "\n" for q, _, a in gsm_tail),
        encoding="utf-8", newline="\n")
    mmlu_rows = read_parquet(sources / "mmlu" / "test-00000-of-00001.parquet")
    by_subject: dict[str, list[dict]] = {}
    for row in mmlu_rows:
        by_subject.setdefault(row["subject"], []).append(row)
    picked: list[dict] = []
    while len(picked) < 500:
        progressed = False
        for subject in sorted(by_subject):
            rows = by_subject[subject]
            index = sum(1 for p in picked if p["subject"] == subject)
            if index < len(rows) and len(picked) < 500:
                row = rows[index]
                picked.append({"subject": subject, "question": row["question"], "choices": list(row["choices"]),
                               "answer": "ABCD"[int(row["answer"])]})
                progressed = True
        if not progressed:
            break
    (tasks / "mmlu-500.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in picked),
                                          encoding="utf-8", newline="\n")
    provenance.append({"id": "tasks/gsm8k-250", "source": "openai/gsm8k test, last 250 items", "license": "MIT"})
    provenance.append({"id": "tasks/mmlu-500", "source": "cais/mmlu all/test, round-robin over subjects",
                       "license": "MIT"})

    manifest = {
        "corpus_id": "iqs-1",
        "version": 1,
        "sets": {
            "main": {"window_tokens": MAIN_WINDOW, "decode_tokens": DECODE_TOKENS, "decode_starts": MAIN_STARTS,
                     "documents": main_docs},
            "long": {"window_tokens": LONG_WINDOW, "decode_tokens": DECODE_TOKENS, "decode_starts": LONG_STARTS,
                     "documents": long_docs},
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8", newline="\n")
    (out / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=1) + "\n",
                                         encoding="utf-8", newline="\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--existing", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args.sources, args.existing, args.out)
    for name, s in manifest["sets"].items():
        print(f"{name}: {len(s['documents'])} documents, {s['window_tokens']}-token windows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
