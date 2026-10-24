"""Build the IQS-2 certification corpus (eval/corpora/iqs-1; docs/maintainer/model-quality.md, section 6).

Deterministic: fixed sources, fixed slices, fixed item order, no randomness. The main set holds --per-domain
documents (8 by default, at most 16) of each of 8 domains, each scored as one 4,096-token window; the long set
is 4 documents scored as 32,768-token windows. Text documents are tokenized raw by each artifact; chat
documents (JSON) are rendered with each model's own chat template by infernix-certify, so chat, reasoning and
tool-call formatting is the model's own.

The main set grows in layers of 4 documents per domain (32 windows), appended in window order: layer 0 is the
original 32 windows, layer 1 windows 32-63, and so on. A layer never changes the documents before it, so the
references, noise twins and reproducible arms stored for earlier windows stay valid (infernix-certify matches
every stored window to its tokens) and growing the corpus scores only the new windows. Each layer's sources are
listed below; layers 2 and 3 (128 windows) need 8 more German books and more competition mathematics than
MATH-500 holds (layers 0 and 1 use 322 of its 500 problems).

Sources: the repository's existing corpora (eval/corpora/perplexity-1m, decode-4m) and the downloaded public
datasets under --sources (one folder each: gsm8k, math500, oasst2, hermes, gutenberg), read as data only.

Usage: python -I tools/quality/iqs_corpus.py --sources DIR --existing eval/corpora --out eval/corpora/iqs-1
       [--per-domain 8]
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

LAYER = 4  # documents per domain per layer
GUTENBERG = [  # Project Gutenberg ebook id, title, author (public domain in the US); one book per German document
    (22367, "Die Verwandlung", "Franz Kafka"),
    (12108, "Der Tod in Venedig", "Thomas Mann"),
    (2499, "Siddhartha: eine indische Dichtung", "Hermann Hesse"),
    (5323, "Effi Briest", "Theodor Fontane"),
    (24746, "Reise in die Aequinoctial-Gegenden des neuen Continents. Band 2", "Alexander von Humboldt"),
    (35312, "Aus dem Leben eines Taugenichts", "Joseph von Eichendorff"),
    (40739, "Die Traumdeutung", "Sigmund Freud"),
    (7205, "Also sprach Zarathustra", "Friedrich Nietzsche"),
]
# Per layer, the character offset of each document in its stream: WikiText-2 test and Chinese Wikipedia
# (perplexity-1m, one stream per document of a layer), far enough apart to hold different articles.
WIKITEXT_OFFSETS = (0, 140_000, 70_000, 210_000)
ZHWIKI_OFFSETS = (0, 40_000, 20_000, 60_000)
# English long form beyond layer 0: PG-19 books of decode-4m's stream (test and validation books not in
# perplexity-1m; the long set's book is the stream's first), as (book, character offset of its start); each
# document starts 20,000 characters into its book, past the front matter. The last two are second slices.
PG19_BOOKS = [("validation/25066", 340_342), ("validation/5195", 884_989), ("validation/38214", 1_187_515),
              ("test/44557", 1_741_748), ("test/26239", 1_981_984), ("test/55871", 2_328_732),
              ("test/7987", 3_745_909), ("test/4047", 4_775_608), ("test/41603", 5_386_525),
              ("test/29594", 256_002), ("test/55871", 2_928_732), ("test/7987", 4_245_909)]
# Code per layer: (corpus file, character offset); the long set's code is sglang at 2,000,000.
CODE_LAYERS = [
    [("perplexity-1m/data/ninfer/00.txt", 0), ("perplexity-1m/data/ninfer/01.txt", 0),
     ("decode-4m/data/sglang/00.txt", 0), ("decode-4m/data/sglang/00.txt", 1_000_000)],
    [("perplexity-1m/data/ninfer/02.txt", 0), ("perplexity-1m/data/ninfer/03.txt", 0),
     ("decode-4m/data/sglang/00.txt", 3_000_000), ("decode-4m/data/sglang/00.txt", 4_000_000)],
    [("perplexity-1m/data/ninfer/00.txt", 120_000), ("perplexity-1m/data/ninfer/01.txt", 120_000),
     ("decode-4m/data/sglang/00.txt", 500_000), ("decode-4m/data/sglang/00.txt", 1_500_000)],
    [("perplexity-1m/data/ninfer/02.txt", 120_000), ("perplexity-1m/data/ninfer/03.txt", 120_000),
     ("decode-4m/data/sglang/00.txt", 2_500_000), ("decode-4m/data/sglang/00.txt", 3_500_000)],
]
LICENSES = {"ninfer": "Apache-2.0 (NInfer)", "sglang": "Apache-2.0 (SGLang)"}


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


def build(sources: Path, existing: Path, out: Path, per_domain: int = 8) -> dict:
    if per_domain % LAYER or not LAYER <= per_domain <= 4 * LAYER:
        raise ValueError(f"--per-domain must be a multiple of {LAYER} from {LAYER} to {4 * LAYER}")
    if per_domain > len(GUTENBERG):
        raise ValueError(f"{per_domain} German documents need {per_domain} books; GUTENBERG lists {len(GUTENBERG)}")
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

    sglang = existing_text("decode-4m/data/sglang/00.txt")
    pg19 = existing_text("decode-4m/data/pg19/00.txt")
    gsm, math = gsm8k_items(sources), math500_items(sources)
    threads = oasst_threads(sources)
    calling = json.loads((sources / "hermes" / "func-calling.json").read_text(encoding="utf-8"))
    json_mode = json.loads((sources / "hermes" / "json-mode-agentic.json").read_text(encoding="utf-8"))
    # Item cursors run on across layers, so a layer continues where the one before it stopped.
    cursor_gsm = cursor_math = cursor_threads = cursor_calling = cursor_json = 0
    for layer in range(per_domain // LAYER):
        first = layer * LAYER
        for n in range(LAYER):
            stream = f"perplexity-1m/data/wikitext/0{n}.txt"
            offset = WIKITEXT_OFFSETS[layer]
            add(main_docs, f"en-ref-{first + n}", "english_reference", "text",
                slice_text(existing_text(stream), offset, MAIN_CHARS["default"]),
                {"source": f"eval/corpora/{stream}", **({"offset": offset} if offset else {}),
                 "license": "CC-BY-SA-4.0 (WikiText)"})
        for n in range(LAYER):
            if layer == 0:
                stream = f"perplexity-1m/data/pg19/0{n}.txt"
                add(main_docs, f"en-long-{n}", "english_long_form", "text",
                    slice_text(existing_text(stream), 0, MAIN_CHARS["default"]),
                    {"source": f"eval/corpora/{stream}", "license": "Apache-2.0 (PG-19)"})
            else:
                book, start = PG19_BOOKS[first - LAYER + n]
                add(main_docs, f"en-long-{first + n}", "english_long_form", "text",
                    slice_text(pg19, start + 20_000, MAIN_CHARS["default"]),
                    {"source": "eval/corpora/decode-4m/data/pg19/00.txt", "book": f"PG-19 {book}",
                     "offset": start + 20_000, "license": "Apache-2.0 (PG-19)"})
        for n in range(LAYER):
            stream = f"perplexity-1m/data/zhwiki/0{n}.txt"
            offset = ZHWIKI_OFFSETS[layer]
            add(main_docs, f"zh-{first + n}", "chinese", "text",
                slice_text(existing_text(stream), offset, MAIN_CHARS["chinese"]),
                {"source": f"eval/corpora/{stream}", **({"offset": offset} if offset else {}),
                 "license": "CC-BY-SA-3.0 (Wikipedia)"})
        for n in range(LAYER):
            book, title, author = GUTENBERG[first + n]
            body = gutenberg_body((sources / "gutenberg" / f"pg{book}.txt").read_text(encoding="utf-8"))
            add(main_docs, f"de-{first + n}", "german", "text", slice_text(body, 2000, MAIN_CHARS["default"]),
                {"source": f"https://www.gutenberg.org/ebooks/{book}", "title": title, "author": author,
                 "license": "Public domain in the US (Project Gutenberg)"})
        for n, (name, offset) in enumerate(CODE_LAYERS[layer]):
            text = sglang if "sglang" in name else existing_text(name)
            add(main_docs, f"code-{first + n}", "code", "text", slice_text(text, offset, MAIN_CHARS["default"]),
                {"source": f"eval/corpora/{name}", "offset": offset,
                 "license": LICENSES["sglang" if "sglang" in name else "ninfer"]})
        for n in range(LAYER):
            if n < 2:
                text, end = math_text(gsm, cursor_gsm, MAIN_CHARS["default"])
                src = {"source": "openai/gsm8k test", "items": [cursor_gsm, end], "license": "MIT"}
                cursor_gsm = end
            else:
                text, end = math_text(math, cursor_math, MAIN_CHARS["default"])
                src = {"source": "HuggingFaceH4/MATH-500 (OpenAI prm800k)", "items": [cursor_math, end], "license": "MIT"}
                cursor_math = end
            add(main_docs, f"math-{first + n}", "mathematics", "text", text, src)
        for n in range(2):
            chat, end = oasst_chat(threads, cursor_threads, MAIN_CHARS["default"])
            add(main_docs, f"chat-{first + n}", "chat", "chat", chat,
                {"source": "OpenAssistant/oasst2 validation (English threads)", "threads": [cursor_threads, end],
                 "license": "Apache-2.0"})
            cursor_threads = end
        chat, end = math_chat(gsm, cursor_gsm, MAIN_CHARS["default"])
        add(main_docs, f"chat-{first + 2}", "chat", "chat", chat,
            {"source": "openai/gsm8k test as a reasoning chat", "items": [cursor_gsm, end], "license": "MIT"})
        cursor_gsm = end
        chat, end = math_chat(math, cursor_math, MAIN_CHARS["default"])
        add(main_docs, f"chat-{first + 3}", "chat", "chat", chat,
            {"source": "HuggingFaceH4/MATH-500 as a reasoning chat", "items": [cursor_math, end], "license": "MIT"})
        cursor_math = end
        for n in range(2):
            chat, end = hermes_tool_chat(calling, cursor_calling, MAIN_CHARS["default"])
            add(main_docs, f"tools-{first + n}", "tool_calls", "chat", chat,
                {"source": "NousResearch/hermes-function-calling-v1 func-calling.json", "items": [cursor_calling, end],
                 "license": "Apache-2.0"})
            cursor_calling = end
        for n in range(2):
            chat, end = hermes_json_chat(json_mode, cursor_json, MAIN_CHARS["default"])
            add(main_docs, f"tools-{first + n + 2}", "tool_calls", "chat", chat,
                {"source": "NousResearch/hermes-function-calling-v1 json-mode-agentic.json", "items": [cursor_json, end],
                 "license": "Apache-2.0"})
            cursor_json = end
    if cursor_gsm > len(gsm) - 250:
        raise ValueError(f"the corpus reaches GSM8K item {cursor_gsm}, into the task audit's last 250 items")

    long_sources = [("long-en-book", "english_long_form", "decode-4m/data/pg19/00.txt", 0, "default", "Apache-2.0 (PG-19)"),
                    ("long-en-ref", "english_reference", "decode-4m/data/wikitext/00.txt", 0, "default", "CC-BY-SA-4.0 (WikiText)"),
                    ("long-zh", "chinese", "decode-4m/data/zhwiki/00.txt", 0, "chinese", "CC-BY-SA-3.0 (Wikipedia)"),
                    ("long-code", "code", "decode-4m/data/sglang/00.txt", 2_000_000, "default", "Apache-2.0 (SGLang)")]
    for doc_id, domain, name, offset, kind_chars, license_ in long_sources:
        text = sglang if "sglang" in name else existing_text(name)
        add(long_docs, doc_id, domain, "text", slice_text(text, offset, LONG_CHARS[kind_chars]),
            {"source": f"eval/corpora/{name}", "offset": offset, "license": license_, "set": "long"})

    # The task flip audit's fixed items (a paired accuracy-loss check, iqs_e2e.py tasks): GSM8K test
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
        "version": 2,  # 2: the main set in layers of 32 windows (version 1 was layer 0 alone)
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
    parser.add_argument("--per-domain", type=int, default=8, help="main-set documents per domain (4, 8, 12 or 16)")
    args = parser.parse_args()
    manifest = build(args.sources, args.existing, args.out, args.per_domain)
    for name, s in manifest["sets"].items():
        print(f"{name}: {len(s['documents'])} documents, {s['window_tokens']}-token windows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
