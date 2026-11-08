"""Qwen's own response parser as a second oracle for the tool-call parser (candidates YH-03).

On 2026-11-07 Qwen's Qwen3.8 repositories (Flash-Next 812d96e10a, 27B 37561816e6, 35B 345d467b11) added a declarative
`response_template` to tokenizer_config.json: how an assistant turn splits into reasoning, tool calls and content.
qwen_response_template.json beside this file is that object, unchanged (identical in the Flash-Next and 35B
repositories). This test interprets it independently and compares the split with the C++ parser's
(`infernix_tool_call_parser_test --probe-parse`, every parameter declared a string so values keep their text) on
the answer part of canonical outputs. Reasoning is split earlier, by the frontend, and is not compared here.

Where the two differ by design, the case names the difference: the template closes a parameter or a call by
lookahead at the next opener, where the strict parser requires the closing tag and returns such text as content.
Content is compared without its surrounding whitespace (the parser drops the blank lines around calls).

    INFERNIX_TOOL_PARSER_PROBE=<infernix_tool_call_parser_test> python test_response_template_oracle.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

SPEC = json.loads((Path(__file__).with_name("qwen_response_template.json")).read_text(encoding="utf-8"))
PROBE = os.environ.get("INFERNIX_TOOL_PARSER_PROBE", "")


def interpret(text: str) -> dict:
    """The answer `text` split by Qwen's response_template: reasoning, tool calls (name, arguments), content."""
    fields = SPEC["fields"]
    think, tools, answer = fields["reasoning_content"], fields["tool_calls"], fields["content"]
    close_think = re.compile(think["close_pattern"])
    open_call, close_call = re.compile(tools["open_pattern"]), re.compile(tools["close_pattern"])
    tag = re.compile(tools["content_args"]["tag_pattern"], re.S)
    reasoning, calls, content = [], [], []
    pos = 0
    while pos < len(text):
        if text.startswith(think["open"], pos):
            start = pos + len(think["open"])
            end = close_think.search(text, start)
            reasoning.append(text[start:end.start() if end else len(text)])
            pos = end.end() if end else len(text)
            continue
        opened = open_call.match(text, pos)
        if opened:
            closed = close_call.search(text, opened.end())
            body = text[opened.end():closed.start() if closed else len(text)]
            calls.append({"name": opened.group("name"),
                          "arguments": {m.group("key"): m.group("value") for m in tag.finditer(body)}})
            pos = closed.end() if closed else len(text)
            continue
        stops = [text.find(think["open"], pos), *(text.find(end, pos) for end in answer["close"])]
        following = open_call.search(text, pos)
        stops.append(following.start() if following else -1)
        stop = min((at for at in stops if at > pos), default=len(text))
        content.append(text[pos:stop])
        pos = stop
        if any(text.startswith(end, pos) for end in answer["close"]):
            break
    return {"reasoning": "".join(reasoning), "calls": calls, "content": answer["join"].join(content)}


def call(name: str, arguments: dict[str, str]) -> str:
    """A call as the model writes it (the chat template's own form)."""
    parameters = "".join(f"<parameter={key}>\n{value}\n</parameter>\n" for key, value in arguments.items())
    return f"<tool_call>\n<function={name}>\n{parameters}</function>\n</tool_call>"


AGREE = {
    "plain content": "The build passed; nothing else to do.",
    "one call": call("bash", {"command": "ls -la"}),
    "content before a call": "Let me look at the directory first.\n\n" + call("bash", {"command": "ls -la"}),
    "two calls": call("read", {"path": "a.txt"}) + "\n" + call("read", {"path": "b.txt"}),
    "multiline value with markup": call("write", {"path": "x.html", "body": "<p>one</p>\n\n<p>three</p>"}),
    "two parameters, one empty": call("edit", {"path": "a.py", "old": ""}),
}
# The template closes by lookahead where the strict parser requires the closing tag: the parser keeps the damaged
# call as content, so it never runs; a complete call after it (the parser retries at the next `<tool_call>`) still
# does. Each case: (text, the parser's content, the parser's calls).
_UNCLOSED_PARAMETER = ("<tool_call>\n<function=edit>\n<parameter=path>\na.py\n<parameter=old>\nx\n</parameter>\n"
                       "</function>\n</tool_call>")
_UNCLOSED_CALL = "<tool_call>\n<function=read>\n<parameter=path>\na.txt\n</parameter>\n</function>"
REFUSED = {
    "parameter closed by the next parameter": (_UNCLOSED_PARAMETER, _UNCLOSED_PARAMETER, []),
    "call closed by the next call": (_UNCLOSED_CALL + "\n" + call("read", {"path": "b.txt"}), _UNCLOSED_CALL,
                                     [{"name": "read", "arguments": {"path": "b.txt"}}]),
}


def probe(requests: list[dict]) -> list[dict]:
    text = "".join(json.dumps(request) + "\n" for request in requests)
    out = subprocess.run([PROBE, "--probe-parse"], input=text, capture_output=True, text=True, encoding="utf-8",
                         check=True).stdout
    return [json.loads(line) for line in out.splitlines()]


def main() -> int:
    if not PROBE:
        print("set INFERNIX_TOOL_PARSER_PROBE to the infernix_tool_call_parser_test executable", file=sys.stderr)
        return 2
    failures = []
    cases = {**AGREE, **{name: text for name, (text, _, _) in REFUSED.items()}}
    expected = {name: interpret(text) for name, text in cases.items()}
    # Every tool the cases call, each declaring every parameter name they use (as strings: values keep their text).
    names = sorted({m for text in cases.values() for m in re.findall(r"<function=([^>\n]+)>", text)})
    keys = sorted({m for text in cases.values() for m in re.findall(r"<parameter=([^>\n]+)>", text)})
    declared = {name: keys for name in names}
    answers = probe([{"text": text, "tools": declared} for text in cases.values()])
    for (name, text), ours in zip(cases.items(), answers):
        if "error" in ours:
            failures.append(f"{name}: the parser probe failed: {ours['error']}")
            continue
        theirs = expected[name]
        if name in REFUSED:
            _, content, calls = REFUSED[name]
            if (ours["calls"] != calls or ours["content"].strip() != content.strip()
                    or ours["tool_call_response"] != bool(calls)):
                failures.append(f"{name}: the parser no longer keeps the damaged call as text: {ours}")
            if len(theirs["calls"]) <= len(calls):
                failures.append(f"{name}: the template no longer runs the damaged call (the case documents nothing)")
            continue
        if ours["calls"] != theirs["calls"]:
            failures.append(f"{name}: calls {ours['calls']} against the template's {theirs['calls']}")
        if ours["content"].strip() != theirs["content"].strip():
            failures.append(f"{name}: content {ours['content']!r} against the template's {theirs['content']!r}")
    for failure in failures:
        print(f"FAIL {failure}", file=sys.stderr)
    print(f"{len(AGREE)} canonical outputs agree with Qwen's response_template, {len(REFUSED)} lookahead closes "
          f"refused by design" if not failures else f"{len(failures)} disagreements")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
