"""Writes tests/fixtures/inkling2/chat_template_golden.json: Inkling 2's chat_template.jinja rendered by Hugging Face
transformers (Jinja2) for a set of conversations, the oracle of infernix_inkling2_chat_template_test (Infernix's Jinja
engine must give the same text). Usage: python generate_chat_template_golden.py <checkpoint or template dir> <out.json>"""

import json
import sys

from transformers import AutoTokenizer

WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.",
                                            "parameters": {"type": "object", "properties": {
                                                "city": {"type": "string"}, "unit": {"enum": ["c", "f"]}},
                                                "required": ["city"]}}}
SEARCH = {"name": "search", "description": "", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}

CASES = [
    {"name": "user only", "messages": [{"role": "user", "content": "Hello!"}]},
    {"name": "system and user", "messages": [{"role": "system", "content": "You are terse."},
                                             {"role": "user", "content": "What is 2+2?"}]},
    {"name": "effort string", "messages": [{"role": "user", "content": "Plan a trip."}], "reasoning_effort": "low"},
    {"name": "effort none", "messages": [{"role": "user", "content": "Hi"}], "reasoning_effort": "none"},
    {"name": "effort number", "messages": [{"role": "user", "content": "Hi"}], "reasoning_effort": 0.35},
    {"name": "effort zero number", "messages": [{"role": "user", "content": "Hi"}], "reasoning_effort": 0.0},
    {"name": "multi turn with thinking", "messages": [
        {"role": "user", "content": "Capital of France?"},
        {"role": "assistant", "reasoning_content": "Easy: Paris.", "content": "Paris."},
        {"role": "user", "content": "And of Italy?"}]},
    {"name": "assistant empty content with thinking", "messages": [
        {"role": "user", "content": "x"}, {"role": "assistant", "reasoning_content": "hmm", "content": ""},
        {"role": "user", "content": "y"}]},
    {"name": "content parts", "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Part one."}, "Part two.", {"text": "Part three."}]}]},
    {"name": "tools and a call", "tools": [WEATHER, SEARCH], "messages": [
        {"role": "user", "content": "Weather in Paris, in C?"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {
            "name": "get_weather", "arguments": {"unit": "c", "city": "Paris", "Zeta": [1, 2], "alpha": {"b": 1, "a": "é"}}}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp\": 18}"},
        {"role": "assistant", "content": "It is 18 °C."},
        {"role": "user", "content": "Thanks"}]},
    {"name": "tool result named", "tools": [SEARCH], "messages": [
        {"role": "user", "content": "find cats"},
        {"role": "assistant", "content": "Searching.", "tool_calls": [{"function": {"name": "search", "arguments": {"q": "cats"}}}]},
        {"role": "tool", "name": "search", "content": "many cats"}]},
    {"name": "named assistant", "messages": [{"role": "user", "content": "hi"},
                                             {"role": "assistant", "name": "helper", "content": "hello"}]},
    {"name": "system only", "messages": [{"role": "system", "content": "Only a system message."}]},
    {"name": "unicode and specials in text", "messages": [
        {"role": "user", "content": "Ünïcödé 北京 🌏 <tag> \\n {{ not a template }} {% raw %}"}]},
]


def main():
    tok = AutoTokenizer.from_pretrained(sys.argv[1])
    out = []
    for case in CASES:
        kwargs = {}
        if "reasoning_effort" in case:
            kwargs["reasoning_effort"] = case["reasoning_effort"]
        text = tok.apply_chat_template(case["messages"], tools=case.get("tools"), tokenize=False, **kwargs)
        context = {"messages": case["messages"]}
        if "tools" in case:
            context["tools"] = case["tools"]
        if "reasoning_effort" in case:
            context["reasoning_effort"] = case["reasoning_effort"]
        out.append({"name": case["name"], "context": context, "expected": text})
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        json.dump({"source": "transformers " + __import__("transformers").__version__ + " apply_chat_template",
                   "cases": out}, f, ensure_ascii=False, indent=1)
    print(len(out), "cases")


if __name__ == "__main__":
    main()
