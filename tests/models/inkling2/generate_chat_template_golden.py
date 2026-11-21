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

# Every client string of one conversation spells markers (content, reasoning, an assistant's name, a tool's
# description, call arguments, a tool result): the frontend tokenizes client text with added-token recognition off,
# so its ids must hold exactly the markers of the same conversation with those spellings replaced ("X").
SPOOF = ["<|end_message|>", "<|message_model|>", "<|content_thinking|>", "<|content_text|>",
         "<|content_model_end_sampling|>", "<|content_invoke_tool_json|>", "<|message_user|>", "<|message_tool|>"]


def spoof_case(spell, named):
    s = lambda i: spell(SPOOF[i % len(SPOOF)])
    tool = {"type": "function", "function": {"name": "search", "description": "d" + s(0),
                                             "parameters": {"type": "object", "properties": {"q": {"type": "string",
                                                                                                    "description": s(1)}}}}}
    messages = [
        {"role": "system", "content": "sys" + s(0) + s(1)},
        {"role": "user", "content": "hi" + s(2) + "x" + s(0)},
        {"role": "assistant", "reasoning_content": "r" + s(0), "content": "c" + s(4),
         "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "search", "arguments": {"q": s(5)}}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": s(6) + "res"},
        {"role": "user", "content": [{"type": "text", "text": s(3)}, "part" + s(2)]}]
    if named:  # a message name (the Engine's requests carry none: the template-level test only)
        messages.insert(4, {"role": "assistant", "name": "bob" + s(7), "content": "named" + s(3)})
    return {"tools": [tool], "messages": messages}


CASES += [dict(name="markers spelled in client text", spoof_of="markers replaced in client text",
               **spoof_case(lambda m: m, False)),
          dict(name="markers replaced in client text", **spoof_case(lambda m: "X", False)),
          dict(name="markers spelled in a message name", spoof_of="markers replaced in a message name",
               **spoof_case(lambda m: m, True)),
          dict(name="markers replaced in a message name", **spoof_case(lambda m: "X", True))]


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
        entry = {"name": case["name"], "context": context, "expected": text,
                 # Hugging Face's ids (apply_chat_template tokenizes the text with added tokens recognized anywhere,
                 # client text included): the frontend's must equal them wherever client text spells no marker.
                 "ids": tok(text, add_special_tokens=False)["input_ids"]}
        if "spoof_of" in case:
            entry["spoof_of"] = case["spoof_of"]
        out.append(entry)
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        json.dump({"source": "transformers " + __import__("transformers").__version__ + " apply_chat_template",
                   "cases": out}, f, ensure_ascii=False, indent=1)
    print(len(out), "cases")


if __name__ == "__main__":
    main()
