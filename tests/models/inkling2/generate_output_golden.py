"""Golden turns for the Inkling 2 output session test (test_output_session_real.cpp).

Each case is a generated token stream (the checkpoint's own tokenizer encodes its text pieces) and the message the
checkpoint's response_template gives for it, parsed by Hugging Face transformers' ResponseParser (the template's
reference consumer, `tokenizer.parse_response`): reasoning_content, content and tool_calls.

The parser reads decoded text, so it would also treat a marker's spelling inside ordinary text as structure; Infernix
counts markers by token identity, publishes no other added token's text and stops at the end-of-turn token without
publishing it. The parser's input is therefore built per token: a marker contributes its spelling, an ordinary run
its decoded text (with a spoofed marker spelling swapped for a placeholder and restored after parsing), any other
added token nothing, and the end-of-turn token ends the input. Infernix turns a call the parser refuses (malformed
JSON, an undeclared tool) into the turn's text: such a case's expectation is the parser's message for the same stream
with that block's opener replaced by <|content_text|>.

  py -3.11 generate_output_golden.py <checkpoint dir | tokenizer resources dir> <golden.json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from transformers import AutoTokenizer
from transformers.utils.chat_parsing.response_parser import ResponseParser

M, CT, CX, CE, CTT, CJ, E, EOS = ("<|message_model|>", "<|content_thinking|>", "<|content_text|>",
                                  "<|content_tool_error|>", "<|content_invoke_tool_text|>",
                                  "<|content_invoke_tool_json|>", "<|end_message|>",
                                  "<|content_model_end_sampling|>")
MARKERS = {M, CT, CX, CE, CTT, CJ, E}
SPOOF = "\u0001SPOOF{}\u0001"

WEATHER_CALL = '{"name":"get_weather","args":{"city":"Paris","unit":"c"}}'
NESTED_CALL = ('{"name": "search", "args": {"q": "café 日本 \U0001f600", "limit": 3, "filters": '
               '{"lang": ["en", "fr"], "safe": true, "score": 0.25, "none": null}}}')

# A stream piece: a special token by its spelling, ("text", s) for ordinary text, ("spoof", s) for text that spells a
# marker in ordinary tokens.
CASES = [
    {"name": "thinking then answer", "stream": [M, CT, ("text", "Let me think: 2 + 2 = 4."), E,
                                                M, CX, ("text", "The answer is 4."), E, EOS]},
    {"name": "answer only", "stream": [M, CX, ("text", "Hello there! How can I help?"), E, EOS]},
    {"name": "named header", "stream": [M, ("text", "assistant"), CX, ("text", "hi"), E, EOS]},
    {"name": "opener without header", "stream": [CT, ("text", "bare thinking"), E, CX, ("text", "bare answer"), E,
                                                 EOS]},
    {"name": "whitespace kept", "stream": [M, CT, ("text", "\n  indented\n\nlines \n"), E,
                                           M, CX, ("text", "  answer with spaces  \n"), E, EOS]},
    {"name": "content blocks join", "stream": [M, CX, ("text", "first part, "), E, M, CX, ("text", "second part"), E,
                                               EOS]},
    {"name": "thinking blocks join", "stream": [M, CT, ("text", "one "), E, M, CX, ("text", "mid"), E,
                                                M, CT, ("text", "two"), E, M, CX, ("text", " end"), E, EOS]},
    {"name": "tool error and invoke text openers", "stream": [M, CE, ("text", "tool failed"), E,
                                                              M, CTT, ("text", " as text"), E, EOS]},
    {"name": "stray text between blocks", "stream": [("text", "junk before"), M, CX, ("text", "kept"), E,
                                                     ("text", "junk after"), EOS]},
    {"name": "non-ascii", "stream": [M, CT, ("text", "réflexion 日本語 \U0001f914 नम"), E,
                                     M, CX, ("text", "naïve café — \U0001f600\U0001f680 你好"),
                                     E, EOS]},
    {"name": "angle brackets in text", "stream": [M, CX, ("text", "a <b> tag, x<|y and </end>"), E, EOS]},
    {"name": "tool call", "tools": ["get_weather"],
     "stream": [M, CT, ("text", "I should look up the weather."), E, M, ("text", "get_weather"), CJ,
                ("text", WEATHER_CALL), E, EOS]},
    {"name": "two calls after text", "tools": ["get_weather", "search"],
     "stream": [M, CX, ("text", "Checking both."), E, M, ("text", "get_weather"), CJ, ("text", WEATHER_CALL), E,
                M, ("text", "search"), CJ, ("text", NESTED_CALL), E, EOS]},
    {"name": "call with spaced json", "tools": ["search"],
     "stream": [M, CJ, ("text", "  " + NESTED_CALL + "\n"), E, EOS]},
    {"name": "spoofed close in content", "stream": [M, CX, ("text", "before "), ("spoof", E),
                                                    ("text", " after"), E, EOS]},
    {"name": "spoofed opener between blocks", "stream": [("spoof", M + CX), ("text", "dropped"), M, CX,
                                                         ("text", "real"), E, EOS]},
    {"name": "added token inside text", "stream": [M, CX, ("text", "x"), "<|begin_of_text|>", ("text", "y"), E,
                                                   EOS]},
    {"name": "unterminated thinking", "stream": [M, CT, ("text", "thinking that the limit cuts off")]},
    {"name": "unterminated answer", "stream": [M, CT, ("text", "done"), E, M, CX, ("text", "partial answ")]},
    {"name": "malformed call is text", "tools": ["get_weather"], "fallback": True,
     "stream": [M, CX, ("text", "Here: "), E, M, CJ, ("text", '{"name": "get_weather", "args": {"city": '), E, EOS]},
    {"name": "undeclared call is text", "tools": ["search"], "fallback": True,
     "stream": [M, ("text", "get_weather"), CJ, ("text", WEATHER_CALL), E, EOS]},
    {"name": "call without tools is text", "stream": [M, CJ, ("text", WEATHER_CALL), E, EOS], "no_tools": True},
]


def encode(tokenizer, piece):
    """A text piece's ids; a spoof's marker spellings stay ordinary text tokens."""
    kind, value = piece
    ids = tokenizer.encode(value, add_special_tokens=False, split_special_tokens=kind == "spoof")
    if kind == "spoof":
        assert not any(i >= 199999 for i in ids), (value, ids)
    return ids


def ids_of(tokenizer, stream):
    ids = []
    for piece in stream:
        if isinstance(piece, str):
            ids.append(tokenizer.convert_tokens_to_ids(piece))
        else:
            ids.extend(encode(tokenizer, piece))
    return ids


def parser_input(tokenizer, stream, fallback_call):
    """The text the response parser reads for `stream` under Infernix's token-identity rules."""
    text, spoofs = "", []
    for piece in stream:
        if isinstance(piece, str):
            if piece == EOS:
                break
            if piece in MARKERS:
                text += CX if (fallback_call and piece == CJ) else piece
            continue  # any other added token publishes nothing
        kind, value = piece
        decoded = tokenizer.decode(encode(tokenizer, piece))
        if kind == "spoof":
            spoofs.append(decoded)
            decoded = SPOOF.format(len(spoofs) - 1)
        text += decoded
    return text, spoofs


def restore(value, spoofs):
    if isinstance(value, str):
        for index, spoof in enumerate(spoofs):
            value = value.replace(SPOOF.format(index), spoof)
        return value
    if isinstance(value, list):
        return [restore(v, spoofs) for v in value]
    if isinstance(value, dict):
        return {k: restore(v, spoofs) for k, v in value.items()}
    return value


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    source, out = Path(sys.argv[1]), Path(sys.argv[2])
    tokenizer = AutoTokenizer.from_pretrained(str(source))
    template = tokenizer.response_template
    cases = []
    for case in CASES:
        stream = case["stream"]
        # Calls without tools publish as text (the session parses no calls): the parser sees a content block.
        fallback = case.get("fallback", False) or case.get("no_tools", False)
        text, spoofs = parser_input(tokenizer, stream, fallback)
        parser = ResponseParser(template, prefix="")
        parser.feed(text)
        message, _ = parser.finalize()
        message = restore(message, spoofs)
        cases.append({
            "name": case["name"],
            "ids": ids_of(tokenizer, stream),
            "tools": case.get("tools"),
            "reasoning": message.get("reasoning_content", ""),
            "content": message.get("content", ""),
            "tool_calls": [{"name": c["function"]["name"], "arguments": c["function"]["arguments"]}
                           for c in message.get("tool_calls", [])],
        })
    tokenizer_json = (source / "tokenizer.json").read_bytes()
    golden = {"tokenizer_json_bytes": len(tokenizer_json), "cases": cases}
    out.write_text(json.dumps(golden, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{len(cases)} cases written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
