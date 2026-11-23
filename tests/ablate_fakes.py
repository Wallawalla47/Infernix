"""A stand-in for infernix-serve, for tests/test_ablate_tools.py: no GPU, no model, any OS.

    python tests/ablate_fakes.py serve <infernix-serve's argv after the binary>

The fake engine answers like the real one where the tool depends on it (/health, /v1/chat/completions,
/v1/score in src/serve/score.cpp's shape with its hidden states, --plan, --expert-state written at stop). A sidecar's
strength is the sum of its alphas: refusals of prompts containing HARM fall, and the first-token distribution moves,
as it grows, so a fit has a real front to find. Its hidden states plant a known refusal direction in harmful prompts
(layer L: planted(L)), so a test can check the math.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

HARM = "HARM"                  # the tests' harmful prompts contain this word
EDGE = "EDGE"                  # a benign prompt the stock model refuses anyway (an over-refusal)
SHAPE_ENV = "ABLATE_FAKE_SHAPE"       # "layers,hidden" of the fake model's hidden states
PLANT = 3.0
MEAN = 5.0
NOISE = 0.05


def fake_ids(prompt: str) -> list[int]:
    """The fake tokenizer."""
    return [1 if HARM in prompt else 2] + [10 + ord(c) % 200 for c in prompt[:20]]


def hidden_states(prompt: str) -> list[list[float]]:
    """The fake model's [layers, hidden] hidden states at a prompt's last token: noise and the shared mean common(L),
    plus planted(L) at every layer of a harmful prompt, BF16-rounded as the engine's are."""
    layers, hidden = (int(x) for x in os.environ[SHAPE_ENV].split(","))
    ids = fake_ids(prompt)
    x = np.random.default_rng(sum(ids) * 7919 + len(ids)).normal(scale=NOISE, size=(layers, hidden)).astype(np.float32)
    for layer in range(layers):
        x[layer] += MEAN * common(layer, hidden)
        if HARM in prompt:
            x[layer] += PLANT * planted(layer, hidden)
    return ((x.view(np.uint32) >> 16) << 16).view(np.float32).tolist()


def planted(layer: int, hidden: int) -> np.ndarray:
    v = np.random.default_rng(1000 + layer).normal(size=hidden)
    return (v / np.linalg.norm(v)).astype(np.float32)


def common(layer: int, hidden: int) -> np.ndarray:
    """Every prompt's shared mean (real hidden states have a large one), orthogonal to planted(layer): what heretic's
    orthogonalization projects out of the direction."""
    v = np.random.default_rng(2000 + layer).normal(size=hidden)
    p = planted(layer, hidden).astype(np.float64)
    v = v - (v @ p) * p
    return (v / np.linalg.norm(v)).astype(np.float32)


def _seed(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def _arg(argv: list[str], flag: str) -> str | None:
    return argv[argv.index(flag) + 1] if flag in argv else None


def strength(sidecar: str | None) -> float:
    if not sidecar:
        return 0.0
    cfg = json.loads(Path(sidecar).read_text(encoding="utf-8"))
    return float(sum(cfg.get("alpha_attn_per_layer", {}).values()) + sum(cfg.get("alpha_mlp_per_layer", {}).values()))


def serve(argv: list[str]) -> int:
    if "--plan" in argv:
        Path(_arg(argv, "--plan") or "plan.json").write_text("{}\n", encoding="utf-8")
        return 0
    port, s, scoring = int(_arg(argv, "--port") or 0), strength(_arg(argv, "--ablate")), "--causal-scoring" in argv
    state = _arg(argv, "--expert-state")

    def stop(*_):
        if state and state != "off":
            Path(state).write_text("fake expert-cache state\n", encoding="utf-8")   # the real engine writes at stop
        os._exit(0)

    for name in ("SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), stop)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # quiet: the tool keeps the engine log, not request lines
            pass

        def reply(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.reply(200, {"status": "ok"}) if self.path == "/health" else self.reply(404, {})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            if body.get("model") != "ablate":
                return self.reply(404, {"error": {"code": "model_not_found"}})
            prompt = body["messages"][-1]["content"]
            if self.path == "/v1/chat/completions":
                if scoring:
                    return self.reply(400, {"error": {"code": "generation_not_enabled"}})
                refuse = (HARM in prompt or EDGE in prompt) and _seed(prompt) % 100 >= 4 * s
                content = "I cannot help with that." if refuse else "Sure, here is what you asked for."
                return self.reply(200, {"choices": [{"message": {"role": "assistant", "content": content,
                                                                 "reasoning_content": ""}, "finish_reason": "stop"}]})
            if self.path == "/v1/score":
                if not scoring:
                    return self.reply(400, {"error": {"code": "scoring_not_enabled"}})
                k, words = int(body.get("top_k") or 0), max(1, len(body["continuation"].split()))
                phase = _seed(prompt) % 1000 / 100.0
                logits = [-0.3 * i + 0.05 * s * math.sin(1.7 * i + phase) for i in range(max(k, 1))]
                top = max(logits)
                norm = top + math.log(sum(math.exp(x - top) for x in logits))
                alts = [{"token_id": i, "token": f"t{i}", "logprob": x - norm} for i, x in enumerate(logits)]
                alts.sort(key=lambda a: -a["logprob"])
                positions = [{"token_id": alts[0]["token_id"], "token": alts[0]["token"],
                              "logprob": alts[0]["logprob"], "top_logprobs": alts[:k]} for _ in range(words)]
                reply = {"object": "score", "model": "ablate", "chat_template_applied": True,
                         "enable_thinking": body.get("enable_thinking"), "positions": positions,
                         "usage": {"prompt_tokens": len(fake_ids(prompt)), "continuation_tokens": words}}
                if body.get("hidden_states"):
                    reply["hidden_states"] = hidden_states(prompt)
                return self.reply(200, reply)
            return self.reply(404, {})

    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit({"serve": serve}[sys.argv[1]](sys.argv[2:]))
