"""Every infernix-serve option, classified for the engines this tool starts.

The rule (and the reason this file exists): a fit is measured on the user's own setup. The engines the tool starts
carry over every production setting that decides *where the model lives* (RAM/VRAM placement, the model, n-gram and
stripe files, the device) or *what it computes* (KV type, activation precision, kernels, the chat template). The tool
owns the settings a fit needs to control (context, concurrency, port, purpose, the sidecar). It drops anything that
only serves production traffic (drafting, sampling defaults, networking, logging) or that would write production
files (the prefix-cache file, the SSD tier, the expert-cache state, log files).

An option this table does not know is refused by name, never passed through: a new flag could change the model or
write a file, and silently forwarding it would make the measurement mean something nobody chose.
tests/test_ablate_tools.py fails when infernix-serve gains an option this table does not classify.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

CARRY = "carry"   # production's value goes to every engine the tool starts
OWNED = "owned"   # the tool sets it; production's value is replaced
DROP = "drop"     # production's value is left out
REFUSE = "refuse" # the tool cannot fit a setup that uses it


@dataclass(frozen=True)
class Rule:
    kind: str
    why: str
    capture: str | None = None   # the capture test's equivalent option, when it has one


_MEMORY = "where the model lives: production's value is known to fit this machine"
_NUMERICS = "changes what the model computes, so the fit must measure with it"
_TRAFFIC = "serves production traffic only"
_WRITES = "would read or write production's files"

RULES: dict[str, Rule] = {
    # MODEL & CONTEXT
    "--max-context": Rule(OWNED, "the tool runs 16384 (raise it for eval/chat with --context)"),
    "--max-concurrency": Rule(OWNED, "the tool runs one sequence"),
    "--prefill-chunk": Rule(CARRY, _MEMORY),
    "--use-original-int8-prefill-kernel": Rule(CARRY, _NUMERICS),
    "--use-original-nvfp4-prefill-kernel": Rule(CARRY, _NUMERICS),
    "--prefill-split-workspace-mib": Rule(CARRY, _MEMORY),
    "--reproducible": Rule(CARRY, _NUMERICS),
    "--lossless": Rule(REFUSE, "infernix-serve refuses --ablate on the lossless route"),
    "--fast-path": Rule(CARRY, _NUMERICS),
    "--no-fast-path": Rule(CARRY, _NUMERICS),
    "--expert-a4-activations": Rule(CARRY, _NUMERICS, capture="--expert-a4-activations"),
    # serve: 16-bit activations for NVIDIA's W4A4 routed experts (Dense8, recipe A, NVFP4-QAD); the capture test's
    # --expert-a16-activations: "a W4A4 checkpoint's banks (NVFP4-QAD) with 16-bit activations"
    "--a16-activations": Rule(CARRY, _NUMERICS, capture="--expert-a16-activations"),
    "--a8-activations": Rule(CARRY, _NUMERICS, capture="--a8-activations"),
    "--embedding-host": Rule(CARRY, _MEMORY),
    "--no-cuda-graph": Rule(CARRY, "decode route"),
    "--default-max-tokens": Rule(DROP, "every request names its own max_tokens"),
    "--default-thinking-budget": Rule(DROP, "every request sets thinking itself"),
    "--no-thinking-answer-room": Rule(DROP, "every request sets thinking itself"),
    "--thinking-budget-message": Rule(DROP, "every request sets thinking itself"),
    "--reasoning-loop": Rule(DROP, _TRAFFIC),
    "--end-in-thinking": Rule(DROP, _TRAFFIC),
    "--model-id": Rule(OWNED, "the tool names its own engine"),
    "--any-model-name": Rule(DROP, _TRAFFIC),
    "--chat-template": Rule(CARRY, "the template the model is served with renders every prompt the fit measures"),
    # Qwen3.5 dense: the adapted model is the one production serves, so the direction and every trial are measured on
    # its activations (--ablate cuts each lane's write with the adapter's term in it).
    "--lora": Rule(CARRY, "the adapter is part of the model production serves"),
    "--claude-code-prompt": Rule(DROP, _TRAFFIC),
    "--ngram-volume": Rule(CARRY, "the model's n-gram embedding volume"),
    "--ngram-residency": Rule(CARRY, _MEMORY),
    "--expert-state": Rule(OWNED, "engines write this file when they stop; the run keeps its own copy, seeded from this one"),
    "--expert-stripe": Rule(CARRY, "part of the model's files"),
    "--context-cost-presets": Rule(DROP, _TRAFFIC),
    "--ablate": Rule(OWNED, "a fit measures the plain model; candidates get their own sidecar"),
    "--rope-yarn-factor": Rule(CARRY, _NUMERICS, capture="--rope-yarn-factor"),
    "--device": Rule(CARRY, "the card the model runs on"),
    "--ram-free-mib": Rule(CARRY, _MEMORY),
    "--ram-resize": Rule(CARRY, _MEMORY),
    "--expert-ram-mib": Rule(CARRY, _MEMORY),
    "--expert-ram-layout": Rule(CARRY, _MEMORY),
    "--expert-pin-mib": Rule(CARRY, _MEMORY),
    "--expert-csf": Rule(CARRY, _MEMORY),
    # KV CACHE
    "--kv-capacity": Rule(OWNED, "sized by the engine from the tool's context and concurrency"),
    "--vram-headroom-mib": Rule(CARRY, _MEMORY),
    "--vram-past-budget": Rule(CARRY, _MEMORY),
    "--expert-tail-skip": Rule(CARRY, _NUMERICS),
    "--kv-dtype": Rule(CARRY, _NUMERICS, capture="--kv"),
    "--logits-dtype": Rule(CARRY, _NUMERICS),
    "--no-prefix-reuse": Rule(CARRY, "how requests reuse the cache"),
    # PREFIX CACHE
    "--host-context-mib": Rule(CARRY, _MEMORY),
    "--prefix-cache-file": Rule(DROP, _WRITES),
    "--prefix-cache-save-mins": Rule(DROP, _WRITES),
    "--ssd-context-gib": Rule(DROP, _WRITES),
    "--ssd-context-dir": Rule(DROP, _WRITES),
    "--ssd-context-gib-per-hour": Rule(DROP, _WRITES),
    "--device-snapshot-slots": Rule(CARRY, _MEMORY),
    "--cache-taps-per-request": Rule(CARRY, _MEMORY),
    "--cache-tap-ladder": Rule(CARRY, _MEMORY),
    "--cache-tap-min-gap": Rule(CARRY, _MEMORY),
    "--no-queue-holds": Rule(DROP, _TRAFFIC),
    "--cache-tool-hold-secs": Rule(DROP, _TRAFFIC),
    "--cache-next-turn-ms": Rule(DROP, _TRAFFIC),
    "--cache-host-state": Rule(CARRY, _MEMORY),
    # SPECULATIVE DECODING: the tool runs --spec off; drafting changes speed, not answers
    "--spec": Rule(OWNED, "the tool runs 'off'"),
    "--draft-tokens": Rule(DROP, _TRAFFIC),
    "--draft-model": Rule(DROP, _TRAFFIC),
    "--lm-head-draft": Rule(DROP, _TRAFFIC),
    "--spec-min-p": Rule(DROP, _TRAFFIC),
    "--draft-tree-nodes": Rule(DROP, _TRAFFIC),
    "--draft-tree-paths": Rule(DROP, _TRAFFIC),
    "--ngram-draft-tokens": Rule(DROP, _TRAFFIC),
    "--ngram-min-match": Rule(DROP, _TRAFFIC),
    "--ngram-tree-nodes": Rule(DROP, _TRAFFIC),
    "--ngram-archive-mib": Rule(DROP, _TRAFFIC),
    "--ngram-session-mib": Rule(DROP, _TRAFFIC),
    "--ngram-native-sessions": Rule(DROP, _TRAFFIC),
    # VISION: the fit is text only
    "--vision": Rule(DROP, "text only"),
    "--vision-offload": Rule(DROP, "text only"),
    "--vision-max-merged": Rule(DROP, "text only"),
    "--media-cache-mib": Rule(DROP, "text only"),
    "--media-live-mib": Rule(DROP, "text only"),
    "--media-preprocess-threads": Rule(DROP, "text only"),
    # SAMPLING: every request sets temperature 0 and enable_thinking itself
    "--temperature": Rule(DROP, "requests are greedy"),
    "--top-p": Rule(DROP, "requests are greedy"),
    "--top-k": Rule(DROP, "requests are greedy"),
    "--min-p": Rule(DROP, "requests are greedy"),
    "--presence-penalty": Rule(DROP, "requests are greedy"),
    "--frequency-penalty": Rule(DROP, "requests are greedy"),
    "--seed": Rule(DROP, "requests are greedy"),
    "--greedy": Rule(DROP, "requests are greedy"),
    "--no-thinking": Rule(DROP, "every request sets thinking itself"),
    "--preserve-thinking": Rule(DROP, _TRAFFIC),
    "--tolerant-tool-calls": Rule(DROP, _TRAFFIC),
    "--tool-parameter-names": Rule(DROP, _TRAFFIC),
    "--restore-reasoning": Rule(DROP, _TRAFFIC),
    # NETWORKING & RESOURCES
    "--host": Rule(OWNED, "the tool listens on 127.0.0.1"),
    "--port": Rule(OWNED, "the tool picks its own port"),
    "--api-key": Rule(DROP, _TRAFFIC),
    "--max-request-mib": Rule(DROP, _TRAFFIC),
    "--max-pending-requests": Rule(DROP, _TRAFFIC),
    "--pending-timeout-ms": Rule(DROP, _TRAFFIC),
    "--keep-alive-timeout-sec": Rule(DROP, _TRAFFIC),
    "--max-http-clients": Rule(DROP, _TRAFFIC),
    "--request-log-jsonl": Rule(DROP, _WRITES),
    "--request-log-max-mib": Rule(DROP, _WRITES),
    "--request-log-keep": Rule(DROP, _WRITES),
    "--route-trace": Rule(DROP, _WRITES),
    "--response-store-max-records": Rule(DROP, _TRAFFIC),
    "--response-store-max-mib": Rule(DROP, _TRAFFIC),
    "--log-stats-interval-ms": Rule(DROP, _TRAFFIC),
    "--log-colours": Rule(DROP, _TRAFFIC),
    "--log-stats-panel": Rule(DROP, _TRAFFIC),
    "--log-level": Rule(DROP, _TRAFFIC),
    "--log-file": Rule(DROP, _WRITES),
    "--cors": Rule(DROP, _TRAFFIC),
    "--usage-chunk-choice": Rule(DROP, _TRAFFIC),
    "--claude-code-usage-compat": Rule(DROP, _TRAFFIC),
    # the tool's own
    "--causal-scoring": Rule(OWNED, "the tool's scoring engine (POST /v1/score)"),
    "--plan": Rule(OWNED, "doctor sizes the tool's engines with it"),
    "--config": Rule(OWNED, "read by production.py, never passed on"),
}


def arities(repo_root: Path) -> dict[str, str]:
    """Flag -> 'switch' | 'value' | 'repeatable', from infernix-serve's own ServeOptionArity table (the parse the
    launcher's coverage test uses). The table is the server's definition of its command line."""
    text = (repo_root / "src" / "serve" / "serve_options.cpp").read_text(encoding="utf-8")
    table: dict[str, str] = {}
    for match in re.finditer(r'ServeOptionArity\{"(--[a-z0-9-]+)"(?:,\s*(true|false))?(?:,\s*(true|false))?\}', text):
        flag, takes_value, repeatable = match.groups()
        table[flag] = "switch" if takes_value == "false" else "repeatable" if repeatable == "true" else "value"
    if len(table) < 100:
        raise RuntimeError(f"{repo_root / 'src/serve/serve_options.cpp'}: the ServeOptionArity table was not found")
    return table


def rule(flag: str) -> Rule:
    found = RULES.get(flag)
    if found is None:
        raise ValueError(f"production uses {flag}, which this tool does not know. It could change the model or write "
                         f"a file, so it is not passed on silently: classify it in tools/ablate/ablatelib/serve_flags.py")
    return found
