"""Capture: run every direction prompt into the model once and record its internal state.

For each prompt (rendered with production's chat template, or the artifact's own, one user turn, thinking off, as
heretic renders it) Ian's forward-pass test (`infernix_qwen4_exp_forward_real_test --blocks`) writes, at every
layer, the attention lane's input and output and the MoE lane's input and output at every token: [layers][4][T][H]
BF16. The direction step reads them; nothing else does.

Facts that shape this module (measured on QAD Dense8, an RTX 5090, 2026-11-21):
  * each prompt runs in its own harness, so prompts are independent and recordings are kept per prompt, named by a
    fingerprint of everything that decides their bytes (token ids, model, KV type, activation options, n-gram
    volume): adding prompts records only those, and an interrupted capture resumes;
  * the test takes none of serve's memory options. By default it pins every expert in host RAM (64.5 GiB + 5 GiB of
    VRAM for QAD Dense8; ~1 s per prompt); --stream-experts reads each layer's experts from disk before it runs
    (~4 GiB of RAM, ~12 s and ~64 GiB of reads per prompt). The two give byte-identical recordings. The mode is
    chosen from free RAM unless --capture-mode says;
  * the test needs one --dump-logits file per prompt to run several prompts on one load; --dump-from keeps only the
    last position, so those files are ~1 MB each instead of ~30.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

from . import platform_util as PU
from . import serve_flags as SF

TOKENIZER_OBJECT = "resource/text/tokenizer.json"
TEMPLATE_OBJECT = "resource/text/chat_template.jinja"
CHUNK = 512                 # one prefill chunk per prompt; longer prompts are refused
TAPS = 4                    # mixer in, mixer out, moe in, moe out
SHAPE = (48, 2560)          # Qwen3.8-Flash-Next: layers, hidden size (the one model family this tool covers)


def _artifact_reader():
    import sys
    from .run import REPO_ROOT
    tools = str(REPO_ROOT / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    from artifact.reader import Artifact   # noqa: E402  (Infernix's own reader, tools/artifact)
    return Artifact


def artifact_object(artifact: Path, object_id: str) -> bytes:
    Artifact = _artifact_reader()
    with Artifact.open(artifact) as a:
        obj = a.object(object_id)
        blob = a.read_object(object_id)
    if len(blob) != obj.bytes:
        raise SystemExit(f"{artifact}: {object_id} truncated ({len(blob)} of {obj.bytes} bytes)")
    return blob


def template_source(run) -> tuple[str, str]:
    """(template text, where it came from): production's --chat-template, else the artifact's own."""
    path = run.production.value("--chat-template")
    if path:
        return Path(path).read_text(encoding="utf-8"), path
    return artifact_object(run.production.artifact, TEMPLATE_OBJECT).decode("utf-8"), f"artifact {TEMPLATE_OBJECT}"


def render(template: str, prompt: str) -> str:
    """One user turn, add_generation_prompt, thinking off, with Hugging Face's Jinja settings (trim_blocks,
    lstrip_blocks, loopcontrols). Without them a template renders a leading newline: 29 tokens where the engine
    counts 28 (measured 2026-11-19)."""
    from jinja2.ext import loopcontrols
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise ValueError(message)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: time.strftime(fmt)
    return env.from_string(template).render(messages=[{"role": "user", "content": prompt}],
                                            add_generation_prompt=True, enable_thinking=False)


class Tokenizer:
    """The artifact's own tokenizer (the engine reads resource/text/tokenizer.json): the only authoritative one."""

    def __init__(self, artifact: Path):
        from tokenizers import Tokenizer as HFTokenizer
        self.tok = HFTokenizer.from_str(artifact_object(artifact, TOKENIZER_OBJECT).decode("utf-8"))

    def encode(self, text: str) -> list[int]:
        return list(self.tok.encode(text, add_special_tokens=False).ids)

    def round_trips(self, text: str) -> bool:
        return self.tok.decode(self.encode(text), skip_special_tokens=False) == text


def capture_options(run) -> tuple[list[str], dict[str, str], list[str]]:
    """(test options, environment, production options the test cannot reproduce)."""
    prod = run.production
    opts = ["--kv", prod.value("--kv-dtype") or "bf16"]
    env = {"INFERNIX_QWEN4_ARTIFACT": str(prod.artifact)}
    if prod.value("--ngram-volume"):
        env["INFERNIX_QWEN4_NGRAM"] = str(prod.value("--ngram-volume"))
    if prod.value("--device"):
        env["CUDA_VISIBLE_DEVICES"] = str(prod.value("--device"))
    not_reproduced = []
    for flag, value in prod.by_kind(SF.CARRY):
        rule = SF.rule(flag)
        if flag in ("--kv-dtype", "--ngram-volume", "--device", "--chat-template"):
            continue
        if rule.capture and rule.capture != "--kv":
            opts += [rule.capture] if value is None else [rule.capture, value]
        elif rule.why.startswith("changes what the model computes"):
            not_reproduced.append(flag if value is None else f"{flag} {value}")
    return opts, env, not_reproduced


def shard_bytes(artifact: Path) -> int:
    m = re.match(r"(.*)-(\d+)-of-(\d+)(\.\w+)$", artifact.name)
    if not m:
        return artifact.stat().st_size
    stem, _, total, ext = m.groups()
    return sum((artifact.parent / f"{stem}-{i:0{len(total)}d}-of-{total}{ext}").stat().st_size
               for i in range(1, int(total) + 1))


def choose_mode(run, requested: str) -> tuple[str, str]:
    """('default' | 'stream', why). Default mode pins the experts: allow it when free RAM covers the whole artifact
    plus 8 GiB (QAD Dense8 measured 65.2 GiB of 71.3 on disk, so this errs safe)."""
    if requested in ("default", "stream"):
        return requested, "chosen with --capture-mode"
    need = shard_bytes(run.production.artifact) + 8 * PU.GIB
    free = PU.available_ram_bytes()
    if free >= need:
        return "default", f"{free / PU.GIB:.1f} GiB free covers the artifact ({need / PU.GIB:.1f} GiB with headroom)"
    return "stream", (f"{free / PU.GIB:.1f} GiB free is less than the artifact needs pinned "
                      f"({need / PU.GIB:.1f} GiB with headroom): experts are read from disk per prompt")


def fingerprint(ids: list[int], identity: dict) -> str:
    return hashlib.sha256(json.dumps([ids, identity], sort_keys=True).encode()).hexdigest()[:20]


def expected_bytes(n_layers: int, hidden: int, n_tokens: int) -> int:
    return n_layers * TAPS * n_tokens * hidden * 2


def tokenize(run, prompts: list[str]) -> tuple[list[list[int]], dict]:
    template, template_from = template_source(run)
    tok = Tokenizer(run.production.artifact)
    texts = [render(template, p) for p in prompts]
    bad = [p[:60] for p, t in zip(prompts, texts) if not tok.round_trips(t)]
    if bad:
        raise SystemExit(f"{len(bad)} prompt(s) do not survive the tokenizer round trip (e.g. {bad[0]!r}): the token "
                         "ids would not describe these prompts")
    ids = [tok.encode(t) for t in texts]
    too_long = [i for i, x in enumerate(ids) if len(x) > CHUNK]
    if too_long:
        raise SystemExit(f"{len(too_long)} prompt(s) render to more than {CHUNK} tokens (first: #{too_long[0]}); "
                         "shorten or drop them")
    info = {"template_from": template_from, "template_sha256": hashlib.sha256(template.encode()).hexdigest(),
            "render": "one user turn, add_generation_prompt, enable_thinking=False, HF Jinja trim/lstrip"}
    return ids, info


def run_capture(run, kinds: dict[str, list[str]], mode_request: str = "auto", log=print) -> dict:
    """Record every prompt not yet recorded. kinds: {'harmful': [...], 'harmless': [...]}. Returns the index."""
    n_layers, hidden = SHAPE
    if run.capture_binary is None or not Path(run.capture_binary).is_file():
        raise SystemExit("no capture binary: build Infernix with -DBUILD_TESTING=ON and pass --capture-binary "
                         "<build>/tests/infernix_qwen4_exp_forward_real_test to `init`")
    out = run.sub("capture")
    opts, env, not_reproduced = capture_options(run)
    identity = {"artifact": run.production.artifact.name, "artifact_bytes": shard_bytes(run.production.artifact),
                "ngram": Path(env.get("INFERNIX_QWEN4_NGRAM", "default")).name, "test_options": opts}
    index: dict = {"identity": identity, "shape": {"layers": n_layers, "hidden": hidden},
                   "not_reproduced": not_reproduced, "prompts": {}}
    plan: list[tuple[str, int, str, list[int]]] = []
    for kind, prompts in kinds.items():
        ids_list, render_info = tokenize(run, prompts)
        index["render"] = render_info
        for i, ids in enumerate(ids_list):
            fp = fingerprint(ids, {**identity, "render": render_info["template_sha256"]})
            index["prompts"][f"{kind}:{i}"] = {"fingerprint": fp, "n_tokens": len(ids)}
            target = out / f"{fp}.blocks.bin"
            if not (target.is_file() and target.stat().st_size == expected_bytes(n_layers, hidden, len(ids))):
                plan.append((kind, i, fp, ids))
    if not_reproduced:
        log(f"  note: the capture test cannot reproduce {', '.join(not_reproduced)}; recorded in capture/index.json")
    if not plan:
        log("  every prompt is already recorded")
        _write_index(out, index)
        return index
    mode, why = choose_mode(run, mode_request)
    log(f"  {len(plan)} prompt(s) to record in {mode} mode ({why})")
    if mode == "stream":
        log(f"  expect about 12 s per prompt on a ~5 GB/s disk (~{len(plan) * 12 / 60:.0f} min here at that speed)")
    batch = 40            # one model load per batch; a crash loses at most one batch
    for start in range(0, len(plan), batch):
        part = plan[start:start + batch]
        argv = [str(run.capture_binary)] + (["--stream-experts"] if mode == "stream" else [])
        for kind, i, fp, ids in part:
            ids_file = out / f"{fp}.ids"
            ids_file.write_text(",".join(map(str, ids)) + "\n", encoding="ascii")
            argv += [f"@{ids_file}", *opts, "--dump-logits", str(out / f"{fp}.logits.bin"),
                     "--dump-from", str(len(ids) - 1), "--chunk", str(CHUNK), "--blocks", str(out / f"{fp}.blocks.bin")]
        log_path = run.sub("logs") / f"capture-{time.strftime('%Y%m%dT%H%M%S')}-{start // batch}.log"
        t0 = time.monotonic()
        proc = PU.start(argv, log_path, env=dict(os.environ, **env))
        rc = proc.wait()
        PU.stop(proc)
        missing = [fp for _, _, fp, ids in part
                   if not (out / f"{fp}.blocks.bin").is_file()
                   or (out / f"{fp}.blocks.bin").stat().st_size != expected_bytes(n_layers, hidden, len(ids))]
        if rc != 0 or missing:
            raise SystemExit(f"capture failed (rc {rc}, {len(missing)} of {len(part)} recordings missing); see "
                             f"{log_path}. Recorded prompts are kept: rerun `capture` to continue.")
        log(f"  recorded {start + len(part)}/{len(plan)} ({time.monotonic() - t0:.0f} s for this batch)")
    index["mode"], index["mode_why"] = mode, why
    index["capture_binary"] = str(run.capture_binary)
    _write_index(out, index)
    return index


def _write_index(out: Path, index: dict) -> None:
    from .run import write_json
    write_json(out / "index.json", index)


def check_against_engine(run, kinds: dict[str, list[str]], n: int = 2, log=print) -> None:
    """Positive control: for the first n prompts of each kind, the engine's own prompt token count (/v1/score,
    production's carried options and template) must equal the ids the capture recorded. A mismatch means this
    tool renders or tokenizes differently from Infernix, so the direction would describe other tokens."""
    from .engine import Engine
    prompts = [p for kind in kinds.values() for p in kind[:n]]
    ids_list, _ = tokenize(run, prompts)
    problems = []
    with Engine(run, "score", log_path=run.sub("logs") / "capture-check.score.log") as engine:
        for p, ids in zip(prompts, ids_list):
            out = engine.post("/v1/score", {"model": "ablate", "apply_chat_template": True, "enable_thinking": False,
                                            "messages": [{"role": "user", "content": p}], "continuation": " x",
                                            "top_k": 1})
            got = (out.get("usage") or {}).get("prompt_tokens")
            if got is None or int(got) != len(ids):
                problems.append(f"{p[:40]!r}: capture {len(ids)} tokens, engine {got}")
    if problems:
        raise SystemExit("the capture's token ids do not match what the engine renders:\n  " + "\n  ".join(problems)
                         + "\n(the recordings are kept; check --chat-template and the artifact)")
    log(f"  positive control: the engine counts the same tokens as the capture on {len(prompts)} prompt(s)")
