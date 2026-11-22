"""The engines the tool starts: infernix-serve with production's carried options plus the tool's own.

One process serves one purpose (infernix-serve: generation, or --causal-scoring), so a trial that needs refusals and
KL starts two. Every engine gets:
  production's carried options (serve_flags CARRY: placement, files, numerics, the chat template), then
  --host 127.0.0.1 --port <free> --model-id ablate --max-context <16384> --max-concurrency 1 --spec off
  --expert-state <run>/expert.state [--causal-scoring] [--ablate <sidecar>].
The run keeps its own expert-cache state: engines write it when they stop, so production's file is never written,
and each boot starts with a warm cache (a cold one costs a fit about 40 s a trial). The first engine seeds it from
production's file when there is one.
"""

from __future__ import annotations

import json
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import platform_util as PU
from . import serve_flags as SF

MODEL_ID = "ablate"


class EngineError(RuntimeError):
    """One engine's problem (died booting, timed out, a request failed): a trial records it and the fit goes on."""


class Engine:
    def __init__(self, run, purpose: str, sidecar: Path | None = None, log_path: Path | None = None,
                 context: int | None = None, boot_timeout_s: float = 900.0):
        if purpose not in ("generate", "score"):
            raise ValueError(purpose)
        self.run, self.purpose, self.sidecar = run, purpose, sidecar
        self.context = int(context or run.context)
        self.log_path = log_path or run.sub("logs") / f"engine-{purpose}.log"
        self.boot_timeout_s = boot_timeout_s
        self.port = 0
        self.proc = None

    def argv(self, port: int | None = None) -> list[str]:
        prod = self.run.production
        cmd = [str(self.run.serve_binary), str(prod.artifact)]
        for flag, value in prod.by_kind(SF.CARRY):
            cmd += [flag] if value is None else [flag, value]
        cmd += ["--host", "127.0.0.1", "--port", str(port or self.port or 0), "--model-id", MODEL_ID,
                "--max-context", str(self.context), "--max-concurrency", "1", "--spec", "off",
                "--expert-state", str(self.expert_state)]
        if self.purpose == "score":
            cmd += ["--causal-scoring"]
        if self.sidecar is not None:
            cmd += ["--ablate", str(self.sidecar)]
        return cmd

    @property
    def expert_state(self) -> Path:
        return self.run.dir / "expert.state"

    def seed_expert_state(self) -> None:
        if self.expert_state.exists():
            return
        named = self.run.production.value("--expert-state")
        source = None if named == "off" else Path(named) if named else Path(str(self.run.production.artifact) + ".expert-state")
        if source is not None and source.is_file():
            shutil.copy2(source, self.expert_state)

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> Engine:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def start(self) -> None:
        self.stop()
        self.seed_expert_state()
        self.port = PU.free_port()
        self.proc = PU.start(self.argv(), self.log_path)
        deadline = time.monotonic() + self.boot_timeout_s
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise EngineError(f"infernix-serve exited during start (rc {self.proc.returncode}); last lines of "
                                  f"{self.log_path}:\n{self.tail()}")
            try:
                with urllib.request.urlopen(self.base + "/health", timeout=2):
                    return
            except (urllib.error.URLError, OSError):
                time.sleep(1.0)
        raise EngineError(f"infernix-serve did not answer /health within {self.boot_timeout_s:.0f} s; see {self.log_path}")

    def stop(self) -> None:
        PU.stop(self.proc)
        self.proc = None

    def tail(self, n: int = 15) -> str:
        try:
            return "".join(self.log_path.read_text(encoding="utf-8", errors="replace").splitlines(True)[-n:])
        except OSError:
            return ""

    def post(self, route: str, body: dict, timeout: float = 600.0) -> dict:
        req = urllib.request.Request(self.base + route, data=json.dumps(body).encode(),
                                     headers={"content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as reply:
                return json.load(reply)
        except urllib.error.HTTPError as e:
            raise EngineError(f"{route}: HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:400]}") from e
        except (urllib.error.URLError, OSError) as e:
            raise EngineError(f"{route}: {type(e).__name__}: {e}") from e

    def chat(self, prompt: str, *, max_tokens: int, enable_thinking: bool, timeout: float = 600.0) -> dict:
        """One fresh user turn, greedy. Returns the first choice. (Infernix: `model` must equal --model-id, the reply
        carries reasoning_content separately, and there is no usage object.)"""
        reply = self.post("/v1/chat/completions", {"model": MODEL_ID, "messages": [{"role": "user", "content": prompt}],
                                                   "temperature": 0.0, "max_tokens": max_tokens,
                                                   "enable_thinking": enable_thinking}, timeout)
        return reply["choices"][0]

    def score(self, prompt: str, continuation: str, *, top_k: int, enable_thinking: bool) -> list[dict[str, float]]:
        """POST /v1/score (a --causal-scoring engine): the top-k next-token distribution at each continuation position,
        with the chat template applied. Refuses anything that would quietly change what KL means."""
        out = self.post("/v1/score", {"model": MODEL_ID, "apply_chat_template": True,
                                      "messages": [{"role": "user", "content": prompt}],
                                      "enable_thinking": enable_thinking, "continuation": continuation, "top_k": top_k})
        if out.get("object") != "score":
            raise EngineError(f"/v1/score returned object={out.get('object')!r}")
        if out.get("chat_template_applied") is not True:
            raise EngineError("/v1/score did not apply the chat template")
        if out.get("enable_thinking") is not enable_thinking:
            raise EngineError(f"/v1/score rendered enable_thinking={out.get('enable_thinking')!r}, asked {enable_thinking}")
        positions = out.get("positions") or []
        if not positions or len(positions[0].get("top_logprobs") or []) != top_k:
            raise EngineError(f"/v1/score returned {len(positions)} position(s) / a top_logprobs width other than {top_k}")
        return [{t["token"]: float(t["logprob"]) for t in p["top_logprobs"]} for p in positions]
