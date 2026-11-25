"""The launcher against the stand-in server (no GPU): its features and its watchdog.

python -I tools/launcher/launcher_check.py --launcher <infernix-launcher> --fake <infernix_launcher_fake_serve>
    --models <dir> [--groups features,watchdog,windowless,autoload,cutoff,ablation]
Runs on Windows and Linux. <dir> holds at least one supported artifact (only its header is read;
nothing in it is changed). Each group starts its own launcher with its memory and logs in a fresh
temporary folder (printed; removed afterwards unless --keep).

features: model ids, named configurations, environment, the memory breakdown, script export and
          import, the fit plan, keep-alive streams while a model loads (OpenAI and Anthropic, a
          failed load, and the server's error answer after the load), model
          routes and the pin, prompt progress, request labels, usage, API keys and sign-in, chats,
          a configuration's idle unload and LoRA adapter folders (about four minutes).
watchdog: crash, lock-up, stall, the restart limit, idle servers, loads that do not fit, the RAM
          floor (about 6 GiB of RAM for a few seconds; --skip-ram leaves it out), the event log and
          the request-content log (about three minutes).
windowless: (Windows) a launcher started on a console without a window stops its server in order.
autoload: the default model loads by itself after a slow first scan (about 20 s).
cutoff: a server that crashes in the middle of a stream: the client's stream ends with an error event
          (about 10 s).
ablation: the Ablation page's runs with ablate_stub.py in place of tools/ablate (pip installs a local wheel,
          no network): the GPU taken and given back, stop, resume, End now, a restart, a killed launcher
          (about two minutes).
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
from contextlib import contextmanager
from pathlib import Path

WINDOWS = os.name == "nt"
results: list[tuple[str, bool, str]] = []
t0 = time.monotonic()


def log(text: str) -> None:
    print(f"[{time.monotonic() - t0:7.1f}s] {text}", flush=True)


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, ok, detail))
    log(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    return ok


def call(port: int, method: str, path: str, body=None, timeout: float = 600.0, headers: dict | None = None,
         full: bool = False):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    sent = dict(headers or {})
    if body is not None:
        sent["Content-Type"] = "application/json"
    conn.request(method, path, body=None if body is None else json.dumps(body).encode(), headers=sent)
    response = conn.getresponse()
    raw = response.read()
    conn.close()
    try:
        value = json.loads(raw) if raw else None
    except ValueError:
        value = raw.decode("utf-8", "replace")
    return (response.status, value, response) if full else (response.status, value)


def stream(port: int, path: str, body: dict, headers: dict | None = None, timeout: float = 120.0):
    """A streamed POST: (status, content type, seconds to the headers, [(seconds, text)])."""
    started = time.monotonic()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    conn.request("POST", path, body=json.dumps(body).encode(),
                 headers={"Content-Type": "application/json", **(headers or {})})
    response = conn.getresponse()
    headers_at = time.monotonic() - started
    pieces: list[tuple[float, str]] = []
    while True:
        data = response.read1(65536)
        if not data:
            break
        pieces.append((time.monotonic() - started, data.decode("utf-8", "replace")))
    conn.close()
    return response.status, response.getheader("Content-Type", ""), headers_at, pieces


def chat(model: str, tokens: int, stream: bool = False) -> dict:
    return {"model": model, "stream": stream, "max_tokens": tokens, "messages": [{"role": "user", "content": "Hi"}]}


def wait_for(predicate, timeout: float, step: float = 0.5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(step)
    return None


class Launcher:
    def __init__(self, exe: Path, fake: Path, data: Path, port: int, windowless: bool = False,
                 extra: tuple[str, ...] = ()) -> None:
        self.exe, self.fake, self.data, self.port = exe, fake, data, port
        self.windowless = windowless  # Windows: a console without a window, as some programs start it
        self.extra = list(extra)
        self.proc: subprocess.Popen | None = None
        self.out = None

    def start(self) -> None:
        kwargs = {}
        if WINDOWS and self.windowless:
            kwargs = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
        elif WINDOWS:
            startup = subprocess.STARTUPINFO()
            startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startup.wShowWindow = 0
            kwargs = {"startupinfo": startup,
                      "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NEW_CONSOLE}
        self.out = open(self.data / "launcher-console.log", "ab")
        self.proc = subprocess.Popen([str(self.exe), "--port", str(self.port), "--data-dir", str(self.data),
                                      "--serve", str(self.fake), "--no-browser", *self.extra],
                                     stdout=self.out, stderr=subprocess.STDOUT, **kwargs)
        if not wait_for(lambda: self._healthy(), 20, 0.1):
            raise RuntimeError("the launcher did not start")

    def _healthy(self) -> bool:
        try:
            return call(self.port, "GET", "/health", timeout=2)[0] == 200
        except OSError:
            return False

    def stop(self) -> int | None:
        """The orderly stop a person gives it (Ctrl+Break / SIGTERM)."""
        assert self.proc
        if WINDOWS:
            import ctypes
            kernel = ctypes.WinDLL("kernel32")
            kernel.FreeConsole()
            kernel.AttachConsole(self.proc.pid)
            kernel.SetConsoleCtrlHandler(None, True)
            kernel.GenerateConsoleCtrlEvent(1, self.proc.pid)
            time.sleep(0.5)
            kernel.FreeConsole()
            kernel.SetConsoleCtrlHandler(None, False)
        else:
            self.proc.send_signal(signal.SIGTERM)
        try:
            return self.proc.wait(60)
        except subprocess.TimeoutExpired:
            return None

    def kill(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(30)
        if self.out:
            self.out.close()
            self.out = None


def instance(port: int) -> dict:
    return call(port, "GET", "/launcher/api/instance")[1]


def events(port: int) -> list[dict]:
    return call(port, "GET", "/launcher/api/eventlog?limit=5000")[1]["lines"]


def activity(port: int) -> list[dict]:
    return call(port, "GET", "/launcher/api/activity")[1]["recent"]


def serve_log(port: int) -> str:
    return "\n".join(line["text"] for line in call(port, "GET", "/launcher/api/logs?limit=20000")[1]["lines"])


def first_model(port: int) -> dict:
    library = wait_for(lambda: (lambda l: l if l.get("models") else None)(call(port, "GET", "/launcher/api/library")[1]), 30)
    if not library:
        raise RuntimeError("no model found in --models")
    return next(m for m in library["models"] if m["family"] != "unknown")


def run_group(name: str, options, settings: dict, body, windowless: bool = False, extra: tuple[str, ...] = ()) -> None:
    """Starts a launcher on a fresh data folder and runs `body(launcher, port)` against it."""
    data = Path(tempfile.mkdtemp(prefix=f"infernix-launcher-{name}-"))
    log(f"== {name}: data folder {data}")
    (data / "infernix-launcher.json").write_text(json.dumps({"schema": 1, "settings": {
        "model_dirs": [options.models], "open_browser": False, **settings}}), encoding="utf-8")
    launcher = Launcher(Path(options.launcher), Path(options.fake), data, options.port, windowless, extra)
    try:
        launcher.start()
        check(f"{name}: the launcher starts with the stand-in server", True)
        body(launcher, options.port, data)
    except Exception as error:  # noqa: BLE001
        check(f"{name}: harness", False, repr(error))
    finally:
        launcher.kill()
        if not options.keep:
            shutil.rmtree(data, ignore_errors=True)


# ---------------------------------------------------------------------------------------------
# Features


@contextmanager
def section(name: str):
    """A part of a group: an exception fails it and the group goes on."""
    try:
        yield
    except Exception as error:  # noqa: BLE001
        check(f"{name}: harness", False, repr(error))


def features(launcher: Launcher, port: int, data: Path) -> None:
    model = first_model(port)
    key = model["key"]
    default_alias = model["config"]["alias"]
    base = list(model["config"]["default_args"])
    fast = base + ["--fake-load-ms", "500"]
    cfg = f"/launcher/api/models/{key}"
    served = 0  # generation requests the server completed

    with section("The listen address"):
        # --port overrides the saved port (1234), and the web UI quotes the address actually in use.
        listen = call(port, "GET", "/launcher/api/state")[1].get("listen", {})
        check("the state names the address in use", listen == {"host": "127.0.0.1", "port": port}, str(listen))

    with section("CORS preflight"):
        # A page on another origin that sends an API key asks for Authorization in its preflight, and a wildcard
        # Access-Control-Allow-Headers does not cover Authorization (Fetch), so it is named (audit 2026-11-18 F-2).
        status, _, response = call(port, "OPTIONS", "/v1/chat/completions", full=True,
                                   headers={"Origin": "http://example.com", "Access-Control-Request-Method": "POST",
                                            "Access-Control-Request-Headers": "authorization, content-type"})
        allowed = [h.strip().lower() for h in (response.getheader("Access-Control-Allow-Headers") or "").split(",")]
        check("a preflight asking for Authorization and Content-Type is allowed both",
              status == 204 and "authorization" in allowed and "content-type" in allowed, f"{status} {allowed}")

    with section("Model listing for coding tools"):
        # The fields other servers' clients read (llama.cpp, llama-swap, vLLM, LM Studio, OpenRouter).
        status, listed = call(port, "GET", "/v1/models")
        ids = [m["id"] for m in listed["data"]] if status == 200 else []
        entry = next((m for m in listed["data"] if m["id"] == default_alias), {}) if status == 200 else {}
        wanted = ["name", "display_name", "description", "context_length", "context_window", "max_model_len",
                  "max_context_length", "max_input_tokens", "max_tokens", "state", "capabilities",
                  "supported_parameters", "created_at"]
        architecture = entry.get("architecture")
        check("/v1/models carries the fields other servers' clients read",
              all(k in entry for k in wanted) and entry.get("status", {}).get("value") in ("loaded", "loading", "unloaded")
              and isinstance(architecture, dict) and "text" in architecture.get("input_modalities", [])
              and entry.get("meta", {}).get("n_ctx") == entry.get("context_length"), json.dumps(entry)[:240])
        status, bare = call(port, "GET", "/models")
        check("/models answers without /v1", status == 200 and [m["id"] for m in bare["data"]] == ids, str(status))
        anthropic = {"anthropic-version": "2023-06-01"}
        status, page = call(port, "GET", "/v1/models?limit=1", headers=anthropic)
        _, after = call(port, "GET", f"/v1/models?limit=1&after_id={ids[0]}", headers=anthropic) if ids else (0, {})
        _, before = call(port, "GET", f"/v1/models?before_id={ids[-1]}", headers=anthropic) if ids else (0, {})
        check("Anthropic paging: limit, after_id, before_id, has_more, first_id and last_id",
              len(ids) >= 2 and status == 200 and [m["id"] for m in page["data"]] == ids[:1] and page["has_more"]
              and page["first_id"] == page["last_id"] == ids[0] and [m["id"] for m in after["data"]] == ids[1:2]
              and [m["id"] for m in before["data"]] == ids[:-1][-20:], f"{len(ids)} models")
        # Claude Code lists only ids containing "claude" or "anthropic".
        claude = {"User-Agent": "claude-cli/2.1.300 (external, cli)", **anthropic}
        status, seen = call(port, "GET", "/v1/models?limit=1000", headers=claude)
        seen_ids = [m["id"] for m in seen.get("data", [])] if status == 200 else []
        check("Claude Code sees every model under an id it keeps",
              len(seen_ids) == len(ids) and all("claude" in i.lower() or "anthropic" in i.lower() for i in seen_ids)
              and [m["display_name"] for m in seen["data"]] == ids, str(seen_ids[:2]))
        spelled = f"{default_alias}:claude-code"
        status, one = call(port, "GET", f"/v1/models/{spelled}", headers=anthropic)
        check("...which /v1/models/{id} answers", status == 200 and one.get("id") == spelled, str(status))
        status, _ = call(port, "POST", "/v1/messages", chat(spelled, 8), headers=anthropic)
        served += status == 200
        last = activity(port)[-1]
        check("...and a request may name", status == 200 and last.get("routed_by") == "name"
              and last.get("model_key") == key, f"{status} {last.get('routed_by')} {last.get('model_key')}")
        check("/v1/health answers", call(port, "GET", "/v1/health")[0] == 200)
        # Nothing stays loaded: the next section starts the model with its own configuration.
        call(port, "POST", "/launcher/api/unload", {})
        wait_for(lambda: instance(port)["state"] == "idle", 30)

    with section("A route that never streams, asked to stream while its model loads"):
        # count_tokens answers JSON whatever `stream` says: the launcher used to start an event stream for the load
        # and relay only stream chunks, so the client got an empty stream (audit 2026-11-18 F-8).
        status, counted, response = call(port, "POST", "/v1/messages/count_tokens",
                                         {"model": default_alias, "stream": True,
                                          "messages": [{"role": "user", "content": "hi"}]},
                                         headers={"anthropic-version": "2023-06-01"}, full=True)
        check("count_tokens with stream true during a load answers its JSON",
              status == 200 and "application/json" in (response.getheader("Content-Type") or "")
              and isinstance(counted, dict) and counted.get("input_tokens") == 56,
              f"{status} {response.getheader('Content-Type')} {str(counted)[:120]}")
        # /v1/score (a --causal-scoring server) is proxied as a JSON-only route too.
        status, scored, response = call(port, "POST", "/v1/score",
                                        {"model": default_alias, "stream": True, "apply_chat_template": False,
                                         "prompt": "The capital of France is", "continuation": " Paris"},
                                        full=True)
        check("/v1/score is proxied and answers its JSON",
              status == 200 and "application/json" in (response.getheader("Content-Type") or "")
              and isinstance(scored, dict) and scored.get("object") == "score",
              f"{status} {response.getheader('Content-Type')} {str(scored)[:120]}")
        call(port, "POST", "/launcher/api/unload", {})
        wait_for(lambda: instance(port)["state"] == "idle", 30)

    with section("llama.cpp endpoints (opt-in)"):
        check("off by default: /props, /models/load and /models/sse are not answered",
              call(port, "GET", "/props")[0] == 404 and call(port, "GET", "/models/sse")[0] == 404
              and call(port, "POST", "/models/load", {"model": default_alias})[0] == 404)
        call(port, "PUT", "/launcher/api/settings", {"llamacpp_api": True})
        entry = next(m for m in call(port, "GET", "/v1/models")[1]["data"] if m["id"] == default_alias)
        status, props = call(port, "GET", "/props?" + urllib.parse.urlencode({"model": default_alias}))
        settings = (props or {}).get("default_generation_settings", {}) if status == 200 else {}
        check("/props?model= describes a configuration as llama-server does, without loading it",
              status == 200 and settings.get("n_ctx") == entry["context_length"] and isinstance(props.get("total_slots"), int)
              and props.get("modalities", {}).get("vision") is False
              and props.get("chat_template_caps", {}).get("supports_tool_calls") is True
              and instance(port)["state"] == "idle", json.dumps(props)[:200])
        status, unknown = call(port, "GET", "/props?model=no-such-model")
        check("...and refuses an unknown id like llama-server (400)", status == 400 and "not found" in json.dumps(unknown),
              str(status))
        seen: list[dict] = []

        def watch() -> None:
            # /models/sse: one `data:` line per event, until a failed load's exit code arrives.
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
            conn.request("GET", "/models/sse")
            response, buffer, deadline = conn.getresponse(), "", time.monotonic() + 60
            while time.monotonic() < deadline and not any((e.get("data") or {}).get("exit_code") for e in seen):
                data = response.read1(65536)
                if not data:
                    break
                buffer += data.decode("utf-8", "replace")
                while "\n\n" in buffer:
                    block, buffer = buffer.split("\n\n", 1)
                    seen.extend(json.loads(line[6:]) for line in block.splitlines() if line.startswith("data: "))
            conn.close()

        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        time.sleep(0.5)
        status, loaded = call(port, "POST", "/models/load", {"model": default_alias})
        ready = wait_for(lambda: instance(port)["state"] == "ready", 60)
        again, _ = call(port, "POST", "/models/load", {"model": default_alias})
        check("/models/load loads it, answering at once; a second load is refused (400)",
              status == 200 and loaded == {"success": True} and bool(ready) and again == 400, f"{status} {loaded} {again}")
        _, current = call(port, "GET", "/props")
        check("/props without a model describes the loaded one", (current or {}).get("model_alias") == default_alias,
              str((current or {}).get("model_alias")))
        status, unloaded = call(port, "POST", "/models/unload", {"model": default_alias})
        idle = wait_for(lambda: instance(port)["state"] == "idle", 30)
        again, _ = call(port, "POST", "/models/unload", {"model": default_alias})
        check("/models/unload stops it; unloading it again is refused (400)",
              status == 200 and unloaded == {"success": True} and bool(idle) and again == 400, f"{status} {again}")
        # A failed load reaches the stream as `unloaded` with a non-zero exit code.
        call(port, "PUT", f"{cfg}/config", {"args": base + ["--fake-load-ms", "300", "--fake-fail-load",
                                                               "cudaMalloc failed: out of memory"]})
        call(port, "POST", "/models/load", {"model": default_alias})
        watcher.join(40)
        statuses = [(e.get("data") or {}).get("status") for e in seen
                    if e.get("event") == "model_status" and e.get("model") == default_alias]
        check("/models/sse reports loading, loaded, unloaded and a failed load's exit code, and list changes",
              statuses[:3] == ["loading", "loaded", "unloaded"]
              and any((e.get("data") or {}).get("exit_code") for e in seen)
              and any(e.get("event") == "models_reload" and e.get("model") == "*" for e in seen), str(statuses))
        call(port, "POST", f"{cfg}/config/reset", {})
        wait_for(lambda: instance(port)["state"] in ("idle", "failed"), 30)
        call(port, "PUT", "/launcher/api/settings", {"llamacpp_api": False})
        check("...and turned off, the paths are not answered again", call(port, "GET", "/props")[0] == 404)

    with section("The model id, the environment"):
        # The model id, the environment.
        status, config = call(port, "PUT", f"{cfg}/config", {"args": fast, "alias": "fake-main",
                                                              "env": [{"name": "INFERNIX_FAKE_PROBE", "value": "main config"}]})
        check("the model id can be set", status == 200 and config["alias"] == "fake-main", str(config)[:160])
        ids = [m["id"] for m in call(port, "GET", "/v1/models")[1]["data"]]
        check("/v1/models lists it by that id", "fake-main" in ids and default_alias not in ids, str(ids))

    with section("A named configuration of the same artifact"):
        # A named configuration of the same artifact.
        status, named = call(port, "POST", f"{cfg}/configs", {"name": "Long context", "args": fast + ["--max-context", "65536"]})
        named_id, named_alias = named.get("id", ""), named.get("alias", "")
        check("a named configuration is created", status == 200 and named_id.startswith(key + "@") and
              named_alias == default_alias + "-long-context" and not named["is_default"], str(named)[:200])
        status, clash = call(port, "PUT", f"/launcher/api/models/{named_id}/config", {"alias": "FAKE-MAIN"})
        check("two configurations cannot share a model id", status == 400, str(clash)[:120])
        ids = [m["id"] for m in call(port, "GET", "/v1/models")[1]["data"]]
        check("both configurations are listed", "fake-main" in ids and named_alias in ids, str(ids))

        status, body, response = call(port, "POST", "/v1/chat/completions", chat("fake-main", 8), full=True)
        served += status == 200
        check("a request by model id loads that configuration", status == 200 and instance(port)["model_key"] == key,
              str(body)[:120])
        check("...and is answered with X-Infernix-Request-Id", (response.getheader("X-Infernix-Request-Id") or "").isdigit())
        check("...with the configuration's environment", "INFERNIX_FAKE_PROBE=main config" in serve_log(port))
        status, body = call(port, "POST", "/v1/chat/completions", chat(named_alias.upper(), 8))
        served += status == 200
        now = instance(port)
        check("a request for the named configuration switches to it", status == 200 and now["model_key"] == named_id and
              "65536" in now["argv"] and now["config_name"] == "Long context", f"{now['model_key']} {now['config_name']}")
        last = activity(port)[-1]
        check("...routed by its name", last["routed_by"] == "name" and last["model_key"] == named_id, str(last)[:160])

    with section("Memory breakdown"):
        # The server's memory by part reaches the dashboard unchanged (the stand-in reports a dense model's).
        def memory() -> dict | None:
            status, metrics = call(port, "GET", "/launcher/api/metrics")
            return metrics.get("memory") if status == 200 and isinstance(metrics, dict) else None

        held = wait_for(memory, 10) or {}
        device = held.get("device") or {}
        parts = device.get("parts") or []
        keys = [part.get("key") for part in parts]
        check("/launcher/api/metrics carries the server's memory by part",
              {"weights", "kv_cache", "mtp_kv", "workspace", "cuda_graphs"} <= set(keys) and device.get("total_bytes", 0) > 0
              and (device.get("process_bytes") or 0) >= sum(part.get("bytes", 0) for part in parts)
              and bool((held.get("host") or {}).get("parts")) and "ssd" in held and held["ssd"] is None, str(keys))

    with section("Script export and import, both shells"):
        # Script export and import, both shells.
        for shell in ("cmd", "sh"):
            status, script = call(port, "GET", f"{cfg}/script?shell={shell}")
            text = script.get("text", "") if isinstance(script, dict) else ""
            check(f"export as {shell}", status == 200 and "fake-main" in text and "INFERNIX_FAKE_PROBE" in text,
                  script.get("file_name", "") if isinstance(script, dict) else str(script)[:100])
            status, imported = call(port, "POST", "/launcher/api/scripts/import", {"text": text, "file_name": script["file_name"]})
            command = (imported or {}).get("commands", [{}])[0]
            check(f"...and imported back unchanged ({shell})",
                  status == 200 and (command.get("model") or {}).get("key") == key and command.get("alias") == "fake-main"
                  and command.get("args") == fast
                  and command.get("env") == [{"name": "INFERNIX_FAKE_PROBE", "value": "main config"}],
                  json.dumps(command)[:240])

    with section("Will it fit?"):
        # Will it fit? (The stand-in's plan fits up to 225,280 tokens of context.)
        status, plan = call(port, "POST", f"{cfg}/plan", {"args": fast + ["--max-context", "65536"]})
        check("a plan that fits", status == 200 and plan.get("fits") is True and plan.get("max_context") and
              plan.get("vram") and plan["vram"]["parts"], str(plan)[:200])
        # A LoRA adapter's factors are a part of the plan beside the weights, and count in what it plans.
        adapter_dir = Path(tempfile.mkdtemp(prefix="infernix-launcher-lora-"))
        status, plain = call(port, "POST", f"{cfg}/plan", {"args": fast + ["--max-context", "65536"]})
        status_lora, adapted = call(port, "POST", f"{cfg}/plan",
                                    {"args": fast + ["--max-context", "65536", "--lora", str(adapter_dir)]})
        labels = [part.get("label") for part in (adapted.get("vram") or {}).get("parts", [])]
        check("a LoRA adapter's VRAM is a part of the plan and of what it plans",
              status == 200 and status_lora == 200 and "Adapters" in labels
              and (adapted.get("vram") or {}).get("planned", 0) - (plain.get("vram") or {}).get("planned", 0)
              == 318767104, str(labels))
        # So is an ablation sidecar's direction (a dense 27B's 20 KiB), alone and beside a LoRA adapter's factors.
        sidecar = Path(tempfile.mkdtemp(prefix="infernix-launcher-ablate-")) / "sidecar.json"
        sidecar.write_text("{}", encoding="utf-8")
        status_cut, cut = call(port, "POST", f"{cfg}/plan",
                               {"args": fast + ["--max-context", "65536", "--ablate", str(sidecar)]})
        status_both, both = call(port, "POST", f"{cfg}/plan", {"args": fast + [
            "--max-context", "65536", "--lora", str(adapter_dir), "--ablate", str(sidecar)]})

        def adapters(plan: dict) -> int | None:
            found = [part for part in (plan.get("vram") or {}).get("parts", []) if part.get("label") == "Adapters"]
            return found[0].get("bytes") if found else None

        planned = lambda plan: (plan.get("vram") or {}).get("planned", 0)  # noqa: E731
        check("an ablation sidecar's direction is the plan's Adapters part and counts in what it plans",
              status_cut == 200 and status_both == 200 and adapters(cut) == 20480 and adapters(both) == 318767104 + 20480
              and planned(cut) - planned(plain) == 20480 and planned(both) - planned(adapted) == 20480,
              f"{adapters(cut)} {adapters(both)} {planned(cut) - planned(plain)} {planned(both) - planned(adapted)}")
        status, plan = call(port, "POST", f"{cfg}/plan", {"args": fast + ["--max-context", "250000"]})
        suggestions = plan.get("suggestions", [])
        first = suggestions[0] if suggestions and isinstance(suggestions[0], dict) else {}
        check("a plan that does not fit, with the largest context that fits first",
              status == 200 and plan.get("fits") is False and all(isinstance(s, dict) for s in suggestions)
              and f"{plan.get('max_context'):,}" in first.get("title", "")
              and str(plan.get("max_context")) in (first.get("args") or []),
              "; ".join(str(s.get("title") if isinstance(s, dict) else s) for s in suggestions))

    with section("Launch preview: an option given without the one it needs"):
        # infernix-serve refuses these pairs at startup; the preview says so while the form is edited.
        def warned(extra: list[str]) -> list[str]:
            status, preview = call(port, "POST", f"{cfg}/preview", {"args": base + extra})
            return preview.get("warnings", []) if status == 200 else [f"HTTP {status}"]

        alone = [
            (["--ssd-context-gib", "64"], "--ssd-context-gib needs --ssd-context-dir"),
            (["--ssd-context-dir", str(data)], "--ssd-context-dir needs --ssd-context-gib"),
            (["--ssd-context-gib-per-hour", "4"], "--ssd-context-gib-per-hour needs --ssd-context-gib"),
            (["--prefix-cache-save-mins", "30"], "--prefix-cache-save-mins needs a prefix cache file"),
            (["--ngram-native-sessions"], "--ngram-native-sessions needs --ngram-archive-mib"),
            (["--request-log-max-mib", "8"], "--request-log-max-mib needs --request-log-jsonl"),
            (["--request-log-keep", "2"], "--request-log-keep needs --request-log-max-mib"),
            (["--vision-offload", "on"], "--vision-offload on needs --vision"),
            (["--causal-scoring", "--max-context", "auto"], "--causal-scoring needs --max-context N"),
        ]
        unnamed = [want for extra, want in alone if not any(w.startswith(want) for w in warned(extra))]
        check("the preview names each", not unnamed, "; ".join(unnamed))
        together = warned(["--ssd-context-gib", "64", "--ssd-context-dir", str(data), "--ssd-context-gib-per-hour", "4",
                           "--prefix-cache-file", str(data / "prefix.cache"), "--prefix-cache-save-mins", "30",
                           "--ngram-archive-mib", "64", "--ngram-native-sessions",
                           "--request-log-jsonl", str(data / "requests.jsonl"), "--request-log-max-mib", "8",
                           "--request-log-keep", "2", "--vision", "--vision-offload", "on",
                           "--causal-scoring", "--max-context", "8192"])
        check("...and none of them when both are given", not any(" needs " in w for w in together), "; ".join(together))
        absent = warned(["--ssd-context-gib", "64", "--ssd-context-dir", str(data / "no-such-folder")])
        check("...and an SSD prefix cache folder that does not exist",
              any(w.startswith("--ssd-context-dir: folder not found") for w in absent), "; ".join(absent))

    with section("A server started above info is taken from /health at once"):
        # infernix-serve logs its "listening on" line at Info, so one started with --log-level warning never shows
        # it: the launcher then takes /health at once, not after the 30 s it gives a server without the line.
        call(port, "POST", "/launcher/api/unload", {})
        wait_for(lambda: instance(port)["state"] == "idle", 30)
        call(port, "PUT", f"{cfg}/config", {"args": fast + ["--log-level", "warning"]})
        started = time.monotonic()
        status, _ = call(port, "POST", "/v1/chat/completions", chat("fake-main", 8))
        took = time.monotonic() - started
        served += status == 200
        check("a server started with --log-level warning is answered without the 30 s wait for its ready line",
              status == 200 and took < 15, f"{status} after {took:.1f} s")
        call(port, "POST", "/launcher/api/unload", {})
        wait_for(lambda: instance(port)["state"] == "idle", 30)

    with section("Keep-alive while a model loads"):
        # Keep-alive while a model loads (the main configuration now takes 12 s to load).
        slow = base + ["--fake-load-ms", "12000"]
        call(port, "PUT", f"{cfg}/config", {"args": slow})
        status, kind, headers_at, pieces = stream(port, "/v1/chat/completions", chat("fake-main", 16, True))
        text = "".join(piece for _, piece in pieces)
        alive = [at for at, piece in pieces if ": loading" in piece]
        first_data = next((at for at, piece in pieces if "data: {" in piece), None)
        served += "[DONE]" in text
        check("a stream that waits for a load is answered at once", status == 200 and "text/event-stream" in kind and
              headers_at < 2.0, f"{status} {kind} after {headers_at:.1f} s")
        gaps = [b - a for a, b in zip([0.0] + alive, alive)]
        check("...with keep-alives every 5 s while it loads", len(alive) >= 2 and max(gaps) < 7.0,
              f"at {[round(a, 1) for a in alive]}")
        check("...then the model's stream", first_data is not None and first_data > 10 and "[DONE]" in text,
              f"first data at {first_data}")
        call(port, "POST", "/launcher/api/unload", {})
        wait_for(lambda: instance(port)["state"] == "idle", 30)
        status, kind, headers_at, pieces = stream(port, "/v1/messages", {**chat("fake-main", 16, True)},
                                                  headers={"anthropic-version": "2023-06-01"})
        text = "".join(piece for _, piece in pieces)
        served += "message_stop" in text
        check("an Anthropic stream gets ping events while it loads", status == 200 and "event: ping" in text and
              text.index("event: ping") < text.find("message_start") and "message_stop" in text, text[:160])
        # The server answers a stream that waited for the load with a JSON error (it refuses a negative output
        # limit): the error event carries the server's message, not only its status.
        call(port, "POST", "/launcher/api/unload", {})
        wait_for(lambda: instance(port)["state"] == "idle", 30)
        status, kind, headers_at, pieces = stream(port, "/v1/chat/completions", chat("fake-main", -1, True))
        text = "".join(piece for _, piece in pieces)
        check("a server's error answer to a stream that waited for the load keeps its message",
              status == 200 and '"error"' in text and "max_tokens must be at least 1" in text, text[:200])
        # Compaction never streams (the server refuses it): asked to stream during a load, it gets the server's
        # own answer, its status and message, not a stream.
        call(port, "POST", "/launcher/api/unload", {})
        wait_for(lambda: instance(port)["state"] == "idle", 30)
        status, kind, headers_at, pieces = stream(port, "/v1/responses/compact",
                                                  {"model": "fake-main", "input": "hello", "stream": True})
        text = "".join(piece for _, piece in pieces)
        check("compaction asked to stream during a load gets the server's own error answer",
              status == 400 and "text/event-stream" not in kind and "compaction is not supported" in text,
              f"{status} {kind} {text[:160]}")
        call(port, "PUT", f"{cfg}/config", {"args": base + ["--fake-load-ms", "300", "--fake-fail-load",
                                                               "cudaMalloc failed: out of memory"]})
        call(port, "POST", "/launcher/api/unload", {})
        wait_for(lambda: instance(port)["state"] == "idle", 30)
        status, kind, headers_at, pieces = stream(port, "/v1/chat/completions", chat("fake-main", 16, True))
        text = "".join(piece for _, piece in pieces)
        check("a failed load ends the stream with an error event", status == 200 and '"error"' in text and
              "out of memory" in text, text[:200])

        # The failed load holds the model: coding tools' retries are refused at once, not loaded again.
        def loads() -> int:
            return sum(1 for e in events(port) if e["category"] == "model" and e["message"].startswith("loading "))

        loads_before = loads()
        started = time.monotonic()
        status, body, response = call(port, "POST", "/v1/chat/completions", chat("fake-main", 8), full=True)
        took, retry = time.monotonic() - started, response.getheader("Retry-After") or ""
        check("a model whose load failed is refused at once, with Retry-After over 60 s",
              status == 503 and took < 2.0 and retry.isdigit() and 60 < int(retry) <= 120 and
              "out of memory" in json.dumps(body), f"{status} after {took:.1f} s, Retry-After {retry!r}")
        status, kind, _, _ = stream(port, "/v1/messages", {**chat("fake-main", 16, True)},
                                    headers={"anthropic-version": "2023-06-01"})
        check("...a stream with an HTTP error (Anthropic 529), not an event stream",
              status == 529 and "text/event-stream" not in kind, f"{status} {kind}")
        check("...without loading it again", loads() == loads_before, f"{loads() - loads_before} more loads")
        call(port, "PUT", f"{cfg}/config", {"args": fast})
        status, _ = call(port, "POST", "/v1/chat/completions", chat("fake-main", 8))
        served += status == 200
        check("a changed configuration ends the hold", status == 200, str(status))

    with section("Responses compaction and cancel reach the server"):
        # The stand-in answers both as infernix-serve does; the launcher's own 404 has another code.
        status, body = call(port, "POST", "/v1/responses/compact", {"model": "fake-main", "input": "Hi"})
        check("/v1/responses/compact gets the server's refusal", status == 400 and
              ((body or {}).get("error") or {}).get("code") == "compaction_not_supported", str(body)[:160])
        status, body = call(port, "POST", "/v1/responses/resp_unknown/cancel", {})
        check("/v1/responses/{id}/cancel reaches the loaded server", status == 404 and
              ((body or {}).get("error") or {}).get("code") == "response_not_found", str(body)[:160])

    with section("Routes and the pin"):
        # Routes and the pin.
        call(port, "PUT", "/launcher/api/settings", {"model_routes": [{"pattern": "claude-*", "target": named_id}]})
        status, _ = call(port, "POST", "/v1/chat/completions", chat("claude-sonnet-4-5", 8))
        served += status == 200
        last = activity(port)[-1]
        check("a model route sends claude-* to its configuration", status == 200 and last["routed_by"] == "route" and
              last["model_key"] == named_id, str(last)[:160])
        status, _ = call(port, "POST", "/v1/chat/completions", chat("gpt-4o", 8))
        served += status == 200
        last = activity(port)[-1]
        check("an unknown name goes to the loaded model", status == 200 and last["routed_by"] == "loaded" and
              last["model_key"] == named_id, str(last)[:160])
        # The web UI's chat names no model for "the loaded model"; the stand-in, as infernix-serve,
        # refuses a request without one.
        nameless = {key: value for key, value in chat("", 8, True).items() if key != "model"}
        status, _, _, pieces = stream(port, "/v1/chat/completions", nameless)
        text = "".join(piece for _, piece in pieces)
        served += "[DONE]" in text
        last = activity(port)[-1]
        check("a request that names no model goes to the loaded model, sent with its id",
              status == 200 and "[DONE]" in text and last["routed_by"] == "loaded" and last["model_key"] == named_id,
              f"{status} {text[:160]}")
        nameless = {key: value for key, value in chat("", 8).items() if key != "model"}
        status, body = call(port, "POST", "/v1/messages", nameless, headers={"anthropic-version": "2023-06-01"})
        served += status == 200
        check("...an Anthropic one too", status == 200 and (body or {}).get("model") == named_alias, str(body)[:160])
        call(port, "PUT", "/launcher/api/settings", {"pin_loaded": True})
        status, _ = call(port, "POST", "/v1/chat/completions", chat("fake-main", 8))
        served += status == 200
        last = activity(port)[-1]
        check("the pin keeps the loaded model whatever a request names", status == 200 and last["routed_by"] == "pinned"
              and instance(port)["model_key"] == named_id and instance(port)["pinned"], str(last)[:160])
        call(port, "PUT", "/launcher/api/settings", {"pin_loaded": False, "model_routes": []})

    with section("Prompt progress"):
        # Prompt progress: a long prompt (about 100,000 tokens to the stand-in) takes seconds.
        long_prompt = {"model": named_alias, "stream": True, "max_tokens": 8, "return_progress": True,
                       "messages": [{"role": "user", "content": "word " * 80000}]}
        seen: list[dict] = []
        finished = threading.Event()

        def watch() -> None:
            while not finished.is_set():
                for record in activity(port):
                    if record["state"] == "running" and record["progress"]:
                        seen.append(record["progress"])
                time.sleep(0.3)

        watcher = threading.Thread(target=watch)
        watcher.start()
        status, kind, _, pieces = stream(port, "/v1/chat/completions", long_prompt)
        finished.set()
        watcher.join()
        text = "".join(piece for _, piece in pieces)
        served += "[DONE]" in text
        chunks = [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: {")]
        progress = [c["prompt_progress"]["processed"] for c in chunks if "prompt_progress" in c]
        timings = [c["timings"] for c in chunks if "timings" in c]
        check("the server's prompt progress reaches the client", len(progress) >= 3 and progress == sorted(progress),
              f"{len(progress)} events")
        check("...and its timings", bool(timings) and timings[-1].get("predicted_n") == 8, str(timings[-1:])[:160])
        prefill = [p for p in seen if p["phase"] == "prefill" and p["processed_tokens"] < p["prompt_tokens"]]
        check("the dashboard sees the prompt's progress", len(prefill) >= 2 and prefill[-1]["processed_tokens"] >
              prefill[0]["processed_tokens"], f"{len(prefill)} readings")

    with section("API keys and sign-in"):
        # API keys and sign-in.
        status, api_key = call(port, "POST", "/launcher/api/keys", {"name": "check", "scope": "api"})
        secret = api_key.get("secret", "")
        check("an API key is created and shown once", status == 200 and secret.startswith("inx-") and len(secret) == 36 and
              "hash" not in json.dumps(call(port, "GET", "/launcher/api/settings")[1]["api_keys"]), secret[:8])
        check("with a key, /v1 needs one", call(port, "GET", "/v1/models")[0] == 401)
        check("...as a Bearer token", call(port, "GET", "/v1/models", headers={"Authorization": f"Bearer {secret}"})[0] == 200)
        check("...or as x-api-key", call(port, "GET", "/v1/models", headers={"x-api-key": secret})[0] == 200)
        check("the web UI still answers this computer", call(port, "GET", "/launcher/api/state")[0] == 200)
        status, _ = call(port, "POST", "/v1/chat/completions", chat(named_alias, 8), headers={"x-api-key": secret})
        served += status == 200
        check("requests are recorded with the key's name", status == 200 and activity(port)[-1]["key_name"] == "check")
        status, admin_key = call(port, "POST", "/launcher/api/keys", {"name": "admin", "scope": "admin"})
        status, refused = call(port, "POST", "/launcher/api/session", {"key": secret})
        check("signing in needs an admin key", status == 401, str(refused)[:100])
        status, session, response = call(port, "POST", "/launcher/api/session", {"key": admin_key["secret"]}, full=True)
        cookie = response.getheader("Set-Cookie") or ""
        check("an admin key signs in with an HttpOnly cookie", status == 200 and session["signed_in"] and
              "HttpOnly" in cookie and "SameSite=Strict" in cookie, cookie[:60])
        for created in (api_key, admin_key):
            call(port, "DELETE", f"/launcher/api/keys/{created['key']['id']}")
        check("without keys, /v1 is open again", call(port, "GET", "/v1/models")[0] == 200)

    with section("Usage"):
        # Usage: every completed request once, by configuration and client.
        def usage_rows():
            return call(port, "GET", "/launcher/api/usage")[1]["rows"]

        counted = wait_for(lambda: (lambda rows: rows if sum(r["requests"] for r in rows) >= served else None)(usage_rows()), 10)
        rows = counted or usage_rows()
        total = sum(r["requests"] for r in rows)
        check("usage counts every request once", total == served, f"{total} counted, {served} served")
        by_key = {r["client"] for r in rows if r["config_id"] == named_id}
        check("...by configuration and client", {r["config_id"] for r in rows} == {key, named_id} and "check" in by_key and
              all(r["prompt_tokens"] > 0 and r["output_tokens"] > 0 for r in rows), str(sorted(by_key)))

    with section("Chats"):
        # Chats.
        chat_doc = {"title": "Check", "model": named_id, "options": {"temperature": None, "max_tokens": None,
                                                                     "thinking": True},
                    "messages": [{"id": "m1", "role": "user", "content": "Hi", "images": [], "reasoning": None,
                                  "stats": None, "model": None, "error": None, "t": 1}]}
        status, _ = call(port, "PUT", "/launcher/api/chats/check-1", chat_doc)
        listed = [c["id"] for c in call(port, "GET", "/launcher/api/chats")[1]]
        stored = call(port, "GET", "/launcher/api/chats/check-1")[1]
        check("a chat is stored and listed", status == 200 and "check-1" in listed and stored["messages"][0]["content"] == "Hi")
        call(port, "DELETE", "/launcher/api/chats/check-1")
        check("...and deleted", call(port, "GET", "/launcher/api/chats/check-1")[0] == 404)

    with section("A configuration's idle unload"):
        # A configuration's idle unload (1 minute; the global setting is off).
        call(port, "PUT", f"/launcher/api/models/{named_id}/config", {"idle_unload_min": 1})
        status, _ = call(port, "POST", "/v1/chat/completions", chat(named_alias, 8))
        now = instance(port)
        planned = (now["idle_unload_at"] or 0) - (now["last_request_at"] or 0)
        check("the dashboard shows when the model unloads", status == 200 and 59000 <= planned <= 61000, str(planned))
        unloaded = wait_for(lambda: instance(port)["state"] == "idle", 150, 2)
        check("an idle configuration unloads after its minutes", bool(unloaded),
              next((e["message"] for e in reversed(events(port)) if "idle" in e["message"]), "no event"))
        call(port, "PUT", f"/launcher/api/models/{named_id}/config", {"idle_unload_min": None})

    with section("Deleting a named configuration; the default one stays"):
        # Deleting a named configuration; the default one stays.
        check("the default configuration cannot be deleted", call(port, "DELETE", f"{cfg}/config")[0] == 400)
        check("a named configuration can", call(port, "DELETE", f"/launcher/api/models/{named_id}/config")[0] == 200 and
              named_alias not in [m["id"] for m in call(port, "GET", "/v1/models")[1]["data"]])

    with section("LoRA adapters in a model folder"):
        # A folder with adapter_config.json and adapter_model.safetensors joins the library, matched to the dense
        # models of the base its config names, and its files can be deleted; a folder without the weights is none.
        folder = data / "adapters"
        adapter = folder / "Qwen3.8-27B-check-adapted"
        adapter.mkdir(parents=True)
        config = {"peft_type": "LORA", "base_model_name_or_path": "Qwen/Qwen3.8-27B", "r": 16, "lora_alpha": 32,
                  "target_modules": ["q_proj", "v_proj"]}
        (adapter / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
        (adapter / "adapter_model.safetensors").write_bytes(b"\0" * 4096)
        (folder / "no-weights").mkdir()
        (folder / "no-weights" / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
        dirs = call(port, "GET", "/launcher/api/settings")[1]["model_dirs"]
        call(port, "PUT", "/launcher/api/settings", {"model_dirs": dirs + [str(folder)]})

        def listed():
            return [a for a in call(port, "GET", "/launcher/api/library")[1].get("adapters", []) if a["name"] == adapter.name]

        found = (wait_for(listed, 30) or [None])[0]
        check("an adapter folder joins the library with its config",
              bool(found) and found["bytes"] == 4096 and found["config"]["rank"] == 16 and
              found["config"]["base_model"] == "Qwen/Qwen3.8-27B" and found["config"]["target_modules"] == ["q_proj", "v_proj"],
              str(found)[:200])
        library = call(port, "GET", "/launcher/api/library")[1]
        dense = sorted(m["key"] for m in library["models"] if m["family"] == "qwen3.8-27b" and not m.get("num_experts"))
        check("it fits the Qwen3.8-27B models in the folders", bool(found) and sorted(found["compatible_models"]) == dense,
              f"{found and found['compatible_models']} vs {dense}")
        check("a folder without the weights is no adapter", all(a["name"] != "no-weights" for a in library["adapters"]))
        if found:
            status, _ = call(port, "DELETE", f"/launcher/api/adapters/{found['key']}", {"confirm": "wrong"})
            check("deleting an adapter needs its folder name", status == 400 and (adapter / "adapter_model.safetensors").exists(),
                  str(status))
            status, _ = call(port, "DELETE", f"/launcher/api/adapters/{found['key']}", {"confirm": adapter.name})
            check("deleting it removes its files and its emptied folder", status == 200 and not adapter.exists(), str(status))
            check("and the library drops it", wait_for(lambda: not listed(), 30) is True)
        call(port, "PUT", "/launcher/api/settings", {"model_dirs": dirs})

    code = launcher.stop()
    check("features: an orderly stop", code == 0, str(code))
    saved = json.loads((data / "infernix-launcher-usage.json").read_text(encoding="utf-8"))
    check("usage is saved when the launcher stops", bool(saved), str(saved)[:120])


# ---------------------------------------------------------------------------------------------
# Watchdog


def watchdog(launcher: Launcher, port: int, data: Path, skip_ram: bool) -> None:
    model = first_model(port)
    key, alias = model["key"], model["config"]["alias"]
    base_args = list(model["config"]["default_args"])
    status, _ = call(port, "PUT", f"/launcher/api/models/{key}/config", {"args": base_args + ["--fake-load-ms", "500"]})
    check("stand-in configuration saved", status == 200)

    status, body = call(port, "POST", "/v1/chat/completions", chat(alias, 16))
    check("a request loads the model", status == 200, str(body)[:100])
    lines = events(port)
    check("the event log records the load", any(e["category"] == "model" and "is ready" in e["message"] for e in lines))
    api = [e for e in lines if e["category"] == "api" and e["fields"]["path"] == "/v1/chat/completions"]
    check("the event log records the request's metadata",
          bool(api) and api[-1]["fields"]["status"] == 200 and api[-1]["fields"]["model_key"] == key
          and api[-1]["fields"]["bytes_out"] > 0, json.dumps(api[-1]["fields"]) if api else "none")
    events_file = Path(call(port, "GET", "/launcher/api/state")[1]["logs"]["events_file"])
    check("the event log file is beside the memory file", events_file.parent == data and events_file.exists(),
          str(events_file))

    def incident_after(first: int, kind: str, timeout: float):
        return wait_for(lambda: (lambda i: i if i["incident"] and i["incident"]["kind"] == kind and
                                 i["instance"] > first and i["state"] == "ready" else None)(instance(port)), timeout)

    # A crash: restarted automatically.
    before = instance(port)
    try:
        call(before["port"], "POST", "/fake/crash", {}, timeout=5)
    except OSError:
        pass
    after = incident_after(before["instance"], "crash", 30)
    check("a crash is detected and the model restarted", bool(after) and after["incident"]["restarted"],
          (after or instance(port))["incident"] and (after or instance(port))["incident"]["message"])

    # An idle server is left alone, however long it waits (both timeouts are 10 s here).
    before = instance(port)
    time.sleep(15)
    now = instance(port)
    check("an idle server keeps running past the timeouts",
          now["instance"] == before["instance"] and now["state"] == "ready", f"instance {now['instance']}")

    # A lock-up while idle is only noted; a request that then waits on it ends it.
    call(before["port"], "POST", "/fake/hang", {}, timeout=5)
    time.sleep(24)  # three unanswered polls, each up to its 5 s read timeout
    now = instance(port)
    noted = [e for e in events(port) if e["category"] == "watchdog" and "while idle" in e["message"]]
    check("an idle server that stops answering is noted and left running",
          now["instance"] == before["instance"] and now["state"] == "ready" and bool(noted),
          noted[-1]["message"] if noted else "no notice")
    waiting = {}

    def waits_on_it():
        try:
            waiting["status"] = call(port, "POST", "/v1/chat/completions", chat(alias, 8), timeout=120)[0]
        except OSError as error:
            waiting["status"] = repr(error)

    started = time.monotonic()
    waiter = threading.Thread(target=waits_on_it)
    waiter.start()
    after = incident_after(before["instance"], "hang", 60)
    waiter.join(60)
    check("a request waiting on a locked-up server gets it ended and the model restarted", bool(after),
          f"{time.monotonic() - started:.0f} s; {after and after['incident']['message']}")
    check("...and that request fails rather than hangs", waiting.get("status") not in (None, 200), str(waiting))

    # A stall: requests running, no tokens.
    before = instance(port)
    stream_result = {}

    def long_stream():
        try:
            stream_result["status"] = call(port, "POST", "/v1/chat/completions", chat(alias, 4000, True), timeout=120)[0]
        except OSError as error:
            stream_result["status"] = repr(error)

    thread = threading.Thread(target=long_stream)
    thread.start()
    time.sleep(1.5)
    call(before["port"], "POST", "/fake/stall", {}, timeout=5)
    after = incident_after(before["instance"], "stall", 60)
    check("a stall is detected and the model restarted", bool(after), after and after["incident"]["message"])
    thread.join(30)

    # The fourth failure within 15 minutes is not restarted.
    before = instance(port)
    try:
        call(before["port"], "POST", "/fake/crash", {}, timeout=5)
    except OSError:
        pass
    failed = wait_for(lambda: (lambda i: i if i["state"] == "failed" and i["instance"] == before["instance"] else None)(
        instance(port)), 30)
    time.sleep(2)
    now = instance(port)
    check("the restart limit holds (3 in 15 minutes)",
          bool(failed) and now["state"] == "failed" and not now["incident"]["restarted"], now["incident"]["action"])
    status, _ = call(port, "POST", "/v1/chat/completions", chat(alias, 8))
    check("the next request loads it again", status == 200, str(status))
    # Incidents carry their kind; notices (an idle server not answering) have none.
    kinds = [e["fields"]["kind"] for e in events(port)
             if e["category"] == "watchdog" and isinstance(e.get("fields"), dict) and "kind" in e["fields"]]
    check("every incident is in the event log", kinds == ["crash", "hang", "stall", "crash"], str(kinds))

    # A load that does not fit in VRAM: the server's own message, and suggestions.
    kv = ("automatic KV capacity requires 11215218944 bytes total (10141477120 bytes minimum Engine runtime "
          "reservation + 1073741824 bytes automatic headroom), but only 9738125312 bytes are available after weights")
    failing = [a for a in base_args] + ["--fake-load-ms", "300", "--fake-fail-load", kv]
    call(port, "PUT", f"/launcher/api/models/{key}/config", {"args": failing})
    call(port, "POST", "/launcher/api/unload", {})
    wait_for(lambda: instance(port)["state"] in ("idle", "failed"), 30)
    status, body = call(port, "POST", "/v1/chat/completions", chat(alias, 8))
    # The suggestions come from the server's plan, made after the refusal (a second or so).
    incident = wait_for(lambda: (lambda i: i if i and i.get("kind") == "vram" and i.get("phase") == "load" else None)(
        instance(port)["incident"]), 15, 0.2) or {}
    suggestions = incident.get("suggestions", [])
    check("a VRAM shortfall refuses the request with the server's reason", status == 503 and "KV capacity" in json.dumps(body),
          str(body)[:160])
    check("...and is reported with suggested changes", len(suggestions) >= 2, "; ".join(s["title"] for s in suggestions))
    lower = next((s for s in suggestions if s["args"] and "--max-context" in s["args"]), None)
    check("one suggestion lowers the context to what the server's plan fits",
          bool(lower) and lower is suggestions[0] and "225,280" in lower["title"], lower and lower["title"])

    # infernix-serve binds its port exclusively. An automatic port another program takes in the
    # moment before the server binds it (the stand-in refuses its port once) is retried, unreported.
    marker = data / "port-taken-once"
    call(port, "PUT", f"/launcher/api/models/{key}/config",
         {"args": base_args + ["--fake-load-ms", "300"],
          "env": [{"name": "INFERNIX_FAKE_BIND_FAIL_ONCE", "value": str(marker)}]})
    call(port, "POST", "/launcher/api/unload", {})
    wait_for(lambda: instance(port)["state"] in ("idle", "failed"), 30)
    status, _ = call(port, "POST", "/v1/chat/completions", chat(alias, 8))
    retried = [e["message"] for e in events(port) if e["category"] == "model" and "was taken by another program" in e["message"]]
    now = instance(port)
    check("an automatic port taken before the server bound it is retried on another, unreported",
          status == 200 and marker.exists() and len(retried) == 1 and now["state"] == "ready" and not now["incident"],
          retried[-1] if retried else f"{status}, no retry")
    # A fixed port another program holds: refused with the server's words and advice.
    holder = socket.socket()
    if WINDOWS:
        holder.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    holder.bind(("127.0.0.1", 0))
    holder.listen(1)
    call(port, "PUT", f"/launcher/api/models/{key}/config", {"args": base_args + ["--fake-load-ms", "300"], "env": []})
    call(port, "PUT", "/launcher/api/settings", {"child_port": holder.getsockname()[1]})
    call(port, "POST", "/launcher/api/unload", {})
    wait_for(lambda: instance(port)["state"] in ("idle", "failed"), 30)
    status, body = call(port, "POST", "/v1/chat/completions", chat(alias, 8))
    incident = wait_for(lambda: (lambda i: i if i and i.get("kind") == "port" else None)(instance(port)["incident"]),
                        10, 0.2) or {}
    titles = [s["title"] for s in incident.get("suggestions", [])]
    check("a fixed port another program holds is refused with the server's reason and advice",
          status == 503 and "port is in use" in json.dumps(body)
          and titles[:1] == ["Let the launcher choose the server's port"], f"{status} {titles}")
    call(port, "PUT", "/launcher/api/settings", {"child_port": 0})
    status, _ = call(port, "POST", "/v1/chat/completions", chat(alias, 8))
    check("...and back on the automatic port it loads at once (changed settings end the hold)", status == 200, str(status))
    holder.close()

    # RAM floor: the launcher ends a load that eats the computer's RAM.
    if not skip_ram:
        _, telemetry = call(port, "GET", "/launcher/api/telemetry")
        samples = telemetry.get("samples", []) if isinstance(telemetry, dict) else []
        available_mib = samples[-1]["ram"]["available"] >> 20 if samples else 0
        if available_mib > 16384:
            floor = available_mib - 2048
            call(port, "PUT", "/launcher/api/settings", {"load_ram_floor_mib": floor})
            hogging = [a for a in base_args] + ["--fake-load-ms", "15000", "--fake-hog-mib", "6144"]
            call(port, "PUT", f"/launcher/api/models/{key}/config", {"args": hogging})
            # A saved configuration applies at the next load.
            call(port, "POST", "/launcher/api/unload", {})
            wait_for(lambda: instance(port)["state"] in ("idle", "failed"), 30)
            status, body = call(port, "POST", "/v1/chat/completions", chat(alias, 8))
            incident = instance(port)["incident"] or {}
            check("the RAM floor ends a load and suggests changes",
                  status == 503 and incident.get("kind") == "ram" and bool(incident.get("suggestions")),
                  f"{status} {incident.get('message')}")
            call(port, "PUT", "/launcher/api/settings", {"load_ram_floor_mib": 2048})
        else:
            check("RAM floor (skipped: under 16 GiB free)", True)

    # Request contents, off by default.
    call(port, "PUT", f"/launcher/api/models/{key}/config", {"args": base_args + ["--fake-load-ms", "300"]})
    requests_file = Path(call(port, "GET", "/launcher/api/state")[1]["logs"]["requests_file"])
    check("no request contents by default", not requests_file.exists())
    call(port, "PUT", "/launcher/api/settings", {"log_bodies": True})
    status, _ = call(port, "POST", "/v1/chat/completions", chat(alias, 12, True))
    # The record is written as the exchange is released, just after the client's last byte.
    wait_for(lambda: requests_file.exists() and requests_file.stat().st_size > 0, 10, 0.2)
    record = json.loads(requests_file.read_text(encoding="utf-8").splitlines()[-1]) if requests_file.exists() else {}
    check("request contents when asked", status == 200 and record.get("request", {}).get("max_tokens") == 12
          and "data:" in record.get("response", ""), str(record)[:160])
    call(port, "PUT", "/launcher/api/settings", {"log_bodies": False})

    # The log folder moves at once.
    moved = data / "moved-logs"
    call(port, "PUT", "/launcher/api/settings", {"log_dir": str(moved)})
    call(port, "GET", "/v1/models")
    logs = call(port, "GET", "/launcher/api/state")[1]["logs"]
    check("the log folder can be changed", Path(logs["events_file"]).parent == moved and
          (moved / "infernix-launcher.log").exists(), logs["events_file"])

    code = launcher.stop()
    check("watchdog: an orderly stop", code == 0, str(code))
    text = (moved / "infernix-launcher.log").read_text(encoding="utf-8")
    check("the stop is in the event log", "shutting down" in text)


# ---------------------------------------------------------------------------------------------
# A console without a window (Windows)


def windowless(launcher: Launcher, port: int, data: Path) -> None:
    """Started by a program that gives it a console without a window, the launcher still shares its
    console with the server, so an unload stops the server in order instead of at the stop timeout."""
    model = first_model(port)
    key, alias = model["key"], model["config"]["alias"]
    call(port, "PUT", f"/launcher/api/models/{key}/config",
         {"args": list(model["config"]["default_args"]) + ["--fake-load-ms", "300"]})
    status, _ = call(port, "POST", "/v1/chat/completions", chat(alias, 8))
    check("windowless: a request loads the model", status == 200, str(status))
    started = time.monotonic()
    call(port, "POST", "/launcher/api/unload", {})
    idle = wait_for(lambda: instance(port)["state"] == "idle", 30, 0.1)
    seconds = time.monotonic() - started
    # The stand-in writes "server stopped" only when the stop signal reaches it (not when ended).
    check("windowless: an unload stops the server in order", bool(idle) and "server stopped" in serve_log(port),
          f"idle after {seconds:.1f} s (stop timeout 10 s)")
    code = launcher.stop()
    check("windowless: an orderly stop", code == 0, str(code))


def cutoff(launcher: Launcher, port: int, data: Path) -> None:
    """A server that crashes in the middle of an answer: the client's stream ends with the protocol's error event,
    not as a complete answer (its own group: the crash counts toward the watchdog's restart limit)."""
    model = first_model(port)
    key, alias = model["key"], model["config"]["alias"]
    call(port, "PUT", f"/launcher/api/models/{key}/config", {"args": list(model["config"]["default_args"]) +
                                                              ["--fake-load-ms", "500"]})
    status, _ = call(port, "POST", "/v1/chat/completions", chat(alias, 8))
    check("cutoff: a request loads the model", status == 200, str(status))
    before = instance(port)
    cut: dict = {}
    reader = threading.Thread(target=lambda: cut.update(
        result=stream(port, "/v1/chat/completions", chat(alias, 4000, True))), daemon=True)
    reader.start()
    time.sleep(2.0)  # the stream is under way (a token every 6.7 ms)
    try:
        call(before["port"], "POST", "/fake/crash", {}, timeout=5)
    except OSError:
        pass
    reader.join(60)
    status, _, _, pieces = cut.get("result", (0, "", 0.0, []))
    text = "".join(piece for _, piece in pieces)
    check("cutoff: a stream the server's crash cut off ends with an error event, not [DONE]",
          status == 200 and '"error"' in text and "ended the stream early" in text and "[DONE]" not in text,
          f"{status} ...{text[-200:]!r}")
    code = launcher.stop()
    check("cutoff: an orderly stop", code == 0, str(code))


def autoload(launcher: Launcher, port: int, data: Path) -> None:
    """With autoload on, the launcher loads the default model by itself once its first scan has found
    it, however long the server's --help takes before that scan (12 s here: a cold first start)."""
    model = first_model(port)
    key = model["key"]
    code = launcher.stop()
    launcher.kill()
    check("autoload: the first launcher stops", code == 0, str(code))
    memory = json.loads((data / "infernix-launcher.json").read_text(encoding="utf-8"))
    memory["settings"].update({"autoload": True, "default_model": key})
    (data / "infernix-launcher.json").write_text(json.dumps(memory), encoding="utf-8")
    os.environ["INFERNIX_FAKE_HELP_DELAY_MS"] = "12000"
    try:
        launcher.start()
    finally:
        del os.environ["INFERNIX_FAKE_HELP_DELAY_MS"]
    ready = wait_for(lambda: (lambda i: i if i["state"] == "ready" else None)(instance(port)), 60)
    failed = [e["message"] for e in events(port) if e.get("level") == "error"]
    check("autoload: the default model loads after a slow first scan",
          bool(ready) and ready["model_key"] == key and not failed,
          f"instance {ready}, errors {failed}")
    code = launcher.stop()
    check("autoload: an orderly stop", code == 0, str(code))


# ---------------------------------------------------------------------------------------------
# Ablation (the Ablation page's runs of tools/ablate, with ablate_stub.py in its place)


def make_stub_tool(root: Path) -> Path:
    """A tool folder: ablate_stub.py as ablate.py, and a requirements.txt naming a local wheel (Set up runs pip on it
    with no network)."""
    import zipfile
    tool = root / "ablate-tool"
    tool.mkdir(parents=True)
    shutil.copy(Path(__file__).with_name("ablate_stub.py"), tool / "ablate.py")
    wheel = root / "stubpkg-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("stubpkg/__init__.py", "VALUE = 1\n")
        archive.writestr("stubpkg-1.0.dist-info/METADATA", "Metadata-Version: 2.1\nName: stubpkg\nVersion: 1.0\n")
        archive.writestr("stubpkg-1.0.dist-info/WHEEL",
                         "Wheel-Version: 1.0\nGenerator: launcher_check\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        archive.writestr("stubpkg-1.0.dist-info/RECORD",
                         "stubpkg/__init__.py,,\nstubpkg-1.0.dist-info/METADATA,,\nstubpkg-1.0.dist-info/WHEEL,,\n"
                         "stubpkg-1.0.dist-info/RECORD,,\n")
    (tool / "requirements.txt").write_text(f"stubpkg @ {wheel.as_uri()}\n", encoding="utf-8")
    return tool


def process_alive(pid: int) -> bool:
    if WINDOWS:
        import ctypes
        kernel = ctypes.WinDLL("kernel32")
        handle = kernel.OpenProcess(0x1000 | 0x00100000, False, pid)  # QUERY_LIMITED_INFORMATION | SYNCHRONIZE
        if not handle:
            return False
        try:
            return kernel.WaitForSingleObject(handle, 0) != 0  # WAIT_OBJECT_0: it has ended
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    stat = Path(f"/proc/{pid}/stat")
    return not (stat.is_file() and stat.read_text().rsplit(")", 1)[-1].split()[0] == "Z")


def ablation(launcher: Launcher, port: int, data: Path, tool: Path) -> None:
    """The Ablation page's backend against a stand-in tool: Python and packages, prompt sets, a run that takes the GPU
    (the model is unloaded, loads are refused with the reason, the gate opens at the end), a sidecar that reaches the
    library, another trial of the run's front kept with choose, a refusal before anything is unloaded, stop and resume, End now with a tool that ignores the stop, the
    outcomes without a sidecar, the last run after a restart, and a launcher killed during a run (about two
    minutes)."""
    api = "/launcher/api/ablation"
    model = first_model(port)
    key, alias = model["key"], model["config"]["alias"]
    base_args = list(model["config"]["default_args"]) + ["--fake-load-ms", "300"]

    def configure(env: dict, extra: tuple[str, ...] = ()) -> None:
        status, body = call(port, "PUT", f"/launcher/api/models/{key}/config",
                            {"args": base_args + list(extra), "env": [{"name": k, "value": v} for k, v in env.items()]})
        if status != 200:
            raise RuntimeError(f"configure: {status} {body}")

    def state() -> dict:
        return call(port, "GET", api)[1]

    def job() -> dict | None:
        return state()["job"]

    def wait_job(states: set[str], timeout: float, trials: int = 0) -> dict | None:
        def ready():
            j = job()
            done = ((j or {}).get("progress") or {}).get("trials", {}).get("done", 0)
            return j if j and j["state"] in states and done >= trials else None
        return wait_for(ready, timeout, 0.2)

    def start(name: str, trials: int):
        return call(port, "POST", f"{api}/runs", {"config": key, "preset": "stub-set", "trials": trials, "name": name})

    def loaded() -> tuple[bool, str]:
        status, body = call(port, "POST", "/v1/chat/completions", chat(alias, 8))
        state_now = instance(port)["state"]
        return status == 200 and state_now == "ready", f"{status} {str(body)[:300]}, instance {state_now}"

    ended = {"done", "no_sidecar", "stopped", "failed"}
    with section("ablation: setup"):
        s = state()
        check("ablation: the launcher uses the tool --ablate-tool names",
              s["tool"]["found"] and Path(s["tool"]["dir"]) == tool, str(s["tool"]))
        status, s = call(port, "POST", f"{api}/python", {"path": sys.executable})
        found = (s or {}).get("python", {}).get("found") or {}
        check("ablation: the Python chosen on the page is the one used",
              status == 200 and found.get("version", "").startswith(f"{sys.version_info[0]}.{sys.version_info[1]}."),
              str(found))
        status, body = start("early", 10)
        check("ablation: a run before Set up is refused", status == 400 and "packages" in json.dumps(body), str(body))
        call(port, "POST", f"{api}/setup", {})
        s = wait_for(lambda: (lambda v: v if v["task"] and v["task"]["state"] != "running" else None)(state()), 180)
        check("ablation: Set up installs the requirements (pip) into the launcher's packages folder",
              bool(s) and s["task"]["state"] == "done" and s["packages"]["ready"] and
              (data / "ablation" / "packages" / "stubpkg" / "__init__.py").is_file(),
              str(s and s["task"]))
    with section("ablation: prompt sets"):
        s = state()
        listed = {p["name"]: p for p in s["catalogue"] or []}
        check("ablation: the tool's prompt sets are listed with their licences",
              "stub-set" in listed and listed["stub-set"]["sources"][1]["licence"] == "CC BY-NC 4.0", str(s["catalogue"]))
        status, body = call(port, "POST", f"{api}/prompts/no-such-set", {})
        check("ablation: an unknown prompt set is refused", status == 400, str(body))
        # A set fetched before the tool had a validation pair: listed as not fetched, so the page fetches it again.
        old = data / "ablation" / "prompts" / "stub-set"
        old.mkdir(parents=True, exist_ok=True)
        for part in ("harmful", "harmless", "test-harmful", "test-benign"):
            (old / f"{part}.txt").write_text("old prompt\n", encoding="utf-8")
        (old / "sets.json").write_text(json.dumps({"preset": "stub-set", "counts": {"harmful": 1}}), encoding="utf-8")
        check("ablation: a set fetched without a validation pair is not listed as fetched",
              "stub-set" not in state()["prompts"], str(state()["prompts"]))
        call(port, "POST", f"{api}/prompts/stub-set", {})
        s = wait_for(lambda: (lambda v: v if v["task"]["state"] != "running" else None)(state()), 60)
        check("ablation: a prompt set downloads into the prompts folder",
              bool(s) and s["task"]["state"] == "done" and s["prompts"].get("stub-set", {}).get("counts", {}).get(
                  "harmful") == 12, str(s and s["prompts"]))
    with section("ablation: a configuration the tool refuses"):
        configure({"STUB_REFUSE": "1"})
        check("ablation: the model loads", *loaded())
        status, body = start("refused", 10)
        check("ablation: init's refusal comes back before anything is unloaded",
              status == 400 and "the stub refuses it" in json.dumps(body) and instance(port)["state"] == "ready",
              f"{status} {body}")
    with section("ablation: a run makes a sidecar"):
        configure({"STUB_AUTO": "done", "STUB_TRIAL_S": "0.5"}, ("--ablate", str(data / "old" / "sidecar.json")))
        status, body = start("stub-cut", 20)
        check("ablation: the run starts", status == 200, f"{status} {body}")
        running = wait_job({"running"}, 30, trials=2)
        inst = instance(port)
        check("ablation: the run unloads the model and holds the GPU",
              bool(running) and inst["state"] in ("idle", "failed") and inst["ablation_holds_gpu"], str(inst))
        status, body = call(port, "POST", "/v1/chat/completions", chat(alias, 8))
        check("ablation: a request meanwhile is refused with the reason", status == 503 and "ablation run" in
              json.dumps(body), f"{status} {body}")
        status, body = call(port, "POST", f"/launcher/api/models/{key}/load", {})
        check("ablation: Load meanwhile is refused", status == 400 and "ablation run" in json.dumps(body), str(body))
        status, body = start("second", 10)
        check("ablation: a second run meanwhile is refused", status == 400 and "in progress" in json.dumps(body))
        run_dir = Path(running["run_dir"])
        setup = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        words = setup["argv"]
        check("ablation: init gets the configuration's command line without --ablate or the managed flags",
              Path(words[1]) == Path(model["path"]).resolve() and "--ablate" not in words and "--port" not in words
              and "--model-id" not in words and "--fake-load-ms" in words, str(words))
        check("ablation: the tool's version is the launcher's",
              setup["tool_version"].startswith("infernix-launcher "), setup["tool_version"])
        done = wait_job(ended, 60)
        exported = Path((done or {}).get("exported") or "")
        check("ablation: the run ends with the exported sidecar", bool(done) and done["state"] == "done" and
              exported.is_file(), str(done and {k: done[k] for k in ("state", "message", "exported")}))
        check("ablation: the tool's engines are gone", not list((run_dir / "engines").glob("*.json")))
        check("ablation: the GPU is free again", not instance(port)["ablation_holds_gpu"])
        sidecar = json.loads(exported.read_text(encoding="utf-8"))
        check("ablation: the prompt set's licences reach the sidecar's source",
              sidecar["generation"]["source"] == "stub prompts (MIT; CC BY-NC 4.0)", str(sidecar["generation"]))
        library = wait_for(lambda: (lambda l: l if any(c["name"] == "stub-cut" for c in l.get("sidecars", [])) else None)(
            call(port, "GET", "/launcher/api/library")[1]), 30)
        settings = call(port, "GET", "/launcher/api/settings")[1]
        check("ablation: the library lists the sidecar, its folder added to the model folders",
              bool(library) and any(Path(d) == data / "ablation" / "sidecars" for d in settings["model_dirs"]),
              str(settings["model_dirs"]))
        configure({"STUB_AUTO": "done"})
        check("ablation: the model loads again after the run", *loaded())
        status, body = start("stub-cut", 10)
        check("ablation: a run is refused a name a sidecar has", status == 400 and "exists" in json.dumps(body))
    with section("ablation: keep another trial of the front"):
        fitted = (job() or {}).get("progress") or {}
        table = fitted.get("front") or []
        front = [r for r in table if not r.get("off_front")]
        check("ablation: the run's progress shows its trade-off front, fewest refusals first, and a validated trial "
              "off it marked",
              len(front) == 4 and [r["refusals"] for r in front] == [0, 1, 2, 3] and
              [r["trial"] for r in table if r.get("off_front")] == [9], str(table))
        # The stub's validation numbers make trial 2 the pick on them, where the fit's numbers pick the front's last row.
        validation = fitted.get("validation") or {}
        check("ablation: the pick is made on the set's validation pair, the validated rows shown with their numbers",
              fitted.get("picked_on") == "validation" and validation.get("harmful") == "validate-harmful.txt" and
              (fitted.get("best") or {}).get("trial") == 2 and
              sum("validation" in r for r in table) == len(validation.get("trials") or []) == 4,
              str({k: fitted.get(k) for k in ("picked_on", "validation", "best")}))
        other = front[0]["trial"] if front else 3
        name = f"stub-cut-t{other}"
        status, body = call(port, "POST", f"{api}/choose", {"trial": other, "name": "stub-cut"})
        check("ablation: choose is refused a name a sidecar has", status == 400 and "exists" in json.dumps(body),
              f"{status} {body}")
        status, body = call(port, "POST", f"{api}/choose", {"trial": other, "name": name})
        check("ablation: choose starts", status == 200, f"{status} {body}")
        done = wait_job(ended, 60)
        exported = Path((done or {}).get("exported") or "")
        invocations = (run_dir / "invocations.txt").read_text(encoding="utf-8").splitlines()
        check("ablation: choose runs the tool's choose on the same run folder, the model unloaded first",
              bool(done) and done["state"] == "done" and done["choice"] == {"trial": other, "name": name} and
              Path(done["run_dir"]) == run_dir and invocations[-1].startswith(f"choose --trial {other} ") and
              instance(port)["state"] in ("idle", "failed"),
              str(done and {k: done[k] for k in ("state", "message", "choice", "run_dir")}))
        check("ablation: a set of matched pairs is fitted as one (--paired), choose takes no prompt lists",
              "--paired" in invocations[0] and "--paired" not in invocations[-1], f"{invocations[0]} | {invocations[-1]}")
        check("ablation: auto and choose get the set's validation and test pairs",
              all(f"--{flag}" in line for line in (invocations[0], invocations[-1])
                  for flag in ("validate-harmful", "validate-benign", "test-harmful", "test-benign")),
              f"{invocations[0]} | {invocations[-1]}")
        check("ablation: the kept trial is exported as a sidecar of its own",
              exported == data / "ablation" / "sidecars" / name / "sidecar.json" and exported.is_file() and
              json.loads(exported.read_text(encoding="utf-8"))["generation"]["notes"] == f"stub trial {other}",
              str(exported))
        check("ablation: choose's engines are gone and the GPU free",
              not list((run_dir / "engines").glob("*.json")) and not instance(port)["ablation_holds_gpu"])
        status, body = call(port, "POST", f"{api}/choose", {"trial": 99, "name": "stub-cut-t99"})
        failed = wait_job(ended, 60) if status == 200 else None
        check("ablation: a trial the study lacks fails with the tool's reason, the front still shown",
              bool(failed) and failed["state"] == "failed" and "did not complete" in failed["message"] and
              len([r for r in ((failed["progress"] or {}).get("front") or []) if not r.get("off_front")]) == 4,
              f"{status} {body if status != 200 else failed and failed['message']}")
    with section("ablation: stop and resume"):
        configure({"STUB_AUTO": "done", "STUB_TRIAL_S": "0.4"})
        status, body = start("resumed", 15)
        running = wait_job({"running"}, 30, trials=3)
        status, body = call(port, "POST", f"{api}/choose", {"trial": 0, "name": "meanwhile"})
        check("ablation: choose while a run is in progress is refused", status == 400 and "in progress" in
              json.dumps(body), f"{status} {body}")
        call(port, "POST", f"{api}/stop", {})
        stopped = wait_job(ended, 60)
        run_dir = Path(running["run_dir"]) if running else data
        first = int((run_dir / "trials-done.txt").read_text()) if (run_dir / "trials-done.txt").is_file() else -1
        check("ablation: Stop ends the run in order, its engine stopped, the GPU let go",
              bool(stopped) and stopped["state"] == "stopped" and 3 <= first < 15 and
              not list((run_dir / "engines").glob("*.json")) and not instance(port)["ablation_holds_gpu"],
              f"{stopped and stopped['state']}, {first} trials")
        status, body = start("resumed", 15)
        done = wait_job(ended, 60)
        invocations = (run_dir / "invocations.txt").read_text(encoding="utf-8").splitlines()
        check("ablation: the same request continues the same run folder",
              bool(done) and done["state"] == "done" and Path(done["run_dir"]) == run_dir and len(invocations) == 2
              and int((run_dir / "trials-done.txt").read_text()) == 15, f"{done and done['state']}, {invocations}")
    with section("ablation: End now"):
        configure({"STUB_AUTO": "stubborn"})
        status, body = start("stubborn", 10)
        running = wait_job({"running"}, 30)
        run_dir = Path(running["run_dir"]) if running else data
        engines = wait_for(lambda: list((run_dir / "engines").glob("*.json")), 10, 0.2) or []
        pids = [json.loads(p.read_text(encoding="utf-8"))["pid"] for p in engines]
        call(port, "POST", f"{api}/stop", {})
        time.sleep(3)
        still = job()
        check("ablation: a tool that ignores the stop keeps stopping", still["state"] == "stopping", still["state"])
        call(port, "POST", f"{api}/stop", {"now": True})
        stopped = wait_job(ended, 60)
        check("ablation: End now ends it and its engine at once",
              bool(stopped) and stopped["state"] == "stopped" and pids and not any(process_alive(p) for p in pids)
              and not instance(port)["ablation_holds_gpu"], f"{stopped and stopped['state']}, engines {pids}")
        warned = [e["message"] for e in events(port) if "did not end" in e.get("message", "")]
        check("ablation: the ended engine's record is cleared, with no false warning",
              not list((run_dir / "engines").glob("*.json")) and not warned, str(warned))
    with section("ablation: a second launcher on the same data folder"):
        # It refuses to start before it reads the folder, so the running run's engines live on (it once ended them as
        # leftovers of a run, then failed to bind the port).
        configure({"STUB_AUTO": "stubborn"})
        status, body = start("stubborn", 10)
        running = wait_job({"running"}, 30)
        run_dir = Path(running["run_dir"]) if running else data
        engines = wait_for(lambda: list((run_dir / "engines").glob("*.json")), 10, 0.2) or []
        pids = [json.loads(p.read_text(encoding="utf-8"))["pid"] for p in engines]
        second = subprocess.run([str(launcher.exe), "--port", str(port + 7), "--data-dir", str(data),
                                 "--serve", str(launcher.fake), "--no-browser"],
                                capture_output=True, text=True, errors="replace", timeout=60)
        check("ablation: a second launcher on the folder refuses to start",
              second.returncode != 0 and "another infernix-launcher uses the data folder" in second.stderr,
              f"exit {second.returncode}: {second.stderr.strip()[-300:]}")
        check("ablation: the running run and its engines live on",
              bool(pids) and all(process_alive(p) for p in pids) and job()["state"] == "running",
              f"engines {pids}, state {job()['state']}")
        call(port, "POST", f"{api}/stop", {"now": True})
        wait_job(ended, 60)
    with section("ablation: runs without a sidecar"):
        configure({"STUB_AUTO": "no_sidecar", "STUB_TRIAL_S": "0.05"})
        start("none", 10)
        none = wait_job(ended, 60)
        check("ablation: a run whose pick finds no cut says why", bool(none) and none["state"] == "no_sidecar" and
              "KL" in none["message"], str(none and none["message"]))
        configure({"STUB_AUTO": "fail"})
        start("broken", 10)
        failed = wait_job(ended, 60)
        check("ablation: a failed run says why", bool(failed) and failed["state"] == "failed" and
              "on purpose" in failed["message"], str(failed and failed["message"]))
    with section("ablation: a restart"):
        configure({"STUB_AUTO": "done", "STUB_TRIAL_S": "1"})
        start("restart", 100)
        running = wait_job({"running"}, 30, trials=1)
        run_dir = Path(running["run_dir"]) if running else data
        code = launcher.stop()
        launcher.kill()
        check("ablation: the launcher stops in order during a run", code == 0, str(code))
        launcher.start()
        # Before anything reads the page's state (a page left open across the restart): Set up finds the Python
        # chosen, instead of taking it for missing and downloading the portable one.
        status, body = call(port, "POST", f"{api}/setup", {})
        message = ((body or {}).get("task") or {}).get("message", "")
        check("ablation: Set up before the page's first read uses the Python chosen, no portable download",
              status == 200 and "portable" not in message and f"Python {sys.version_info[0]}." in message,
              f"{status} {message or body}")
        wait_for(lambda: (lambda v: v if v["task"] and v["task"]["state"] != "running" else None)(state()), 180)
        last = job()
        check("ablation: the last run is shown after a restart, stopped, its engine gone",
              bool(last) and last["run"]["name"] == "restart" and last["state"] == "stopped" and not last["active"] and
              not list((run_dir / "engines").glob("*.json")), str(last and {k: last[k] for k in ("state", "message")}))
    with section("ablation: a launcher killed during a run"):
        first_model(port)   # the restarted launcher's first scan (slow on a network or WSL drive)
        configure({"STUB_AUTO": "done", "STUB_TRIAL_S": "1"})
        start("killed", 100)
        running = wait_job({"running"}, 30, trials=1)
        run_dir = Path(running["run_dir"]) if running else data
        engines = wait_for(lambda: list((run_dir / "engines").glob("*.json")), 10, 0.2) or []
        pids = [json.loads(p.read_text(encoding="utf-8"))["pid"] for p in engines]
        launcher.kill()
        if WINDOWS:   # its job object ends everything the launcher started, the tool's engines among them
            gone = wait_for(lambda: not any(process_alive(p) for p in pids), 15, 0.2)
            check("ablation: the tool's engine ends with a killed launcher", bool(pids) and bool(gone), str(pids))
        launcher.start()
        last = job()
        check("ablation: the next launcher shows the run stopped and leaves no engine of it",
              bool(last) and last["run"]["name"] == "killed" and last["state"] == "stopped" and bool(pids) and
              not any(process_alive(p) for p in pids), str(last and {k: last[k] for k in ("state", "message")}))
        code = launcher.stop()
        check("ablation: an orderly stop", code == 0, str(code))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launcher", required=True)
    parser.add_argument("--fake", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--port", type=int, default=1238)
    parser.add_argument("--groups", default="features,watchdog,windowless,autoload,cutoff,ablation")
    parser.add_argument("--skip-ram", action="store_true")
    parser.add_argument("--keep", action="store_true", help="keep the temporary folders")
    options = parser.parse_args()
    groups = [g.strip() for g in options.groups.split(",") if g.strip()]
    if "features" in groups:
        run_group("features", options, {"load_timeout_s": 60, "drain_timeout_s": 5, "stop_timeout_s": 10}, features)
    if "watchdog" in groups:
        run_group("watchdog", options, {"hang_timeout_s": 10, "stall_timeout_s": 10, "load_timeout_s": 60,
                                        "drain_timeout_s": 5, "stop_timeout_s": 10},
                  lambda launcher, port, data: watchdog(launcher, port, data, options.skip_ram))
    if "windowless" in groups and WINDOWS:
        run_group("windowless", options, {"load_timeout_s": 60, "stop_timeout_s": 10}, windowless, windowless=True)
    if "autoload" in groups:
        run_group("autoload", options, {"load_timeout_s": 60, "stop_timeout_s": 10}, autoload)
    if "cutoff" in groups:
        run_group("cutoff", options, {"load_timeout_s": 60, "stop_timeout_s": 10}, cutoff)
    if "ablation" in groups:
        tool_root = Path(tempfile.mkdtemp(prefix="infernix-launcher-ablate-tool-"))
        try:
            tool = make_stub_tool(tool_root)
            run_group("ablation", options, {"load_timeout_s": 60, "stop_timeout_s": 10},
                      lambda launcher, port, data: ablation(launcher, port, data, tool),
                      extra=("--ablate-tool", str(tool)))
        finally:
            if not options.keep:
                shutil.rmtree(tool_root, ignore_errors=True)
    failed = [r for r in results if not r[1]]
    log(f"SUMMARY: {len(results) - len(failed)} passed, {len(failed)} failed")
    for name, _, detail in failed:
        log(f"  FAILED {name}: {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
