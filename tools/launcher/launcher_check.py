"""The launcher against the stand-in server (no GPU): its features and its watchdog.

python -I tools/launcher/launcher_check.py --launcher <infernix-launcher> --fake <infernix_launcher_fake_serve>
    --models <dir> [--groups features,watchdog,windowless]
Runs on Windows and Linux. <dir> holds at least one supported artifact (only its header is read;
nothing in it is changed). Each group starts its own launcher with its memory and logs in a fresh
temporary folder (printed; removed afterwards unless --keep).

features: model ids, named configurations, environment, script export and import, the fit plan,
          keep-alive streams while a model loads (OpenAI and Anthropic, a failed load, and the
          server's error answer after the load), model
          routes and the pin, prompt progress, request labels, usage, API keys and sign-in, chats,
          and a configuration's idle unload (about four minutes).
watchdog: crash, lock-up, stall, the restart limit, idle servers, loads that do not fit, the RAM
          floor (about 6 GiB of RAM for a few seconds; --skip-ram leaves it out), the event log and
          the request-content log (about three minutes).
windowless: (Windows) a launcher started on a console without a window stops its server in order.
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
    def __init__(self, exe: Path, fake: Path, data: Path, port: int, windowless: bool = False) -> None:
        self.exe, self.fake, self.data, self.port = exe, fake, data, port
        self.windowless = windowless  # Windows: a console without a window, as some programs start it
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
                                      "--serve", str(self.fake), "--no-browser"],
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


def run_group(name: str, options, settings: dict, body, windowless: bool = False) -> None:
    """Starts a launcher on a fresh data folder and runs `body(launcher, port)` against it."""
    data = Path(tempfile.mkdtemp(prefix=f"infernix-launcher-{name}-"))
    log(f"== {name}: data folder {data}")
    (data / "infernix-launcher.json").write_text(json.dumps({"schema": 1, "settings": {
        "model_dirs": [options.models], "open_browser": False, **settings}}), encoding="utf-8")
    launcher = Launcher(Path(options.launcher), Path(options.fake), data, options.port, windowless)
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
        ]
        unnamed = [want for extra, want in alone if not any(w.startswith(want) for w in warned(extra))]
        check("the preview names each", not unnamed, "; ".join(unnamed))
        together = warned(["--ssd-context-gib", "64", "--ssd-context-dir", str(data), "--ssd-context-gib-per-hour", "4",
                           "--prefix-cache-file", str(data / "prefix.cache"), "--prefix-cache-save-mins", "30",
                           "--ngram-archive-mib", "64", "--ngram-native-sessions",
                           "--request-log-jsonl", str(data / "requests.jsonl"), "--request-log-max-mib", "8",
                           "--request-log-keep", "2", "--vision", "--vision-offload", "on"])
        check("...and none of them when both are given", not any(" needs " in w for w in together), "; ".join(together))
        absent = warned(["--ssd-context-gib", "64", "--ssd-context-dir", str(data / "no-such-folder")])
        check("...and an SSD prefix cache folder that does not exist",
              any(w.startswith("--ssd-context-dir: folder not found") for w in absent), "; ".join(absent))

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
        # The server answers a stream that waited for the load with a JSON error (it refuses compaction): the
        # error event carries the server's message, not only its status.
        call(port, "POST", "/launcher/api/unload", {})
        wait_for(lambda: instance(port)["state"] == "idle", 30)
        status, kind, headers_at, pieces = stream(port, "/v1/responses/compact",
                                                  {"model": "fake-main", "input": "hello", "stream": True})
        text = "".join(piece for _, piece in pieces)
        check("a server's error answer to a stream that waited for the load keeps its message",
              status == 200 and '"error"' in text and "compaction is not supported" in text, text[:200])
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launcher", required=True)
    parser.add_argument("--fake", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--port", type=int, default=1238)
    parser.add_argument("--groups", default="features,watchdog,windowless")
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
    failed = [r for r in results if not r[1]]
    log(f"SUMMARY: {len(results) - len(failed)} passed, {len(failed)} failed")
    for name, _, detail in failed:
        log(f"  FAILED {name}: {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
