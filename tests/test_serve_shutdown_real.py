"""infernix-serve stops cleanly while a client streams (candidates YJ-04; NInfer fixed a shutdown that joined its
HTTP workers while they waited on generation). A stop sent while a request is resident and its client stays
connected (Ctrl+Break on Windows, SIGTERM elsewhere: the immediate stop of docs/serving.md, "Stop the server") must
end that stream with an error event, save the --prefix-cache-file, and exit with status 0 within a bound.

    python tests/test_serve_shutdown_real.py SERVE_EXE    (INFERNIX_TEST_ARTIFACT names the artifact; exit 77 skips)
"""

from __future__ import annotations

import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

READY_SECONDS = 600     # the artifact's load
STREAM_SECONDS = 180    # until the request's first streamed output
STOP_SECONDS = 30       # from the stop to the process's exit, the prefix cache save included


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def served_model(port: int, server: subprocess.Popen) -> str:
    """The served model id once /v1/models answers; fails if the server exits or does not get ready."""
    deadline = time.monotonic() + READY_SECONDS
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise SystemExit(f"FAIL: infernix-serve exited during startup with status {server.returncode}")
        try:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            connection.request("GET", "/v1/models")
            response = connection.getresponse()
            if response.status == 200:
                return json.loads(response.read())["data"][0]["id"]
        except (OSError, http.client.HTTPException, ValueError, KeyError, IndexError):
            pass
        time.sleep(1)
    raise SystemExit(f"FAIL: infernix-serve was not ready within {READY_SECONDS} s")


def stop(server: subprocess.Popen) -> None:
    if os.name == "nt":
        os.kill(server.pid, signal.CTRL_BREAK_EVENT)  # the server is its own process group's leader
    else:
        server.send_signal(signal.SIGTERM)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    artifact = os.environ.get("INFERNIX_TEST_ARTIFACT", "")
    if not artifact:
        print("SKIP: set INFERNIX_TEST_ARTIFACT")
        return 77
    port = free_port()
    failures = []
    with tempfile.TemporaryDirectory(prefix="infernix_shutdown_") as scratch:
        cache = Path(scratch) / "prefix.cache"
        log_path = Path(scratch) / "serve.log"
        args = [sys.argv[1], artifact, "--host", "127.0.0.1", "--port", str(port), "--max-context", "8192",
                "--max-concurrency", "1", "--host-context-mib", "1024", "--prefix-cache-file", str(cache)]
        with log_path.open("w", encoding="utf-8") as log:
            server = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT,
                                      creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0)
            try:
                model = served_model(port, server)
                # A long streamed request: resident, and its client reading, when the stop arrives.
                body = {"model": model, "stream": True, "max_tokens": 4000, "temperature": 0,
                        "messages": [{"role": "user", "content": "Count from 1 to 3000, one number per line."}]}
                connection = http.client.HTTPConnection("127.0.0.1", port, timeout=STREAM_SECONDS)
                connection.request("POST", "/v1/chat/completions", json.dumps(body),
                                   {"Content-Type": "application/json"})
                response = connection.getresponse()
                if response.status != 200:
                    raise SystemExit(f"FAIL: the streamed request was answered {response.status}")
                while True:
                    line = response.readline()
                    if not line:
                        raise SystemExit("FAIL: the stream ended before its first output")
                    if line.startswith(b"data: {"):
                        break
                started = time.monotonic()
                stop(server)
                rest = []
                while True:
                    line = response.readline()
                    if not line:
                        break
                    rest.append(line.decode("utf-8", "replace").strip())
                status = server.wait(timeout=STOP_SECONDS)
                seconds = time.monotonic() - started
            except subprocess.TimeoutExpired:
                failures.append(f"infernix-serve did not exit within {STOP_SECONDS} s of the stop")
                status, seconds, rest = None, None, []
            finally:
                if server.poll() is None:
                    server.kill()
                    server.wait()
        events = [line[len("data: "):] for line in rest if line.startswith("data: ")]
        if not any('"error"' in event for event in events):
            failures.append(f"the stream did not end with an error event (its last lines: {rest[-3:]})")
        if status is not None and status != 0:
            failures.append(f"infernix-serve exited with status {status}, not 0")
        if not cache.is_file() or cache.stat().st_size == 0:
            failures.append("the stop did not save the prefix cache file")
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        if "server stopped" not in log_text:
            failures.append("the log has no `server stopped` line")
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        if failures:
            print(log_text[-4000:], file=sys.stderr)
            return 1
    print(f"OK: the stop ended the stream with an error event, saved the prefix cache and exited 0 in {seconds:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
