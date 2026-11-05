"""Cross-process digest check of the reproducible route (candidates WD-02; FlashInfer #8567's method).

The route promises a request the same output whatever else the process did (docs/cli.md, --reproducible). The
in-process real tests compare requests inside one Engine; this tool compares two processes whose launch-time state
differs. It starts infernix-serve twice on one artifact with --reproducible and a request log, each with its own
launch arguments (A, B), sends the same requests to both, and compares the SHA-256 digest of every request's
committed output token ids (request_done.result.generated_token_ids of --request-log-jsonl). Process B first serves
other requests (--b-warmup), so its prefix cache, captured graphs, expert placement and learned draft policies
differ, and B may also run with fewer expert frames (--vram-headroom-mib), another RAM tier, another expert state
and its requests at once (--b-concurrent, which needs --max-concurrency in B's arguments). Each process gets its
own copy of --expert-state (or `off`; --b-cold: B always `off`), so no sidecar of the artifact is written.

Requests file: a JSON list of {"label": str, "body": {...}}, each body an OpenAI Chat Completions request (give
sampled requests a "seed"); "stream" is forced off and "model" set to the server's own id (GET /v1/models). Exit status: 0 when every digest matches, 1 on a difference,
2 on a failure to run.

py -3.11 -I tools/reproducible_digest.py --serve <infernix-serve> --model <artifact> --requests <file.json>
    --out <dir> [--a "ARGS"] [--b "ARGS"] [--b-warmup <file.json>] [--b-concurrent] [--expert-state FILE]
    [--b-cold] [--port 18431] [--timeout 1800]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def read_requests(path: Path) -> list[dict]:
    requests = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(requests, list) or not requests:
        raise ValueError(f"{path}: expected a non-empty JSON list")
    labels = set()
    for item in requests:
        if not isinstance(item, dict) or not isinstance(item.get("label"), str) or not isinstance(item.get("body"), dict):
            raise ValueError(f"{path}: every request needs a string label and an object body")
        if item["label"] in labels:
            raise ValueError(f"{path}: label {item['label']!r} repeats")
        labels.add(item["label"])
    return requests


def digests_from_log(path: Path, labels: list[str]) -> dict[str, tuple[str, int]]:
    """Each label's (SHA-256 of its generated token ids, token count), from one server's request log. A label must
    finish exactly once."""
    out: dict[str, tuple[str, int]] = {}
    wanted = set(labels)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        label = (record.get("request") or {}).get("label")
        if record.get("event") == "request_done" and label in wanted:
            ids = (record.get("result") or {}).get("generated_token_ids")
            if not isinstance(ids, list):
                raise ValueError(f"{path}: request {label!r} has no generated_token_ids")
            if label in out:
                raise ValueError(f"{path}: request {label!r} finished twice")
            payload = json.dumps(ids, separators=(",", ":")).encode()
            out[label] = (hashlib.sha256(payload).hexdigest(), len(ids))
    missing = [label for label in labels if label not in out]
    if missing:
        raise ValueError(f"{path}: no request_done for {missing}")
    return out


def compare(a: dict[str, tuple[str, int]], b: dict[str, tuple[str, int]], labels: list[str]) -> list[str]:
    """The labels whose digests differ."""
    return [label for label in labels if a[label][0] != b[label][0]]


def served_model(port: int) -> str:
    """The server's public model id (the requests' "model" is replaced by it)."""
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=30) as response:
        return json.loads(response.read())["data"][0]["id"]


def post(port: int, item: dict, timeout: float, model: str) -> None:
    body = dict(item["body"])
    body["stream"] = False
    body["model"] = model
    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                     data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json",
                                              "X-Infernix-Request-Label": item["label"]})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"{item['label']}: HTTP {response.status}")
        response.read()


def wait_ready(port: int, process: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"infernix-serve exited with {process.returncode} before it was ready")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(2)
    raise RuntimeError("infernix-serve was not ready in time")


def run_process(tag: str, args: argparse.Namespace, extra: str, port: int, warmup: list[dict],
                requests: list[dict], at_once: bool, cold: bool) -> Path:
    out = Path(args.out)
    log = out / f"{tag}.jsonl"
    log.unlink(missing_ok=True)
    state = "off"
    if args.expert_state and not cold:
        state = str(out / f"{tag}.expert-state")
        shutil.copyfile(args.expert_state, state)
    argv = [args.serve, args.model, "--reproducible", "--port", str(port), "--request-log-jsonl", str(log),
            "--expert-state", state, *extra.split()]
    with open(out / f"{tag}.serve.log", "w", encoding="utf-8") as stream:
        process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT)
        try:
            wait_ready(port, process, args.timeout)
            model = served_model(port)
            for item in warmup:
                post(port, item, args.timeout, model)
            if at_once:
                with concurrent.futures.ThreadPoolExecutor(len(requests)) as pool:
                    for future in [pool.submit(post, port, item, args.timeout, model) for item in requests]:
                        future.result()
            else:
                for item in requests:
                    post(port, item, args.timeout, model)
        finally:
            process.terminate()
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    return log


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--serve", required=True, help="the infernix-serve executable")
    ap.add_argument("--model", required=True, help="the artifact (its first part)")
    ap.add_argument("--requests", required=True, type=Path, help="the compared requests (JSON list)")
    ap.add_argument("--out", required=True, help="an existing directory for the logs")
    ap.add_argument("--a", default="", help="process A's further infernix-serve arguments (split at whitespace)")
    ap.add_argument("--b", default="", help="process B's further infernix-serve arguments (split at whitespace)")
    ap.add_argument("--b-warmup", type=Path, help="requests process B serves first (JSON list)")
    ap.add_argument("--b-concurrent", action="store_true", help="B sends the compared requests at once")
    ap.add_argument("--expert-state", help="a state file each process starts from (a copy); default off")
    ap.add_argument("--b-cold", action="store_true", help="process B starts without the expert state")
    ap.add_argument("--port", type=int, default=18431, help="the first of two ports (never 8080's server)")
    ap.add_argument("--timeout", type=float, default=1800.0, help="seconds for start-up and each request")
    args = ap.parse_args()
    if args.port in (8080, 8081):
        ap.error("--port must leave 8080 alone")
    try:
        requests = read_requests(args.requests)
        warmup = read_requests(args.b_warmup) if args.b_warmup else []
        labels = [item["label"] for item in requests]
        if set(labels) & {item["label"] for item in warmup}:
            raise ValueError("a warm-up label repeats a compared label")
        log_a = run_process("a", args, args.a, args.port, [], requests, False, False)
        log_b = run_process("b", args, args.b, args.port + 1, warmup, requests, args.b_concurrent, args.b_cold)
        a = digests_from_log(log_a, labels)
        b = digests_from_log(log_b, labels)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"reproducible_digest: {error}", file=sys.stderr)
        return 2
    differ = compare(a, b, labels)
    for label in labels:
        mark = "DIFFERENT" if label in differ else "same"
        print(f"{label:24s} {a[label][1]:5d} / {b[label][1]:5d} tokens  {a[label][0][:16]}  {b[label][0][:16]}  {mark}")
    print(f"{len(labels) - len(differ)} of {len(labels)} requests give the same digest in both processes")
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main())
