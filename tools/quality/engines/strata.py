"""Strata's logits on a base file's chunks, streamed to standard output for infernix-certify llama-rows.

  python -m tools.quality.engines.strata --base BASE --config STRATA.json --exe PATCHED_STRATA.exe --work DIR
         [--chunks N] [--env K=V ...]

Strata computes vocabulary logits only in its decode (verify) windows; its batched prompt path computes none. Each
chunk is therefore one request whose tokens are all read through the decode windows (`--short-read` at least the
chunk), as Strata's own quality tool (tools/quant_quality.py, STRATA_LOGPOS) reads them, and the base's rows
(positions n_ctx/2 .. n_ctx-2) are taken from the window rows Strata dumps. The dump needs a Strata built with
STRATA_LOGPOS_ROWS (src/core/verify.cpp, Verifier::window_logprobs: each row as int64 position, int64 vocabulary
and the binary32 logits); everything else is the CONFIG's own engine arguments (the shipped configuration's
KV format, drafting, expert cache, route tail skip and every default), with --max-context and --short-read set
for the chunk. Rows go to standard output as binary32, rows_per_chunk x vocabulary per chunk, in base order.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from tools.quality.engines.base_file import read_header

# Arguments the adapter sets (value-taking) or that do not belong to a scoring run (flags).
DROP_VALUE = {"--max-context", "--short-read"}
DROP_FLAG = {"--serve", "--vision"}


def engine_args(config: dict, exe: str, context: int) -> list[str]:
    args, out, i = list(config["args"]), [], 0
    while i < len(args):
        if args[i] in DROP_FLAG:
            i += 1
        elif args[i] in DROP_VALUE:
            i += 2
        else:
            out.append(args[i])
            i += 1
    return [exe, "--serve", *out, "--max-context", str(context + 512), "--short-read", str(context + 64)]


def read_rows(path: Path, vocab: int) -> dict[int, np.ndarray]:
    """The dumped rows by position (each position once)."""
    data = np.fromfile(path, dtype=np.uint8) if path.exists() else np.zeros(0, np.uint8)
    record = 16 + 4 * vocab
    if data.size % record:
        raise ValueError(f"{path}: {data.size} bytes is not a whole number of {vocab}-token rows")
    rows: dict[int, np.ndarray] = {}
    for k in range(data.size // record):
        chunk = data[k * record:(k + 1) * record]
        position, row_vocab = (int(v) for v in chunk[:16].view("<i8"))
        if row_vocab != vocab:
            raise ValueError(f"{path}: a row of {row_vocab} tokens; the base has {vocab}")
        if position in rows:
            raise ValueError(f"{path}: position {position} dumped twice")
        rows[position] = chunk[16:].view("<f4")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="a Strata configuration (exe, args, cwd, lib_dirs)")
    parser.add_argument("--exe", required=True, help="a Strata built with STRATA_LOGPOS_ROWS")
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--chunks", type=int, default=0)
    parser.add_argument("--env", action="append", default=[], help="K=V for the engine's environment")
    args = parser.parse_args(argv)
    header = read_header(args.base)
    chunks = header.chunks if args.chunks == 0 else args.chunks
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    args.work.mkdir(parents=True, exist_ok=True)
    rows_path, logpos = args.work / "rows.bin", args.work / "logpos.tsv"
    env = dict(os.environ, STRATA_LOGPOS=str(logpos), STRATA_LOGPOS_ROWS=str(rows_path))
    if config.get("lib_dirs"):
        env["PATH"] = os.pathsep.join([*config["lib_dirs"], env.get("PATH", "")])
    for item in args.env:
        key, _, value = item.partition("=")
        env[key] = value
    command = engine_args(config, args.exe, header.context)
    print("strata: " + subprocess.list2cmdline(command), file=sys.stderr, flush=True)
    with (args.work / "strata.stderr.txt").open("w", encoding="utf-8") as err:
        engine = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True,
                                  cwd=config.get("cwd"), env=env, bufsize=1)
        for line in engine.stdout:
            if line.startswith("READY"):
                break
        else:
            raise SystemExit(f"Strata ended before READY (exit {engine.wait()}); see {args.work / 'strata.stderr.txt'}")
        out = sys.stdout.buffer
        wanted = range(header.first_row, header.context - 1)
        for c in range(chunks):
            rows_path.unlink(missing_ok=True)
            engine.stdin.write("GEN 1 " + ",".join(str(int(t)) for t in header.chunk(c)) + "\n")
            engine.stdin.flush()
            done = None
            for line in engine.stdout:
                if line.startswith(("DONE", "ERR")):
                    done = line.strip()
                    break
            if done is None or not done.startswith("DONE"):
                engine.kill()
                raise SystemExit(f"chunk {c}: Strata answered {done!r}")
            rows = read_rows(rows_path, header.vocab)
            missing = [r for r in wanted if r not in rows]
            if missing:
                engine.kill()
                raise SystemExit(f"chunk {c}: no logits for positions {missing[:5]}... (the prompt took the batched path?)")
            for r in wanted:
                out.write(np.ascontiguousarray(rows[r], dtype="<f4").tobytes())
            out.flush()
            if (c + 1) % 20 == 0 or c + 1 == chunks:
                print(f"strata: {c + 1}/{chunks} chunks", file=sys.stderr, flush=True)
        engine.stdin.write("QUIT\n")
        engine.stdin.flush()
        try:
            engine.wait(timeout=600)
        except subprocess.TimeoutExpired:
            engine.kill()
    rows_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
