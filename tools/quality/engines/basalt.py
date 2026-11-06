"""Basalt's logits on a base file's chunks, streamed to standard output for infernix-certify llama-rows.

  python -m tools.quality.engines.basalt --base BASE --engine BIN --model FILE.basalt --work DIR [--chunks N]
         [--t 3] [--wsl] [-- ENGINE_ARGS...]

Basalt's own quality gate (tools/ppl_gate.py, include/basalt/program/ppl_gate.hpp) reads a llama.cpp base file: per
chunk, the first n_ctx/2 + 1 tokens as a prompt (batched prefill), the rest teacher-forced through verify windows of
--t tokens (its rows are byte-identical for T = 1, 3 and 7, docs/precision/features.md). This adapter runs that
protocol with the engine's own defaults (only what the measurement needs: --ppl-base and --ppl-out, and the
measurement's --spec 5 with adaptive swaps off and --max-context 4096, as ppl_gate.py sets them) and takes each
scored row's logits from BASALT_PPL_ROWS (a build with that dump: int64 position, int64 vocabulary, binary32
logits per row), so Basalt is scored by the same arithmetic as every other engine. Its own rows.tsv stays in --work as
a cross-check. --wsl runs the engine in WSL (Basalt's measured platform): BIN and the paths it receives are Linux paths,
the base and work paths are translated from Windows drive paths.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path, PureWindowsPath

import numpy as np

from tools.quality.engines.base_file import read_header
from tools.quality.engines.strata import read_rows


def wsl_path(path: Path | str) -> str:
    """A Windows drive path as WSL sees it (E:\\x\\y -> /mnt/e/x/y)."""
    p = PureWindowsPath(str(path))
    if not p.drive:
        return str(path).replace("\\", "/")
    return "/mnt/" + p.drive[0].lower() + "/" + "/".join(p.parts[1:])


def engine_command(engine: str, model: str, base: str, out: str, extra: list[str], wsl: bool, rows: str) -> list[str]:
    args = ["--serve", "--model", model, "--spec", "5", "--adapt-swaps", "0", "--peer-adapt-swaps", "0",
            "--max-context", "4096", "--ppl-base", base, "--ppl-out", out, *extra]
    if wsl:
        return ["wsl.exe", "-e", "env", f"BASALT_PPL_ROWS={rows}", engine, *args]
    return [engine, *args]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--engine", required=True, help="a Basalt built with BASALT_PPL_ROWS")
    parser.add_argument("--model", required=True, help="the .basalt pack")
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--chunks", type=int, default=0)
    parser.add_argument("--t", type=int, default=3, help="tokens per verify window (ppl_gate.py's default)")
    parser.add_argument("--wsl", action="store_true")
    parser.add_argument("--cwd", help="the engine's working directory (its repository: data/profiles)")
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="-- then engine arguments")
    args = parser.parse_args(argv)
    extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra
    header = read_header(args.base)
    chunks = header.chunks if args.chunks == 0 else args.chunks
    args.work.mkdir(parents=True, exist_ok=True)
    rows_path = args.work / "rows.bin"
    to = wsl_path if args.wsl else str
    command = engine_command(args.engine, args.model, to(args.base.resolve()), to((args.work / "rows.tsv").resolve()),
                             extra, args.wsl, to(rows_path.resolve()))
    env = dict(os.environ, BASALT_PPL_ROWS=str(rows_path))
    print("basalt: " + subprocess.list2cmdline(command), file=sys.stderr, flush=True)
    with (args.work / "basalt.stderr.txt").open("w", encoding="utf-8") as err:
        engine = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True,
                                  env=env, bufsize=1, cwd=args.cwd)
        for line in engine.stdout:
            if line.startswith("READY"):
                break
        else:
            raise SystemExit(f"Basalt ended before READY (exit {engine.wait()}); see {args.work / 'basalt.stderr.txt'}")
        out = sys.stdout.buffer
        first = header.first_row
        wanted = range(first, header.context - 1)
        for c in range(chunks):
            rows_path.unlink(missing_ok=True)
            ids = ",".join(str(int(t)) for t in header.chunk(c)[: first + 1])
            engine.stdin.write(f"GEN {header.context - 1 - first} ppl={c} ppl_t={args.t} {ids}\n")
            engine.stdin.flush()
            done = None
            for line in engine.stdout:
                if line.startswith(("DONE", "ERR")):
                    done = line.strip()
                    break
            if done is None or not done.startswith("DONE"):
                engine.kill()
                raise SystemExit(f"chunk {c}: Basalt answered {done!r}")
            rows = read_rows(rows_path, header.vocab)
            missing = [r for r in wanted if r not in rows]
            if missing:
                engine.kill()
                raise SystemExit(f"chunk {c}: no logits for positions {missing[:5]}")
            for r in wanted:
                out.write(np.ascontiguousarray(rows[r], dtype="<f4").tobytes())
            out.flush()
            if (c + 1) % 20 == 0 or c + 1 == chunks:
                print(f"basalt: {c + 1}/{chunks} chunks", file=sys.stderr, flush=True)
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
