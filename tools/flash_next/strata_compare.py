"""Teacher-forced quality of NInfer against Strata on the same token ids.

    python -m tools.flash_next.strata_compare --texts DIR [--names code,doc,chat]

DIR holds, per text NAME: NAME.ids (comma-separated ids), ninfer-NAME.bin (the --dump-logits output of
ninfer_qwen4_exp_forward_real_test: int32 vocabulary, int32 rows, FP32 rows) and strata.logpos (one
Strata session over the texts in order with STRATA_LOGPOS and STRATA_LOGPOS_TOPK=20; positions restart
at 0 for each request).

Per text: each engine's perplexity of the actual next tokens, the paired mean NLL difference
(NInfer - Strata) with its standard error, how often each engine's top token is the actual next
token, how often the two top tokens agree, and KL(Strata || NInfer) over Strata's top 20 plus one
bucket for the remaining mass (the method of Strata's docs/UNSLOTH_Q4.md).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


def read_logpos(path: Path) -> list[list[dict]]:
    """Strata rows grouped by request: each request's positions start again at 0."""
    groups: list[list[dict]] = []
    last = None
    for line in path.read_text().splitlines():
        f = line.split("\t")
        pos = int(f[0])
        if last is None or pos <= last:
            groups.append([])
        last = pos
        top = [(int(a), float(b)) for a, b in (item.split(":") for item in f[8:] if ":" in item)]
        groups[-1].append({"pos": pos, "target": int(f[1]), "logprob": float(f[2]), "top": int(f[3]), "topk": top})
    return groups


def ninfer_rows(path: Path):
    header = np.fromfile(path, dtype=np.int32, count=2)
    vocab, rows = int(header[0]), int(header[1])
    return np.memmap(path, dtype=np.float32, mode="r", offset=8, shape=(rows, vocab))


def log_softmax(row: np.ndarray) -> np.ndarray:
    x = row.astype(np.float64)
    m = x.max()
    return x - (m + math.log(np.exp(x - m).sum()))


def compare(ids: list[int], logits, strata: list[dict]) -> dict:
    nll_n, nll_s, kl, agree, top_n, top_s = [], [], [], 0, 0, 0
    for row in strata:
        pos, target = row["pos"], row["target"]
        if pos + 1 >= len(ids) or ids[pos + 1] != target:
            raise ValueError(f"Strata position {pos} targets {target}, the text has {ids[pos + 1]}")
        lp = log_softmax(np.asarray(logits[pos]))
        nll_n.append(-lp[target])
        nll_s.append(-row["logprob"])
        mine = int(lp.argmax())
        agree += mine == row["top"]
        top_n += mine == target
        top_s += row["top"] == target
        ids_k = [i for i, _ in row["topk"]]
        p = np.exp([v for _, v in row["topk"]])
        q = np.exp(lp[ids_k])
        p_rest, q_rest = max(1.0 - p.sum(), 1e-12), max(1.0 - q.sum(), 1e-12)
        kl.append(float((p * (np.log(p) - np.log(q))).sum() + p_rest * (math.log(p_rest) - math.log(q_rest))))
    n = len(strata)
    diff = np.array(nll_n) - np.array(nll_s)
    return {
        "positions": n,
        "ppl_ninfer": math.exp(np.mean(nll_n)),
        "ppl_strata": math.exp(np.mean(nll_s)),
        "nll_diff": float(diff.mean()),
        "nll_diff_se": float(diff.std(ddof=1) / math.sqrt(n)),
        "top1_next_ninfer": top_n / n,
        "top1_next_strata": top_s / n,
        "top1_agree": agree / n,
        "kl_mean": float(np.mean(kl)),
        "kl_median": float(np.median(kl)),
        "kl_p99": float(np.percentile(kl, 99)),
        "nll_ninfer": np.array(nll_n),
        "nll_strata": np.array(nll_s),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--texts", type=Path, required=True)
    parser.add_argument("--names", default="code,doc,chat")
    args = parser.parse_args()
    names = args.names.split(",")
    groups = read_logpos(args.texts / "strata.logpos")
    if len(groups) != len(names):
        raise ValueError(f"strata.logpos holds {len(groups)} requests for {len(names)} texts")
    print(f"{'text':<6}{'pos':>6}{'ppl NInfer':>12}{'ppl Strata':>12}{'dNLL (N-S)':>18}"
          f"{'top1 N':>8}{'top1 S':>8}{'agree':>7}{'KL mean':>9}{'KL p50':>8}{'KL p99':>8}")
    all_n, all_s = [], []
    for name, strata in zip(names, groups):
        ids = [int(x) for x in (args.texts / f"{name}.ids").read_text().split(",") if x]
        r = compare(ids, ninfer_rows(args.texts / f"ninfer-{name}.bin"), strata)
        all_n.append(r["nll_ninfer"])
        all_s.append(r["nll_strata"])
        print(f"{name:<6}{r['positions']:>6}{r['ppl_ninfer']:>12.4f}{r['ppl_strata']:>12.4f}"
              f"{r['nll_diff']:>+10.4f} ±{r['nll_diff_se']:.4f}{100 * r['top1_next_ninfer']:>7.1f}%"
              f"{100 * r['top1_next_strata']:>7.1f}%{100 * r['top1_agree']:>6.1f}%{r['kl_mean']:>9.4f}"
              f"{r['kl_median']:>8.4f}{r['kl_p99']:>8.3f}")
    n, s = np.concatenate(all_n), np.concatenate(all_s)
    d = n - s
    print(f"all {len(d)} positions: ppl NInfer {math.exp(n.mean()):.4f}, Strata {math.exp(s.mean()):.4f}, "
          f"mean dNLL {d.mean():+.4f} ± {d.std(ddof=1) / math.sqrt(len(d)):.4f} nats")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
