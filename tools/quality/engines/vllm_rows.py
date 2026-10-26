"""vLLM's logits on a base file's chunks (the prompt path), streamed to standard output for infernix-certify
llama-rows. Runs in the Python environment that holds vLLM (on this machine: WSL), offline and in-process.

  python vllm_rows.py --base BASE --model DIR [--chunks N] [--max-model-len 1024] [-- ENGINE_KWARGS_JSON]

Each chunk is one request of its token ids with prompt log-probabilities on, so vLLM computes the logits of every
prompt position on its own prefill path. The logits themselves (as the model's compute_logits returns them, before any
log-softmax or processor) are taken from that call, so the rows are vLLM's own, BF16 where the model dtype is BF16;
the base's rows (positions n_ctx/2 .. n_ctx-2) go to standard output as binary32, chunk by chunk. Everything else is
vLLM's defaults for the checkpoint (kv-cache dtype auto, its kernels); --enforce-eager is vLLM's own advice for NVFP4
on one RTX 5090 and changes no arithmetic. The file needs only numpy and vLLM, so it can run outside the Infernix
tree (python /mnt/e/.../vllm_rows.py).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # the model runs in this process, so its call is visible


def read_base(path: str):
    with open(path, "rb") as f:
        if f.read(8) != b"_logits_":
            raise SystemExit(f"{path} is not a llama.cpp _logits_ file")
        context, vocab, chunks = (int(v) for v in np.frombuffer(f.read(12), dtype="<i4"))
        tokens = np.frombuffer(f.read(4 * context * chunks), dtype="<i4").reshape(chunks, context)
    return context, vocab, chunks, tokens


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--chunks", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--engine-kwargs", default="{}", help="JSON: more vllm.LLM arguments")
    args = parser.parse_args(argv)
    context, vocab, total, tokens = read_base(args.base)
    chunks = total if args.chunks == 0 else args.chunks
    # The rows own standard output: vLLM and its libraries log to stdout, so file descriptor 1 (and sys.stdout) go
    # to stderr and the rows to a private copy of the original descriptor.
    sys.stdout.flush()
    out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    import torch
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    kwargs = {"model": args.model, "max_model_len": args.max_model_len, "enforce_eager": True,
              "enable_prefix_caching": False, "max_logprobs": 1, **json.loads(args.engine_kwargs)}
    llm = LLM(**kwargs)
    captured: list[torch.Tensor] = []

    def hook(model):
        original = model.compute_logits

        def compute_logits(*a, **k):
            logits = original(*a, **k)
            if logits is not None:
                captured.append(logits.detach().to("cpu"))
            return logits

        model.compute_logits = compute_logits
        return True

    if not all(llm.apply_model(hook)):
        raise SystemExit("could not reach the model's compute_logits")
    params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
    first = context // 2
    for c in range(chunks):
        captured.clear()
        llm.generate([TokensPrompt(prompt_token_ids=[int(t) for t in tokens[c]])], params, use_tqdm=False)
        # The prompt's multi-row calls hold positions 0 .. n_ctx-1 in order (vLLM 0.32 computes the whole prompt,
        # the sampled last position included, in one call); a one-row call (a separately sampled position) is not a
        # prompt row. Row i is position i either way.
        prompt = [t.reshape(-1, t.shape[-1]) for t in captured if t.reshape(-1, t.shape[-1]).shape[0] > 1]
        rows = torch.cat(prompt, dim=0) if prompt else torch.empty(0, 0)
        if rows.shape[0] not in (context - 1, context) or rows.shape[1] < vocab:
            raise SystemExit(f"chunk {c}: captured {tuple(rows.shape)} logits for a {context}-token prompt "
                             f"of a {vocab}-token vocabulary")
        scored = rows[first:context - 1, :vocab].to(torch.float32).numpy()
        out.write(np.ascontiguousarray(scored, dtype="<f4").tobytes())
        out.flush()
        if (c + 1) % 20 == 0 or c + 1 == chunks:
            print(f"vllm: {c + 1}/{chunks} chunks", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
