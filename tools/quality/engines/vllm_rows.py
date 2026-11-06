"""vLLM's logits on a base file's chunks, streamed to standard output for infernix-certify llama-rows. Runs in the Python
environment that holds vLLM (on this machine: WSL), offline and in-process.

  python vllm_rows.py --base BASE --model DIR [--chunks N] [--max-model-len 1024] [--decode [--mtp K]]
                      [--engine-kwargs JSON]

Prompt path (default): each chunk is one request of its token ids with prompt log-probabilities on, so vLLM computes the
logits of every prompt position on its own prefill path.

Decode path (--decode): each chunk's first n_ctx/2 + 1 tokens are the prompt and the remaining n_ctx/2 - 1 tokens are
generated greedily, one request at a time, with the chunk's own tokens forced: at every target head row the logit of the
text's next token is raised to 16384 after the row is read, so greedy selection takes the text's token and speculative
verification (--mtp K: the checkpoint's MTP drafter, K drafts a round) accepts a draft exactly when it is the text's token.
The prompt's last row comes from the prompt path, every other row from a decode step or a verify window; a row computed
after a rejected draft is rewritten by the window that recomputes its position, so the last row read for a position is
the one its token was emitted from.

Either way the logits are the target model's compute_logits output (before any log-softmax or processor), BF16 where the
model dtype is BF16; the base's rows (positions n_ctx/2 .. n_ctx-2) go to standard output as binary32, chunk by chunk.
Everything else is vLLM's defaults for the checkpoint (kv-cache dtype auto, its kernels); --enforce-eager (vLLM's own advice
for NVFP4 on one RTX 5090) keeps the model's Python calls visible and changes no arithmetic. The file needs only numpy and
vLLM, so it can run outside the Infernix tree (python /mnt/e/.../vllm_rows.py).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # the model runs in this process, so its call is visible

RAISED = 16384.0


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
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--mtp", type=int, default=0, help="decode: the MTP drafter's drafts a round (0: no drafting)")
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
              "enable_prefix_caching": False, "max_logprobs": 1}
    if args.decode:
        kwargs["max_num_seqs"] = 1
        if args.mtp > 0:
            kwargs["speculative_config"] = {"method": "mtp", "num_speculative_tokens": args.mtp}
    kwargs.update(json.loads(args.engine_kwargs))
    llm = LLM(**kwargs)
    first = context // 2

    captured: list[torch.Tensor] = []
    state = {"active": False, "chunk": None, "positions": None, "rows": {}}

    def hook(model):
        original_logits = model.compute_logits
        original_forward = model.forward

        def forward(*a, **k):
            positions = k.get("positions", a[1] if len(a) > 1 else None)
            if state["active"] and positions is not None:
                # M-RoPE models pass (3, tokens) positions; text positions are equal on every axis.
                flat = positions[0] if positions.dim() == 2 else positions
                state["positions"] = flat.detach().reshape(-1).to("cpu")
            return original_forward(*a, **k)

        def compute_logits(*a, **k):
            logits = original_logits(*a, **k)
            if logits is None:
                return logits
            if not args.decode:
                captured.append(logits.detach().to("cpu"))
                return logits
            if not state["active"]:
                return logits
            positions = state["positions"]
            n = logits.shape[0]
            if positions is None or positions.numel() < n:
                raise RuntimeError(f"{n} logits rows but positions {None if positions is None else positions.numel()}")
            # One request: the rows are the call's last n scheduled tokens (the prompt's last token; a decode step;
            # a verify window's anchor and drafts).
            row_positions = positions[positions.numel() - n:].tolist()
            text = state["chunk"]
            host = logits[:, :vocab].detach().to(torch.float32).to("cpu").numpy()
            for r, p in enumerate(row_positions):
                state["rows"][int(p)] = host[r].copy()
            raised = logits.clone()
            for r, p in enumerate(row_positions):
                if p + 1 < context:
                    raised[r, int(text[p + 1])] = RAISED
            return raised

        model.forward = forward
        model.compute_logits = compute_logits
        return True

    if not all(llm.apply_model(hook)):
        raise SystemExit("could not reach the model's compute_logits")
    for c in range(chunks):
        if args.decode:
            state.update(active=True, chunk=tokens[c], positions=None, rows={})
            params = SamplingParams(max_tokens=context - 1 - first, temperature=0.0, ignore_eos=True,
                                    detokenize=False)
            result = llm.generate([TokensPrompt(prompt_token_ids=[int(t) for t in tokens[c][: first + 1]])], params,
                                  use_tqdm=False)
            state["active"] = False
            generated = list(result[0].outputs[0].token_ids)
            if generated != [int(t) for t in tokens[c][first + 1:]]:
                raise SystemExit(f"chunk {c}: generated {len(generated)} tokens that are not the chunk's")
            missing = [p for p in range(first, context - 1) if p not in state["rows"]]
            if missing:
                raise SystemExit(f"chunk {c}: no logits for positions {missing[:5]}")
            scored = np.stack([state["rows"][p] for p in range(first, context - 1)])
        else:
            captured.clear()
            params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
            llm.generate([TokensPrompt(prompt_token_ids=[int(t) for t in tokens[c]])], params, use_tqdm=False)
            # The prompt's multi-row calls hold positions 0 .. n_ctx-1 in order (the whole prompt, the sampled last
            # position included, in one call); a one-row call (a separately sampled position) is not a prompt row.
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
