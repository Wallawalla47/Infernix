# DFlash2 drafter for Inkling 2 Mini (Infernix)

incoai's [DFlash2 drafter for Inkling 2 Mini](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2) (Apache-2.0),
converted for [Infernix](https://github.com/Wallawalla47/Infernix) as a companion artifact. It is not a model by
itself: Infernix loads it beside this repository's Inkling 2 Mini artifact (`--spec dflash2 --draft-model`), and each
decoding round it drafts a block of 7 tokens in one pass, which the full model then verifies. A draft stands only
while it equals the model's own choice, so greedy and sampled output are the same as without it: only the speed
changes.

It is an option for **short prompts**: there it decodes faster than the artifact's own DSpark drafter. On long
contexts and in agentic use DSpark is faster, so `--spec auto` keeps DSpark unless `--draft-model` names this file.
It reads the main artifact's token embedding and output head (or its `--lm-head-draft` shortlist), so this file holds
only the drafter's own layers and its candidate selector.

| File | Bytes | What it is |
|---|---:|---|
| `Inkling-2-Mini-DFlash2-NVFP4-MLP-Infernix.infernix` | 515,118,080 | the drafter (an Infernix companion artifact) |
| `Inkling-2-Mini-DFlash2-NVFP4-MLP-Infernix.infernix.conversion.json` | 33,046 | conversion report: sources, methods and formats per object |
| `LICENSE`, `NOTICE` | | the Apache License 2.0; attribution and changes |

## Use

Infernix `main` at `6dbeb011d` or later (a release after `windows-2026.11.25`); older builds refuse
`--spec dflash2` on Inkling 2 Mini.

```text
hf download wallawalla47/Inkling-2-Mini-Infernix drafter/Inkling-2-Mini-DFlash2-NVFP4-MLP-Infernix.infernix --local-dir <dir>
infernix-serve Inkling-2-Mini-Infernix.infernix --max-context auto --kv-dtype int8 --spec dflash2 \
  --draft-model <dir>/drafter/Inkling-2-Mini-DFlash2-NVFP4-MLP-Infernix.infernix --draft-tokens auto --lm-head-draft
```

- `--spec dflash2 --draft-model FILE` loads the drafter beside the main artifact; Infernix refuses a drafter whose
  hidden width, vocabulary, layer count or loop visits differ from the model's.
- The artifact's DSpark drafter is then not loaded: none of its weights or KV pages take VRAM. At eight lanes, a
  240,000-token context, INT8 KV and `--lm-head-draft --vision --vision-offload on`, the KV pool holds 554,560 tokens
  against DSpark's 502,528 (+10.4 %). The drafter's rings take 40 MiB a lane.
- `--draft-tokens auto` verifies the number of the block's drafts their calibrated probabilities make fastest, a
  draft costing more on a longer context; up to eight requests decoding together draft in one call. `--lm-head-draft`
  drafts from the artifact's 65,536-token shortlist.
- A request sampling the whole vocabulary (`top_k: 0`) has its drafts drawn with the noise its own verification
  draws with, so they match the model's draws far more often; the output is unchanged.

## Quantization

Converted from incoai's checkpoint (revision `4f9088ea66c21f2b99ed657d1941c306b03e84d2`) with Infernix's recipe
`inkling2_mini_dflash2_nvfp4_mlp`: each layer's MLP gate/up and down in NVFP4 (FP8 scales per 16 weights, run with
16-bit activations); the attention query/key/value, the attention output and the target-feature projection in
`q8_g32_fp16` (8-bit codes with an FP16 scale per 32); the candidate selector, the dynamic convolutions and every norm
in BF16; the selector's codebooks cut to the 200,058 assigned tokens. The `artifact_id` is
`275baa9914ff417ca90f5ed7ae768a8a`.

An all-Q8 conversion (`inkling2_mini_dflash2`, 635 MB, not uploaded) took the same wall time over the agentic replay
below (this one -0.08 %: -0.21, -0.07 and +0.04 % by seed). On short prompts the all-Q8 one is a little faster: this one
accepts slightly fewer drafts (2,357 against 2,373 over eight prompts) and ran 0.9 % slower one at a time and 1.8 %
slower with four at once, 0.6 % faster with four questions on a ~30K-token prefix, and level with eight short
requests at once and on a 122K-token prefix. This one is recommended for its smaller file, 1.3 % more KV tokens and
2-3 % faster draft. The drafter only proposes tokens, so its precision changes acceptance and speed, never output.

## Performance

RTX 5090 on PCIe 5.0 x8, the configuration above (eight lanes, `--max-context 240000`, INT8 KV, `--lm-head-draft`,
`--draft-tokens auto`, vision offload, a 16,384-token thinking budget), output tok/s over each phase's wall time
(prefill and first token included), 512 tokens a request, the same text in every arm. Against the artifact's DSpark
drafter, measured with the all-Q8 conversion (2026-11-26, two runs each), which this one matches within 2 %:

| Phase | DSpark | DFlash2 | Change |
|---|---:|---:|---:|
| 8 short prompts one at a time, default sampling (top 20, T 1) | 278.0 | 319.4 | +14.9 % |
| 8 short prompts one at a time, greedy | 309.9 | 392.1 | +26.5 % |
| 8 short prompts one at a time, whole vocabulary | 281.2 | 355.5 | +26.4 % |
| 4 short requests at once | 499.2 | 578.5 | +15.9 % |
| 8 short requests at once | 683.5 | 814.4 | +19.2 % |
| 4 questions on a 122K-token prefix, one at a time | 180.0 | 168.2 | -6.6 % |
| 4 questions on a 122K-token prefix, at once | 317.0 | 294.2 | -7.2 % |

Since then the drafter's rounds got cheaper (fused kernels, a CUDA graph for its context writes, draft lengths that
follow the context: 1.6 % faster on the short prompts). In an agentic replay (an agent's main-session and subagent
turns, 130 requests a seed with prompts of 11K-134K tokens, three seeds, both arm orders; 2026-11-27) DFlash2 took
2.4 % more wall time than DSpark (+1.3 to +4.2 % by seed): level while one request decoded, 3.4 % slower while several
did, and slowest on the most predictable text, which DSpark's full-attention layer drafts better.

## How it works

DFlash2 (incoai's [blog](https://inco.ai/blog/dflash2/), [code](https://github.com/z-lab/dflash)) is a block-diffusion
drafter: five sliding-window layers (2,048 tokens) read the model's hidden states (the inputs of layer visits 5, 23 and
41 and the final normed state), draft a block of 8 (the last token and seven masks) in one pass with two-tap dynamic
convolutions, and a conditional selector picks each draft among the head's 16 best tokens. See Infernix's
[DFlash reference](https://github.com/Wallawalla47/Infernix/blob/main/docs/maintainer/dflash.md#inkling-2-mini) and the
[Inkling 2 Mini guide](https://github.com/Wallawalla47/Infernix/blob/main/docs/inkling2-mini.md#dflash2-drafting).

## License

Apache License 2.0, as the source checkpoint ([LICENSE](LICENSE)); see [NOTICE](NOTICE) for the attribution and the
changes made in this conversion. The model it drafts for is also Apache 2.0 (this repository's card).
