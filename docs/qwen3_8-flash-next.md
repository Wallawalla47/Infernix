# Qwen3.8-Flash-Next (`Qwen4ExpForCausalLM`)

NInfer runs NVIDIA's NVFP4 checkpoint of Qwen3.8-Flash-Next on one RTX 5090 (32 GB). The 24,576
routed experts (63 GiB) live in pinned host RAM. A VRAM expert cache holds the hot ones, and the
CPU computes part of each layer's misses. The 52 GB PLE n-gram table lives on an NVMe volume. The
design, measurements and open work are in the
[design document](maintainer/qwen3_8-flash-next-design.md) (§16.5, §19.2).

Text generation works through `ninfer`, `ninfer-serve` and `ninfer_bench`, with the model's MTP
drafter (`--spec mtp`) and n-gram copy proposals. Not yet supported: vision, CausalScoring
(perplexity), the prefix cache.

## Requirements

- **GPU.** An RTX 5090 (sm_120a). A **PCIe x16** link matters: expert misses cross PCIe, and an
  x8 link halves their bandwidth. Check with
  `nvidia-smi --query-gpu=pcie.link.width.current --format=csv`.
- **RAM.** About 96 GB. A run pins ~64.5 GiB, and the loader refuses to start if less than 8 GiB of
  RAM would remain free.
- **Disk.** ~80 GB for the artifact, plus a 52 GB n-gram volume, ideally on its own NVMe drive.

## Convert

Recipe B is recommended. It is recipe A, which imports every NVIDIA tensor bit-exactly, with the
BF16 dense projections and `lm_head` stored in `q8_g32_fp16`. Its measured quality cost is
+0.008 ± 0.010 nats (not significant), and it decodes ~22 % faster. Its MTP drafter is stored in
`q8_g32_fp16` with `q4_g64_fp16` routed experts. The drafter only proposes tokens, so its
precision changes speed, never output. `--proposal` adds the smaller draft head that
`--lm-head-draft` uses.

```text
python -m tools.convert --model <Qwen3.8-Flash-Next-NVFP4 dir> --recipe qwen3_8_flash_next_nvfp4_dense8 \
  --components text,vision,mtp --proposal --device cpu --out <dir>/qwen3_8_flash_next_nvfp4_dense8.ninfer \
  --ngram-out <nvme>/qwen3_8_flash_next.ngram
```

- **Recipe A.** Use `--recipe qwen3_8_flash_next_nvfp4` instead.
- **Second artifact of the same checkpoint.** It can share an existing n-gram volume: pass
  `--ngram-reuse <volume>` instead of `--ngram-out`.
- **Duration.** A conversion takes ~11 minutes from a warm disk cache; the MTP experts' MSE
  scale search accounts for several of them.

## Run

```text
ninfer <artifact>.ninfer --ngram-volume <volume>.ngram --kv-dtype int8 --max-context 16384 \
  --prompt "..."
ninfer-serve <artifact>.ninfer --ngram-volume <volume>.ngram --kv-dtype int8 --max-context 16384 \
  --prefill-chunk 4096
```

- `--kv-dtype` accepts `int8` (recommended) or `bf16`.
- **Context.** Measurements so far cover 4-8K contexts. Longer contexts work the same way, but
  their KV cache takes VRAM from the expert cache, and they have not been benchmarked yet.
- `--prefill-chunk 2048` or `4096` speeds up long prompts by 1.4-1.6×. Each costs ~2-5 % of
  decode speed, because the larger prefill workspace takes expert frames.
- `--spec mtp --draft-tokens 4 --lm-head-draft` drafts with the model's MTP layer.
  - `--draft-tokens` sets the maximum. Each round drafts the number of tokens the recent
    acceptance makes worthwhile, down to none on text it predicts poorly.
  - The drafter's 512 experts (1.34 GB), its workspace and the proposal head (178 MB) stay in
    VRAM, so the expert cache gets about 710 fewer frames.
  - Greedy output is the same as without it when one request runs at a time. With several
    lanes, speculative verification can round differently and flip a near-tie (design §19.2),
    and two lanes measured slower than one, so `--max-concurrency 1` is recommended with
    speculation.
- `--ngram-draft-tokens 7` verifies copy proposals. It works alone or beside MTP; a longer copy
  proposal replaces a round's MTP drafts. On code-editing prompts it accepts most drafts; on prose
  it finds none and costs nothing. See [ngram copy proposals](ngram.md).
- The engine starts six CPU worker threads for missed experts (up to eight per layer call). They
  spin while decoding. More
  workers measured slower: the CPU and the PCIe stage share the host's memory bandwidth.
- The expert cache fills all VRAM but 384 MiB (98.8 % used on the 5090).

## Performance

RTX 5090 on PCIe Gen5 x8, i9-13900K, DDR5-5800, recipe B, INT8 KV, one request. The expert cache
warms up over the first few hundred tokens of a session.

| Workload | tok/s |
|---|---:|
| `ninfer_bench` tg512, `--spec mtp --draft-tokens 4 --lm-head-draft` (warm cache) | ~139 |
| `ninfer_bench` tg512, plain decode (warm cache) | ~90 |
| Single CLI request, cold cache, plain | ~65 |
| Single CLI request, cold cache, MTP: code rewrite / prose story | ~84 / ~66 |
| Prompt, 4,096 tokens, `--prefill-chunk 4096` | ~670-690 |
| Prompt, 4,096 tokens, default chunk 1024 | ~410 |

MTP accepts most drafts on code and other predictable text, and fewer on free prose. The draft
length follows the measured acceptance, so prose mostly drafts one token and stays at least as
fast as plain decode (design §19.2).

**Quality.** Teacher-forced perplexity over three frozen texts (2,557 positions, INT8 KV): recipe
A 4.564, recipe B 4.600. For comparison, Strata with UD-Q4_K_XL and INT8 KV gives 4.864.
