# Basalt port assessment

**Basalt** (`jesdga95/basalt`, MIT, a stripped-down fork of Strata) is a local engine for
Qwen3.8-Flash-Next on sm_120, the same model and hardware class as Infernix. This reference records
which of its mechanisms were judged worth porting, which are rejected and which are parked behind a
product change, so they are not re-evaluated from scratch.

The assessment was made by Fedor Suchkov on the `kido5217/Infernix` fork (map
[#54](https://github.com/kido5217/Infernix/issues/54), October 2026). Five research tickets
([#55](https://github.com/kido5217/Infernix/issues/55)–[#59](https://github.com/kido5217/Infernix/issues/59))
read both trees and left notes on that fork's unmerged `research/basalt-*` branches. **The verdicts
come from code reading only:** Basalt's published numbers are claims from its own environment, and no
benchmark informed them. Where this repository's own reading or measurement differs, the
**Here** column says so.

## Port candidates

| Fork ticket | Candidate | Here |
|---|---|---|
| [#61](https://github.com/kido5217/Infernix/issues/61) | Expose the one-row MTP draft-accept threshold as `--spec-min-p` | **Adopted** (2026-10-14): `--spec-min-p P` on `infernix`, `infernix-serve` and `infernix_bench`, default 0.5 ([serving](../serving.md)); the Basalt idea of an artifact-carried default is not adopted, since the threshold depends on the workload more than on the artifact |
| — | The same draft-confidence gate on the Qwen3.5/3.8-27B MTP path | Built 2026-10-23, not yet measured: `--spec-min-p` is accepted on Qwen3.5 MTP (default 0). The draft step's argmax pass also yields each draft's probability (`ops::argmax_with_probability`); a one-row `adaptive-mtp` round verifies at the cut's width while the proposal depth stays the policy's (a gated proposal chain with a device limit), and the device gate (temporary `INFERNIX_TMP_QWEN35_MTP_GATE`) skips the proposal steps past the cut. Its threshold needs its own measurement, since 0.5 was tuned on Flash-Next |
| [#62](https://github.com/kido5217/Infernix/issues/62) | Learned per-round draft policy: n-gram vs MTP from per-match-bucket acceptance and measured width cost | Candidate. Infernix already has an online width policy (EWMA acceptance, measured width costs, forced probes); the gain is bucket granularity, and Basalt documents a 2-8 % loss only when the choice is mis-made |
| [#63](https://github.com/kido5217/Infernix/issues/63) | Request-time RAM-low abort with named remedies | **Not adopted.** Basalt's request reads stream through the page cache and its RAM use grows per prompt, so stopping a request frees memory. Infernix pins its host memory when the Program starts (expert arena, prefix-cache Host tier, Program buffers; `memory_plan.h`), its pageable parts are a bounded n-gram row cache and a 1 GiB growth budget, and it reserves its own working set; aborting a request would free almost nothing while another program held the RAM |
| [#64](https://github.com/kido5217/Infernix/issues/64) | SSD file-cache floor in the RAM expert plan | **Not applicable.** Landed on the fork (`ec65e88a`), but Infernix's SSD expert tier reads the artifact unbuffered on both platforms (`DirectReadQueue`: `FILE_FLAG_NO_BUFFERING` / `O_DIRECT`), so expert pages never enter the OS page cache; the floor would take RAM from the expert tier and add SSD reads. The OS mapped-file tier the floor serves is rejected in the [design record](qwen3_8-flash-next-design.md) (Strata's `--mmap-experts`: 7.54 vs 12.18 tok/s) |
| [#65](https://github.com/kido5217/Infernix/issues/65) | Single `--config` JSON, CLI > file > defaults, validated at startup | Candidate (usability only) |
| [#66](https://github.com/kido5217/Infernix/issues/66) | Warn when binding beyond loopback without `--api-key` | Candidate (tiny) |
| [#67](https://github.com/kido5217/Infernix/issues/67) | Converter-shipped expert profile seeding the warm-start pipeline | Candidate. Its motivation cites a −19 % miss reduction, which is a leave-one-out **upper bound** for a category-keyed seed, not a measured gain over the existing warm-start seed ([design record](qwen3_8-flash-next-design.md) §9.3) |
| [#68](https://github.com/kido5217/Infernix/issues/68) | 2 MiB alignment of streamed expert objects | **Not adopted.** Landed on the fork (`945d301d`) as a writer-only change; only each layer's bank start moves (records keep a 4 KiB pitch and are not 2 MiB multiples), and Infernix's reader neither maps the file nor issues reads across banks, so nothing in this tree uses the alignment. No gain was measured |
| [#69](https://github.com/kido5217/Infernix/issues/69) | Decode-path full-vocabulary KLD gate with per-row diff output | Partly present: `infernix-certify decode` writes per-position teacher-forced decode KL over the reference's top 256 tokens, top-1 flips and NLL change; full-vocabulary KL would only tighten the KL bound |
| [#70](https://github.com/kido5217/Infernix/issues/70) | Exact greedy-identity build gate for numerically neutral refactors | Present: The golden reproducible hashes (`tools/quality/iqs_e2e.py identity --golden`) and `infernix-certify verify` compare a build's outputs with the stored ones bit for bit |
| [#71](https://github.com/kido5217/Infernix/issues/71) | Hours-long mixed-protocol serve soak (offload and preemption routes) | Candidate (dev tooling) |
| [#72](https://github.com/kido5217/Infernix/issues/72) | Whole-model per-matmul precision map kept current with each change | Candidate (documentation practice) |
| [#73](https://github.com/kido5217/Infernix/issues/73) | Expert-cache-lent prefill chunk sizing (`--prefill auto`) | **Not needed** (2026-10-16). The fork qualified it on its GPU ([#76](https://github.com/kido5217/Infernix/issues/76): +5-10 % over a fixed 4,096 chunk at 16K). Here Flash-Next's exact route measured bit-identical across chunks and now defaults to a fixed 16,384-token chunk, +14-16 % at 32K-64K over 4,096 with decode unchanged ([design record](qwen3_8-flash-next-design.md) §19.3.24), which captures the gain without the lent io view and its lifetime hazards |

## Speed mechanisms (second review, 2026-10-18)

A review of Basalt at `6e53eee` read its decode, scheduling and kernel code for what explains its
speed. Its published decode figures run at a 400 W limit with an RTX 5060 Ti as a peer expert tier
(the two cards hold 71-83 % of the experts) on short prompts, with int8 (q8_1) activations into
every quantized matmul and every decode miss on a CPU path that rounds differently from the GPU.
It publishes no single-GPU IQ3_S decode figure.

| Mechanism | Here |
|---|---|
| Draft chain stops on the device at the draft cut (conditional graph nodes) | **Adopted**: one-row argmax MTP rounds skip the steps past `--spec-min-p`, decode +2.2 % on mixed-topic text ([design record](qwen3_8-flash-next-design.md) §19.3.27) |
| Small table uploads ordered ahead of expert copies | **Adopted** as `upload_pinned` for the KV page tables and grammar masks; no resolved speed change (§19.3.27) |
| Expert usage seeded from the prompt's routing | **Adopted** as LFRU count credit (256 × each expert's share of the prompt's routes, no copies): decode −1.8 % on mixed-topic text (§19.3.27, the design's option H8) |
| Two-stage Q6_K verify head (2-bit screen, exact rescore of the top rows) | Candidate as a gated fast path: −1.8 to −3.3 % per window there, softmax tail approximate; a converter-written 2-bit copy (no runtime repacking) |
| Every decode miss on the CPU, planned on the device | Not applicable: our CPU route is exact and compute-bound (every miss on the CPU measured −2.3 %); device-side planning is already ours (§19.3.20e/f) |
| int8 activations (q8_1) into dense and expert matmuls; int8 hyper-connection copies; 6-bit dense | Not adopted: our one- and two-limb activation fast paths and the Q8 hyper-connection variant failed the [acceptance policy](model-quality.md#6-numerical-acceptance-policy-and-fast-paths); 6-bit dense costs KL +0.0063 there |
| Relaxed drafting (keep a draft with target probability ≥ P) | Product decision: 0.022-0.089 total variation per token, beyond the fast-path budget; only an explicit per-request mode |
| `dense_tc` int8 tensor-core GEMV | Not needed: our Dense8 GEMVs run at 90-97 % of DRAM bandwidth |

## Parked: needs a product change or a later decision

- **Peer-GPU expert tier.** Expert slots on a second card computed with the same kernels, so rows
  stay bit-identical; it breaks the one-GPU product contract.
- **Self-distilled per-pack MTP drafter.** The engine already loads a drafter per source; the cost is
  training and shipping a tuned MTP layer per pack. Basalt now publishes its comparison (NVFP4 pack,
  peer card, 400 W): prose +1.9 % tok/s and +4.3 % tokens per window, conversation and structured
  output within noise; its training recipe is not published.
- **Shared slot contexts (`max-context / N`).** Full-context fixed lanes are a scheduling invariant
  here (bounded FIFO ingress, one compact decode batch, per-lane graphs).
- **Per-section SHA-256 in the container.** No current integrity gap.
- **Importance-driven mixed-format allocation.** A converter-only recipe policy over registered
  formats, belonging with OSFP4's selection rule ([NVFP4 methods](nvfp4-quantization-methods.md)).

## Rejected

- **Suffix drafter:** duplicates Infernix's n-gram copy drafting (only the trigram min-3 admission
  differs, a parameter rather than a mechanism).
- **`ple_reader`:** not a drafter; it streams the model's own n-gram PLE table, which
  `--ngram-volume` already covers.
- **Per-device placement lines and the VRAM startup-refusal extra:** the RAM/VRAM ledgers and the
  `--vram-headroom-mib` pressure warnings cover them.
- **`experts.layout`:** the v3 object directory generalizes it. It does show that the streamed tier
  needs a uniform `record_stride`, should heterogeneous expert formats ever ship.
- **`hc8`:** recipe B already stores the hyper-connection mixers as `q8_g32_fp16` (int8 codes, FP16
  scale per 32), the same arithmetic; Basalt's precision notes confirm the choice.
- **64 KiB binary section header and the blanket no-requantization rule:** framing taste; the v3
  container and converter recipes cover the needs.
- **Live-stats web UI and JSON `/slots`:** the console panel and the Prometheus endpoint remain the
  observability surface.
- **GGUF format internals** (i-quant codebooks, sub-scale packing, affine K-quant minima): format
  changes with no site in the closed registry and the hardware-fixed per-16 `ue4m3` block-scale MMA.
- **DNS-rebinding Host allow-list:** defends a multi-tenant trust model this local, single-owner
  product does not have.
