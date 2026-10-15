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
| — | The same draft-confidence gate on the Qwen3.5/3.8-27B MTP path | Candidate (future work, noted 2026-10-15). `--spec-min-p` is refused there today because the gate does not exist: the 27B MTP draft step computes no draft-head probabilities (`ops::top_logprobs` is used only for scoring and prefill), and its round width comes from its own policy. It would need the draft step to emit each draft's log-probability, the cut in its round-width choice, and its own measurement, since 0.5 was tuned on Flash-Next. It helps only `--spec mtp` on 27B, not DFlash/DFlash2, the usual 27B configuration |
| [#62](https://github.com/kido5217/Infernix/issues/62) | Learned per-round draft policy: n-gram vs MTP from per-match-bucket acceptance and measured width cost | Candidate. Infernix already has an online width policy (EWMA acceptance, measured width costs, forced probes); the gain is bucket granularity, and Basalt documents a 2-8 % loss only when the choice is mis-made |
| [#63](https://github.com/kido5217/Infernix/issues/63) | Request-time RAM-low abort with named remedies | **Not adopted.** Basalt's request reads stream through the page cache and its RAM use grows per prompt, so stopping a request frees memory. Infernix pins its host memory when the Program starts (expert arena, prefix-cache Host tier, Program buffers; `memory_plan.h`), its pageable parts are a bounded n-gram row cache and a 1 GiB growth budget, and it reserves its own working set; aborting a request would free almost nothing while another program held the RAM |
| [#64](https://github.com/kido5217/Infernix/issues/64) | SSD file-cache floor in the RAM expert plan | **Not applicable.** Landed on the fork (`ec65e88a`), but Infernix's SSD expert tier reads the artifact unbuffered on both platforms (`DirectReadQueue`: `FILE_FLAG_NO_BUFFERING` / `O_DIRECT`), so expert pages never enter the OS page cache; the floor would take RAM from the expert tier and add SSD reads. The OS mapped-file tier the floor serves is rejected in the [design record](qwen3_8-flash-next-design.md) (Strata's `--mmap-experts`: 7.54 vs 12.18 tok/s) |
| [#65](https://github.com/kido5217/Infernix/issues/65) | Single `--config` JSON, CLI > file > defaults, validated at startup | Candidate (usability only) |
| [#66](https://github.com/kido5217/Infernix/issues/66) | Warn when binding beyond loopback without `--api-key` | Candidate (tiny) |
| [#67](https://github.com/kido5217/Infernix/issues/67) | Converter-shipped expert profile seeding the warm-start pipeline | Candidate. Its motivation cites a −19 % miss reduction, which is a leave-one-out **upper bound** for a category-keyed seed, not a measured gain over the existing warm-start seed ([design record](qwen3_8-flash-next-design.md) §9.3) |
| [#68](https://github.com/kido5217/Infernix/issues/68) | 2 MiB alignment of streamed expert objects | **Not adopted.** Landed on the fork (`945d301d`) as a writer-only change; only each layer's bank start moves (records keep a 4 KiB pitch and are not 2 MiB multiples), and Infernix's reader neither maps the file nor issues reads across banks, so nothing in this tree uses the alignment. No gain was measured |
| [#69](https://github.com/kido5217/Infernix/issues/69) | Decode-path full-vocabulary KLD gate with per-row diff output | Candidate (quality tooling; the existing KL tool is prefill-only) |
| [#70](https://github.com/kido5217/Infernix/issues/70) | Exact greedy-identity build gate for numerically neutral refactors | Candidate (dev tooling) |
| [#71](https://github.com/kido5217/Infernix/issues/71) | Hours-long mixed-protocol serve soak (offload and preemption routes) | Candidate (dev tooling) |
| [#72](https://github.com/kido5217/Infernix/issues/72) | Whole-model per-matmul precision map kept current with each change | Candidate (documentation practice) |
| [#73](https://github.com/kido5217/Infernix/issues/73) | Expert-cache-lent prefill chunk sizing (`--prefill auto`) | **Held.** It is this repository's own target design (§9.2, §13). The fork's implementation (`34c40bd9`) was not run on a GPU; its qualification is [#76](https://github.com/kido5217/Infernix/issues/76). It also changes the default route's memory layout (io staging, residency tables), so the fixed-chunk path needs re-measuring as well |

## Parked: needs a product change or a later decision

- **Peer-GPU expert tier.** Expert slots on a second card computed with the same kernels, so rows
  stay bit-identical; it breaks the one-GPU product contract.
- **Self-distilled per-pack MTP drafter.** The engine already loads a drafter per source; the cost is
  training and shipping a tuned MTP layer per pack, and Basalt publishes no distilled-vs-stock
  comparison.
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
