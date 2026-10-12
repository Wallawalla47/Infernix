# Infernix documentation

Start with the [project README](../README.md) to build Infernix, download a published artifact, and
run the CLI or HTTP server.

## User guides

| Document | Purpose |
|---|---|
| [CLI](cli.md) | text, chat-history, image/video input, output streams, sampling, MTP, and common runtime options |
| [HTTP serving](serving.md) | OpenAI Responses/Chat Completions, Anthropic Messages, state, streaming, token counting, authentication, and tool calls |
| [Qwen3.8-Flash-Next](qwen3_8-flash-next.md) | requirements, conversion (recipes A and B), recommended flags, measured speed and quality of the offloaded-expert model |
| [Ngram copy proposals](ngram.md) | optional copy acceleration alongside MTP, DFlash or DFlash2 at any concurrency |
| [Performance](performance.md) | RTX 5090 measurement coverage, per-model serving results, methodology, and publication rules |
| [Weight conversion](weight-conversion.md) | official recipes, custom formats and sources, conversion methods, optional components and artifact output |
| [Perplexity](perplexity.md) | fixed-corpus and custom-text causal perplexity, comparison rules, progress, and reports |
| [CLI examples](../examples/cli/) | committed text, multimodal, thinking, long-decode, and long-context inputs |

The executable `--help` output is the exact source for command-line option spelling and defaults.

## Model artifacts

| Model | Weights | Download | Versioned model card source |
|---|---|---|---|
| Qwen3.6-27B | `groupwise-int` | [Hugging Face](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) | [model card](../model-cards/Qwen3.6-27B-NInfer/README.md) |
| Qwen3.6-27B | `nvfp4` | [Hugging Face](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) | [model card](../model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md) |
| Qwen3.8-27B | `groupwise-int` | [Hugging Face](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | [model card](../model-cards/Qwen3.8-27B-NInfer/README.md) |
| Qwen3.8-27B | `nvfp4` | [Hugging Face](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) | [model card](../model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md) |
| Qwen3.6-35B-A3B | `groupwise-int` | [Hugging Face](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) | [model card](../model-cards/Qwen3.6-35B-A3B-NInfer/README.md) |

## Repository-local guides

- [Benchmarks](../bench/README.md)
- [Tests](../tests/README.md)
- [Tools](../tools/README.md)
- [Capability evaluation](../eval/README.md)

## Maintainer references

The active references under [`maintainer/`](maintainer/) record current architecture, model,
artifact, and maintenance contracts. These files are not additional user workflows or installed
API documentation.

[Engine architecture](maintainer/engine-architecture.md) is the single top-level reference. The
other references own narrower contracts:

| Document | Responsibility |
|---|---|
| [Engine architecture](maintainer/engine-architecture.md) | model/config/weight ownership, loading-to-execution flow, requests, scheduling, transactions and graphs |
| [Build system](maintainer/build-system.md) | CMake targets, explicit source ownership, CUDA compilation boundaries, presets and developer configuration |
| [Artifact container](maintainer/artifact-container.md) | v3 directory, objects, logical bindings, Uses, resources and file framing/sharding |
| [Numeric formats](maintainer/tensor-formats.md) | represented values, codes/scales, conversion arithmetic and numerical interpretation |
| [Storage layouts](maintainer/storage-layouts.md) | packing, plane offsets, padding, encoded sizes and view addressing |
| [Qwen3.5 model](maintainer/qwen3_5-model.md) | Dense/MoE mathematics, instance config, logical parameters, MTP, Vision and state semantics |
| [DFlash and DFlash2](maintainer/dflash.md) | conditioning, masked draft computation, proposal distributions and backend state |
| [Constrained decoding](maintainer/constrained-decoding.md) | GBNF/JSON/choice/regex, tool policies, masks, speculative transactions and observations |
| [DFlash2 tree verification](maintainer/tree-verification.md) | per-round lattice draft trees, tree acceptance and its exactness, automatic and per-batch-size tree widths, accepted-path compaction |
| [Resource scheduling and context cache](maintainer/resource-scheduling-and-context-cache.md) | core design for incremental resources, admission, preemption and Replay recovery |
| [Hybrid prefix cache](maintainer/hybrid-prefix-cache-spec.md) | the prefix cache: block tree, sparse state snapshots, tap placement, Host slab tier, eviction and automatic configuration |
| [Paged KV context store](maintainer/paged-kv-cache.md) | typed pools, pages, replicas, address spaces, reservations and consumer views |
| [ReplaySSM GDN](maintainer/replayssm-gdn.md) | raw transition records and faithful commitment of the verified state prefix |
| [Op development](maintainer/op-development.md) | semantic boundaries, source ownership, numerical qualification and performance evidence |
| [Model-level numerical quality](maintainer/model-quality.md) | rounding noise at model scale, the 16-bit-activation reference, Infernix-versus-NInfer prefill and decode results, Flash-Next NVFP4 against Unsloth UD-Q4_K_XL |
| [Operational logging](maintainer/logging.md) | log ownership, presentation, severity and data policy |
| [Linear benchmark](maintainer/linear-benchmark.md) | pure Linear measurement, metrics and suites |
| [Linear tuning and reports](maintainer/linear-tuning.md) | tuning ranges, priority points, dispatch tradeoffs and final performance report format |
| [Qwen3.8-Flash-Next design](maintainer/qwen3_8-flash-next-design.md) | temporary design proposal (partly implemented; status in §19.1): bit-exact import of NVIDIA's NVFP4 checkpoint with exact W4A4 expert arithmetic, host-resident experts with CPU miss service and a VRAM expert cache, NVMe n-gram table, QSA over every KV profile (262K `int8` first), performance model and milestones |
| [Flash-Next GSQ-RCO IQ3_S plan](maintainer/qwen3_8-flash-next-iq3s-plan.md) | temporary plan (not implemented): bit-exact import of ISTA-DASLab's GSQ-RCO IQ3_S GGUF (per-layer i-quant expert records, k-quant dense, exact integer arithmetic), IQ4_NL n-gram volume with optional whole-table RAM residency, MTP drafter fetched from Qwen's checkpoint |
| [Flash-Next official FP8 plan](maintainer/qwen3_8-flash-next-fp8-plan.md) | temporary plan (not implemented): bit-exact import of Qwen's block-FP8 experts with exact W8A16 arithmetic, three-tier VRAM/RAM/SSD residency over striped expert volumes, BF16 or 8-bit dense recipes |
| [GLM-5.3-Flash design](maintainer/glm5_3-flash-design.md) | temporary design proposal (not implemented): the EXL3 3.05 bpw GLM-5.3-Flash builds (MikeRoz's uncensored, turboderp's base) on one RTX 5090 with exact E3A16 experts in VRAM, pinned RAM and expert volumes striped over several NVMe drives; sparse MLA, pooled indexer, KDA and mHC Ops; measured storage topology, performance model and milestones |

Model cards contain official artifact facts and source provenance. The
[conversion guide](weight-conversion.md) is the entry point for making an artifact. Exact config
fields, parameter expansion and native supported domains are maintained by the code linked from
these references.
