# Infernix documentation

Start with the [project README](../README.md) to build Infernix, download a published artifact, and
run the CLI or HTTP server.

## User guides

| Document | Purpose |
|---|---|
| [CLI](cli.md) | text, chat-history, image/video input, output streams, sampling, MTP, and common runtime options |
| [HTTP serving](serving.md) | OpenAI Responses/Chat Completions, Anthropic Messages, state, streaming, token counting, authentication, and tool calls |
| [Qwen3.8-Flash-Next](qwen3_8-flash-next.md) | requirements, conversion (recipes A and B, the GSQ-RCO IQ3_S GGUF import), recommended flags, the n-gram table in RAM, measured speed and quality of the offloaded-expert model |
| [Qwen3.5-122B-A10B](qwen3_5-122b.md) | Ornith-1.5-122B-NVFP4 and its UNCENSORED export with offloaded experts (in progress): requirements, conversion and verification, running, measured speed |
| [Ngram copy proposals](ngram.md) | optional copy acceleration alongside MTP, DFlash or DFlash2 at any concurrency |
| [Performance](performance.md) | RTX 5090 measurement coverage, per-model serving results, methodology, and publication rules |
| [Weight conversion](weight-conversion.md) | official recipes, custom formats and sources, conversion methods, optional components and artifact output |
| [Perplexity](perplexity.md) | fixed-corpus and custom-text causal perplexity, comparison rules, progress, and reports |
| [CLI examples](../examples/cli/) | committed text, multimodal, thinking, long-decode, and long-context inputs |

The executable `--help` output is the exact source for command-line option spelling and defaults.

## Model artifacts

The [project README](../README.md#models) lists every supported model with its Hugging Face
repository and the Qwen3.8-Flash-Next n-gram volumes. [`model-cards/`](../model-cards/) holds the
versioned source of each repository's model card.

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
| [Tree verification](maintainer/tree-verification.md) | DFlash2 and DFlash per-round draft trees, tree acceptance and its exactness, automatic and per-batch-size tree widths, accepted-path compaction |
| [Resource scheduling and context cache](maintainer/resource-scheduling-and-context-cache.md) | core design for incremental resources, admission, preemption and Replay recovery |
| [Hybrid prefix cache](maintainer/hybrid-prefix-cache-spec.md) | the prefix cache: block tree, sparse state snapshots, tap placement, Host slab tier, eviction and automatic configuration |
| [Paged KV context store](maintainer/paged-kv-cache.md) | typed pools, pages, replicas, address spaces, reservations and consumer views |
| [ReplaySSM GDN](maintainer/replayssm-gdn.md) | raw transition records and faithful commitment of the verified state prefix |
| [Op development](maintainer/op-development.md) | semantic boundaries, source ownership, numerical qualification and performance evidence |
| [Model-level numerical quality](maintainer/model-quality.md) | rounding noise at model scale, the lossless-route reference, Infernix-versus-NInfer prefill and decode results, Flash-Next NVFP4 against Unsloth UD-Q4_K_XL |
| [Basalt port assessment](maintainer/basalt-port-assessment.md) | which mechanisms of Basalt (a Strata fork for Flash-Next) are adopted, candidates, parked or rejected, and why |
| [NVFP4 quantization-method adoption](maintainer/nvfp4-quantization-methods.md) | evaluated converter-side 4-bit methods, what each demands of the stored representation, and the measured reason none is adopted |
| [Operational logging](maintainer/logging.md) | log ownership, presentation, severity and data policy |
| [Linear benchmark](maintainer/linear-benchmark.md) | pure Linear measurement, metrics and suites |
| [Linear tuning and reports](maintainer/linear-tuning.md) | tuning ranges, priority points, dispatch tradeoffs and final performance report format |
| [Qwen3.8-Flash-Next design](maintainer/qwen3_8-flash-next-design.md) | temporary design proposal (partly implemented; status in §19.1): bit-exact import of NVIDIA's NVFP4 checkpoint with exact W4A4 expert arithmetic, host-resident experts with CPU miss service and a VRAM expert cache, NVMe n-gram table, QSA over every KV profile (262K `int8` first), performance model and milestones; the exact GGUF IQ3_S and official FP8 imports (§16.2.2 arithmetic, §12.5 n-gram rows and RAM residency, §19.3.25 routes, results and rejected alternatives) |
| [Qwen3.5-122B-A10B offload plan](maintainer/qwen3_5-122b-offload-design.md) | temporary implementation plan (in progress; runs end to end, first measurements in §0): Ornith-1.5-122B-NVFP4 on one RTX 5090 with every weight bit-exact except the re-quantized MTP drafter, FP8 projections at static A8 in prefill and BF16 in decode, canonical W4A16 experts/shared expert/head, host-resident experts behind a VRAM cache via a shared offload component, GQA-16 attention over every KV profile, Vision with offload; checkpoint findings, vLLM/SGLang/KTransformers comparison, performance model, targets and phases |
| [GLM-5.3-Flash design](maintainer/glm5_3-flash-design.md) | temporary design proposal (not implemented): the EXL3 3.05 bpw GLM-5.3-Flash builds (MikeRoz's uncensored, turboderp's base) on one RTX 5090 with exact E3A16 experts in VRAM, pinned RAM and expert volumes striped over several NVMe drives; sparse MLA, pooled indexer, KDA and mHC Ops; measured storage topology, performance model and milestones |
| [DeepSeek-V4.1-Flash design](maintainer/deepseek_v4_1-flash-design.md) | temporary design proposal (not implemented): DeepSeek-V4.1-Flash-1017 on one RTX 5090, first as a per-expert-bit EXL3 build (2-5 bits per expert, E3A16 on DeepSeek's 8-bit MX8 activations) in a page pool shared by VRAM, RAM and three striped NVMe volumes, the lossless MXFP4 (NVFP4-valued) build deferred; CSA2 attention and the two-level indexer, Engram volumes in FP8, MXFP4 or IQ4_NL rows, single-pass mHC, the exact full-model prompt rule by default with bounded replay as an option, DSpark; a streaming conversion that fits the free disk; DeepSeek's own serving code (DeepInfer, its kernel libraries, deepseek-recipe), DwarfStar, Strata, Project Maya and other engines; the published 1017 quantizations weighed for speed against quality, performance model and milestones |

Model cards contain official artifact facts and source provenance. The
[conversion guide](weight-conversion.md) is the entry point for making an artifact. Exact config
fields, parameter expansion and native supported domains are maintained by the code linked from
these references.
