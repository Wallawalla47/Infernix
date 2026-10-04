# Qwen3.8-Flash-Next: handoff to the GPU machine

**Temporary working note.** It goes with the [design](qwen3_8-flash-next-design.md) and is deleted
when the feature lands, just as the design is. Status of record lives in design
[§19.1](qwen3_8-flash-next-design.md#191-implementation-status-and-handoff); if this note and §19.1
ever disagree, §19.1 wins. Update §19.1 as work proceeds.

## 1. Goal and requirements

Add `Qwen4ExpForCausalLM` (Qwen3.8-Flash-Next, ~180B MoE) to NInfer and beat Strata's 72-80 tok/s
decode on the target machine. The target is one RTX 5090 (32 GB), 96 GB DDR5 and one NVMe SSD.
The user's requirements, all reflected in the design:

- The source is `nvidia/Qwen3.8-Flash-Next-NVFP4`. Every tensor NVIDIA quantized (8 bits or fewer) is
  imported **bit-exactly**, with **NVIDIA's activation formats** (A4 with ModelOpt `input_scale`).
- Routed experts live in pinned host RAM, with a VRAM expert cache. Misses are served by the CPU in
  place or by DMA. The PLE n-gram table lives on NVMe.
- Every NInfer KV profile is supported. **262K context with `int8` KV is the primary configuration**;
  optimize for it first, then the others.
- BF16 tensors move to 8 bits only where that helps significantly (recipe B). Recipe A keeps them BF16.
- The work must be original: ninfer-ext and other projects are sources of facts and ideas only, with
  no copied code.

Read [`AGENTS.md`](../../AGENTS.md) first: the repository rules, evidence standards, and the Python
interpreter (`/home/neroued/miniconda3/envs/py311/bin/python`).

## 2. Where everything is

Branch: `claude/wonderful-ritchie-65xtnh` (17 commits on top of `master`, from `ab25aac` to `9389d59`).

| What | Path |
|---|---|
| **Design (the authority)** | [`docs/maintainer/qwen3_8-flash-next-design.md`](qwen3_8-flash-next-design.md): performance model §4, artifact §6, decode pipeline and kernels §8, expert cache §9, CPU engine §10, speculation §11, NVMe PLE §12, calibration §14, numerics §16, milestones §19, **status §19.1** |
| Format and layout definitions | [`tensor-formats.md`](tensor-formats.md) §3.5 `nvfp4_mul`, §3.6 `fp8_e4m3fn_block128_f32`; [`storage-layouts.md`](storage-layouts.md) §6 `nvfp4_expert_rg16_v1`, §7 `block128_scale_v1` |
| Canonical W4A4 arithmetic, shared by CPU and GPU | `src/ops/common/canonical_math.h` |
| CPU expert kernels (scalar, AVX2, AVX-VNNI, AVX-512 VNNI) | `src/ops/offloaded_sparse_moe/cpu/w4a4_expert.{h,cpp}` |
| CPU worker team (§10.3) | `src/ops/offloaded_sparse_moe/cpu/expert_team.{h,cpp}` |
| GPU narrow expert route, first correctness version | `src/ops/offloaded_sparse_moe/cuda/narrow_expert.{h,cu}` |
| Op build wiring | `src/ops/offloaded_sparse_moe/sources.cmake` |
| Host expert cache: LFRU, residency words, frame epochs, controller | `src/models/qwen4_exp/program/expert_cache/expert_cache.{h,cpp}` |
| PLE n-gram row ids | `src/models/qwen4_exp/frontend/ngram_hash.{h,cpp}` |
| Model build wiring | `src/models/qwen4_exp/program_sources.cmake` |
| C++ format registration | `src/core/weight.h` (`QType::NVFP4_MUL`, `FP8_E4M3FN_BLOCK128_F32`; `QuantLayout::ExpertRg16`, `Block128Scale`), `src/artifact/formats.cpp`, `src/core/weight_view.{h,cpp}` (`expert_bank_planes`, `block128_planes`) |
| Python formats, layouts and codecs | `tools/artifact/formats.py`, `tools/artifact/layouts.py`, `tools/artifact/codecs/nvfp4_expert.py`, `tools/artifact/codecs/fp8_block.py` |
| ModelOpt source readers | `tools/convert/sources/modelopt.py` |
| M0 checkpoint inspector | `tools/flash_next/inspect_checkpoint.py` |
| A4 quantizer reference vs ModelOpt | `tools/flash_next/a4_reference.py` |
| n-gram specification and fixture generator | `tools/flash_next/ngram.py` |
| Machine probes and runner | `tools/flash_next_probe/{run_m0.sh,host_probe.cpp,nvme_probe.cpp,gpu_probe.cu}` |
| Routing-trace cache replay and LFRU conformance | `tools/expert_cache_replay/{replay.py,conformance.py}` |
| C++ tests | `tests/ops/test_offloaded_moe_{cpu.cpp,team.cpp,cuda.cu}`, `tests/ops/offloaded_moe_fixtures.h`, `tests/models/qwen4_exp/test_{expert_cache,ngram_hash}.cpp`, `tests/artifact/test_reader.cpp` |
| Python tests | `tests/artifact/test_flash_next_formats.py`, `tests/convert/test_modelopt_sources.py`, `tests/test_flash_next_tools.py`, `tests/test_flash_next_ngram.py`, `tests/artifact/flash_next_interop.py` (CTest) |
| Fixtures | `tests/fixtures/expert_cache/lfru_conformance.txt`, `tests/fixtures/qwen4_exp/ngram_rows.txt` |

## 3. What was done, and how far it is verified

All work so far was done in a cloud VM: 4 vCPU Xeon at 2.1 GHz with AVX-512 VNNI and 15 GB RAM,
no GPU, and no Hugging Face access. nvcc 13.4 from PyPI was used to compile only. The CMake build
configured there with a stub `libcuda.so`.

| Piece | Verified on the VM | Not yet verified |
|---|---|---|
| Canonical arithmetic | Encoders vs grid enumeration; BF16 rounding; `exp_c`/SiLU exhaustive over BF16 vs FP64; exact int64 vs FP64 | Device execution |
| CPU expert kernels | Bit equality across ISAs, batch, split and compilers; golden hashes `kGolden1=0x8e4b822b07293949`, `kGolden4=0x947202285aeb28bd` | Speed on the target CPU |
| CPU worker team | Equals `expert_forward` for 1-40 workers; TSan clean in `--quick` mode | Latency, pinning and CCD split on the target |
| GPU narrow route | Compiles for `sm_120a` | **Never run.** `ninfer_offloaded_moe_cuda_test` must pass |
| Formats and loader | Python round trips are word-exact; C++ geometry equals Python; Python writer to C++ reader interop | A real converted artifact |
| Expert cache | Victims identical to the replay tool on 1,200 groups; agent simulation shows no early frame reuse | The transfer agent around it (copy stream, `cuStreamWriteValue32`, loans) |
| n-gram row ids | Python equals an independent tensor form; C++ equals the fixture (800 positions) | Real config values; device kernel |
| A4 vs ModelOpt | Rules ported from ModelOpt 0.47 source agree except one guard (below) | ModelOpt's own kernel on the GPU; guard frequency on real activations |
| Probes | Host and NVMe probes run; GPU probe compiles | Everything on the target |

## 4. Findings the next agent must know

1. **Measure everything on this machine.** The VM figures in the design (§10.2 table, DRAM rates,
   team latency, prefetch gain) show only which mechanisms matter. Worker count, software prefetch
   distance (`prefetch_bytes`; the default of 2,048 is a placeholder), ISA variant, spin/park
   thresholds, DRAM bandwidth and contention, NVMe latency, and PCIe/copy-engine behavior all come
   from `run_m0.sh` and calibration (§14).
2. **The CPU expert kernel is compute-bound from n = 2 columns** on the VM: n = 8 costs ~6× n = 1.
   T = 8 may not be DRAM-bound as §10.2 originally assumed. M3 must optimize the multi-column path
   (decode once, then loop columns) and decide whether n ≥ 4 experts go to the GPU route.
3. **ModelOpt's 1e-5 block-scale guard** (a dequantized scale below 1e-5 becomes 1.0) is
   deliberately not adopted (§16.2). TensorRT-LLM has no such guard.
4. **Slack frames:** the frame pool needs at least peak admissions per round × (D + 1) frames
   outside the policy's capacity (§9.5), or policy hits become CPU-served.
5. **Pre-existing build issue:** `src/ops/kv_cache/append/vq2_launch.cu` failed to compile with
   nvcc 13.4 on the VM. That code is untouched by this branch. Check that it builds with the local
   CUDA 13.1 before assuming anything.
6. TensorSharp lessons and other engines are in design §3.3. The verify-width-invariance option is in §11.3.

## 5. What to do next, in order

1. **Build and run the existing tests on the GPU machine.**
   ```bash
   cmake -S . -B build -G Ninja -DBUILD_TESTING=ON
   cmake --build build -j
   ctest --test-dir build -R "offloaded_moe|artifact|expert_cache|ngram_hash" --output-on-failure
   /home/neroued/miniconda3/envs/py311/bin/python -m pytest -q tests/artifact tests/convert \
     tests/test_flash_next_tools.py tests/test_flash_next_ngram.py
   ```
   `ninfer_offloaded_moe_cuda_test` is the key one. It must reproduce the golden hashes, match the
   CPU engine bit for bit (device and mapped-host records), and match the canonical scalar functions.
   If it fails, fix the GPU side: the CPU side is the qualified reference.
2. **M0: run the probes and inspector**, then record the facts in the design (§4, §6.1, §8.3, §10,
   §12) and replace the VM numbers.
   ```bash
   PYTHON=/home/neroued/miniconda3/envs/py311/bin/python \
     tools/flash_next_probe/run_m0.sh /path/to/Qwen3.8-Flash-Next-NVFP4 /path/on/nvme/large_file
   ```
   This confirms or revises: one `weight_scale_2` per matrix; whether gate and up share
   `input_scale`; the MTP activation scheme; the n-gram config. Also run
   `python -m tools.flash_next.a4_reference` with CUDA and `nvidia-modelopt` installed. That compares
   our A4 rule with ModelOpt's real kernel.
3. **M2: recipe A conversion** (`qwen3_8_flash_next_nvfp4`): converter recipe, model contract and
   bindings, using the formats and `ModelOpt` readers already in place. Exit criterion: word equality
   with the checkpoint (§16.3).
4. **M3 kernels:**
   - optimized K7a/K7b (§8.5: landing tickets, PDL, TMA ring, persistent items, IMMA for n ≥ 3),
     qualified bit-exactly against `narrow_expert.cu` and the CPU engine;
   - the CPU multi-column optimization, re-measured with `host_probe`;
   - QSA (`int8` KV profile first), hyper-connection kernels, PLE device kernel against
     `NgramHash`.
5. **M4-M6:** functional model with CPU-only experts, then the expert cache transfer agent (wrapping
   `CacheController`), then decode performance (§19 has the exit criteria). M1 (real routing traces
   replayed through `tools/expert_cache_replay`) runs in parallel and must finish before the cache
   policy is frozen.

## 6. Working conventions

- Commit with Conventional Commit subjects. Keep design §19.1 current. Delete this note and the design
  when their content has moved into the active references (design §20).
- Floating-point Ops need an independent FP32/FP64 oracle; the expert route is exact (integer
  oracle and golden hashes). See [op development](op-development.md).
- Report favorable and unfavorable results alike (`AGENTS.md`, "Reporting and completion").
