# Tests

The retained tests protect current `.ninfer`, numerical operator, model, runtime-transaction,
benchmark-report, and external protocol behavior. Op qualification and CUDA implementation rules
are defined in [Op development](../docs/maintainer/op-development.md); product execution and
ownership contracts are defined in [Engine architecture](../docs/maintainer/engine-architecture.md).

## Organization

- `artifact/` — v3 framing, directory/binding records, codecs, sharding, selected-object
  materialization and Python-writer/C++-reader interoperability;
- `convert/` — source interpretation, Qwen logical mapping, recipe overrides/sharing, optional
  components, resources, proposals and numerical conversion methods;
- `models/qwen3_5/` — config/binding, frontend, state/context stores, workspace, MTP alignment and
  opt-in real Engine integration;
- `ops/` — semantic Op qualification with independent mathematical or state-transition oracles;
  Linear and fused Linear suites are separated by their supported weight/activation paths;
- `runtime/` — scheduler fairness, cache resource policy, capacity and context-cost calculations;
- `bench/ttft/` — HTTP measurements, campaign orchestration and workload construction;
- root C++ tests — core storage, public API, serving protocols, logging, benchmark reports and
  causal-scoring evaluation;
- `test_serve_corpus.py` — agreement between the serving request-log schema and its measurement
  consumer.

Tests are grouped by observable risk, not by mirroring every source file or class.
`CMakeLists.txt` includes explicit registrations from `cmake/`, `artifact/`, `models/qwen3_5/`
and `ops/`. Registration helpers live in `cmake/InfernixTests.cmake`; included manifests keep
executables and CTest working directories under `build/tests/`.
`ops/op_tester.h` and `ops/op_check.h` own only reusable device/guard and comparison mechanics.
Concrete numerical criteria remain named by the semantic Op suite; there are no cross-Op tolerance
presets.

`ops/quantized_weight.h` is the common packed-weight fixture for Q4/Q5/Q6/Q8, FP8 and NVFP4 Op tests. It
owns deterministic payload generation, device `Weight` views, row views, and independent logical
weight decoding.

## Build and run

Select a Python environment with the dependencies for the tests first. The maintained environment
uses Python 3.11; CMake finds Python 3 without restricting its minor version.
`Python3_EXECUTABLE` selects the interpreter used by interop and frontend tests explicitly.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Alternatively, `cmake --preset dev` enables products, tests and benchmarks together.
After building, `ctest --preset dev` runs the same CTest suite. See
[Build system](../docs/maintainer/build-system.md) for local interpreter presets.

The chat-template reference test uses Python Jinja2.

Run a focused target for a localized change:

```bash
cmake --build build --parallel --target infernix_sampling_test
ctest --test-dir build -R infernix_sampling_test --output-on-failure
```

Enable uniform floating-point error records when establishing or reviewing an Op criterion:

```bash
INFERNIX_OP_REPORT_STATS=1 \
  ctest --test-dir build -V -R '^infernix_(rmsnorm|softmax_attention)_test$'
```

Every participating comparison emits one `OP_ERROR_STATS` record containing the stable case label,
actual error, active limit, and error-to-limit ratio. The switch changes reporting only; the same
statistics still drive the normal verdict. Passing tests remain quiet without it.

`infernix_softmax_attention_test --causal-only` runs both D256 geometries and all five KV types;
`--kv-dtype bf16|int8|fp8|nvfp4|k8v4` selects the same complete causal suite for one type, and
`--non-causal-only` runs only the packed and context sections. CTest registers the latter as
`infernix_softmax_attention_test` and the causal suite as one entry per KV type
(`infernix_softmax_attention_<type>_test`, and `infernix_softmax_attention_wide_<type>_test` for
`--wide-only`), so `ctest -j` runs the KV types concurrently; together the entries run exactly the
cases of one unsharded invocation.

`infernix_gdn_replay_fold_test --wide-only` likewise splits into one CTest entry per width and layer
profile (`infernix_gdn_replay_fold_wide_w<width>_l<layers>_test`, from `--width 33|48|64 --layers
48|30`) plus `infernix_gdn_replay_fold_wide_batched_test` (`--batched`). All seven together peak at
about 17 GB of device memory on an RTX 5090.

The vector-quantized `vq2` and `k4v2` caches have their own suites. `infernix_kv_cache_vq_test`
compares every stored code, row scale, exact-window slot and tag with independent host encoders
(`kv_cache_vq_reference.h`), checks untouched bytes, and moves tree-compacted slots between state
slots. `infernix_vq_attention_test` checks both geometries and both formats against an FP64 oracle
of the per-query read rule (sinks and keys at most 768 positions before the query as exact INT8-G64
rows, every other key from its stored codes): grouped decode and verification widths, batched rows with partial valid columns, tree
masks, prompt chunks with key splits, cached-only calls, stale window slots, and the window a wide
call commits from staging. Each batch row's window lives in a different state slot from its KV table
row.
The suite covers prefill, decode/spec widths, batched prefixes, cache effects, and Graph replay and
updates with changing live lengths. Numerical cases include small, unit-RMS and RMS1.8 Q/K inputs.
The FP64 oracle retains internal Q quantization error; the INT8/FP8 compute budgets account for
that accepted approximation. The default invocation also runs packed and context attention.
Large causal reference calculations spread their query rows over the host threads while
preserving each output's FP64 accumulation order, and a batch case evaluates its requests'
references concurrently when they use distinct cache rows; small cases remain serial. Fixture
values are the serial mt19937 stream mapped in parallel, so they equal a serial fill exactly. Cache fixtures populate the reachable KV
prefix while reserving the full execution-envelope page table. The runner reports elapsed time
per KV type.

`infernix_qsa_test` checks Qwen3.8-Flash-Next's QSA at the real geometry against FP64 oracles:
index queries, pooled keys read back from the paged plane, the raw-key tails exactly (including
rewriting the last position of a block completed inside a call or a verification commit, as the
MTP drafter does), chunking invariance, block selection (exact, on a value grid where FP32 scores
are exact), and attention over decoded `bf16` and `int8` K/V across the dense/selected boundary.

`infernix_qsa_select_test` qualifies QSA block selection (`ops::detail::qsa_select`) against three
oracles: the one-CTA-per-column kernel it replaced, kept verbatim in the test (counts and selected
ids bitwise equal, FP32 scores bitwise equal where both keep them); the exact top-k of the
production kernel's own scores (equal scores to the lower block id, ids ascending); and an FP64
score oracle (FP32 scores within an accumulation bound, no unselected block above a selected one
beyond the bounds). Cases span 0.5K-131K blocks per column, both sides of the 512-block budget, 1-9
rows and widths up to 1,024 columns (groups of 128), index heads 1-4, ratios 4-128, exact,
zero-score and near ties, bins beyond the candidate buffer, and CUDA Graph replays with new
positions over garbage scratch.

Linear tests are independently runnable by weight and activation-compute profile:

```bash
cmake --build build --parallel --target \
  infernix_linear_q4_a16_test infernix_linear_q5_a16_test \
  infernix_linear_q6_a16_test infernix_linear_q8_a16_test
ctest --test-dir build -R '^infernix_linear_(q4|q5|q6|q8)_a16_test$' --output-on-failure
```

For a change confined to one Q8 geometry, use the same public conformance cases with
`./build/tests/infernix_linear_q8_a16_test --shape N K`. The default CTest invocation still covers
all registered Q8 geometries.

For a change confined to a newly supported BF16 geometry, use
`./build/tests/infernix_linear_bf16_a16_test --shape N K`. These cases use the common full-K FP64
oracle with exact BF16 weights, complete output checks for small N, route boundaries, changed-input
Graph replay, both public overloads, workspace domains, and preservation/guard checks. The small
control projections also cover terminal-K pulses and paired cancellation.

All Linear files use `ops/linear/linear_test_common.{h,cpp}` and the same
`ops/quantized_weight.h` fixture as the fused projection tests. The fixture produces the complete
packed GPU payload and exact-decodes the logical float rows used by the one
`cpu_linear_gemm_fp64()` reference. The reference performs naive double accumulation and never
reproduces a production route's activation quantization, staging, reduction tree, or BF16 output
rounding. Each activation compute path selects one centrally defined comparison tolerance for its
whole suite; private kernel, schedule, launcher, and T selection do not change it. Individual test
files call public `linear()` and contain no private selector, launcher, schedule, or kernel
assertions.

The Q8 suite includes `[2560,6144]`, `[6144,2560]`, `[10240,2560]`, `[12288,2560]`, and
`[16384,2560]`: full-output FP64 comparisons at T=1/4/8,
sampled-output checks at larger extents including 512/1024 and 129/1025, production boundaries,
changed-input Graph replay, both public overloads, permissive policies, input/weight preservation,
output guards, and valid/invalid workspace intervals.

The Linear, LinearAdd and LinearSwiGLU common `.cpp` implementations each compile once into a
test support library. Both those libraries and the Op test executables receive the oracle's
`-fno-fast-math` and `-ffp-contract=off` options on GNU/Clang C++ compilers.

Run the native Python suites with the project Python environment:

```bash
python3 -m pytest \
  tests/artifact tests/convert tests/bench/ttft \
  tests/test_serve_corpus.py
```

The Python suites exercise conversion, encoded output and measurement tools without model inference.
The maintained environment uses Python 3.11 with the dependencies for those suites. C++ binding and Engine tests
cover consumption of their resulting representation.

The real loading test accepts an explicit artifact path and optional component selection:

```bash
./build/tests/infernix_qwen3_5_loading_real_test \
  --artifact out/qwen3_6_27b.ninfer --vision --speculative mtp --proposal optimized
```

Add `--host-only` to check semantic binding without uploading weights. This does not construct a
Program or establish native Op support.

The C++ prefix/MTP integration test is separately opt-in because it loads the full artifact and
runs the real engine:

```bash
INFERNIX_TEST_ARTIFACT=$PWD/out/qwen3_6_27b.ninfer \
  ctest --test-dir build -R infernix_qwen3_5_prefix_real_test --output-on-failure
```

The causal-scoring integration test uses the same artifact variable and checks a full 1,024-column
score tile, overlapping target suffixes, and repeated-window State/KV isolation:

```bash
INFERNIX_TEST_ARTIFACT=$PWD/out/qwen3_8_27b_nvfp4.ninfer \
  ctest --test-dir build -R infernix_qwen3_5_score_real_test --output-on-failure
```

Run the 35B-A3B MoE route independently:

```bash
INFERNIX_TEST_ARTIFACT=$PWD/out/qwen3_6_35b_a3b.ninfer \
  ctest --test-dir build -R infernix_qwen3_5_moe_real_test --output-on-failure
```

The Qwen3.8-Flash-Next (`qwen4_exp`) artifact tests read `INFERNIX_QWEN4_ARTIFACT` instead (the first
shard of a `.infernix` artifact) and, when the n-gram volume is not beside it,
`INFERNIX_QWEN4_NGRAM`:

```bash
INFERNIX_QWEN4_ARTIFACT=D:/out/qwen3_8_flash_next-00001-of-00003.infernix \
INFERNIX_QWEN4_NGRAM=G:/out/qwen3_8_flash_next.ninfer.ngram \
  ctest --test-dir build -R "infernix_qwen4_exp_.*real" --output-on-failure
```

They cover the forward pass against the FP64 reference, the SSD tier's records, the route trace, VRAM
pressure, the prefix cache, preemption, `/v1/decide` and recovery from a failed expert read.

Without their artifact variable, CTest marks these real Engine tests as skipped. Real-artifact targets
carry the `real` label and `RUN_SERIAL` so CTest runs them alone. Run directly invoked GPU
integration tests serially. `INFERNIX_PREFIX_REAL_SCENARIO` selects a focused prefix scenario such as
`vision`, `late-instructions` or `concurrent`; the default is `all`. These integration checks
use behavior and state accounting rather than another numerical path's generated tokens as a golden.

The `attention` scenario checks the selected KV type, chunked prefill, concurrent Graph decode
across a resource tier, prefix continuation, and workspace bounds:

```bash
INFERNIX_TEST_ARTIFACT=$PWD/out/qwen3_6_27b.ninfer \
INFERNIX_PREFIX_REAL_SCENARIO=attention INFERNIX_TEST_KV_DTYPE=fp8 \
INFERNIX_TEST_SPECULATIVE=mtp INFERNIX_TEST_BATCH=2 \
  ./build/tests/infernix_qwen3_5_prefix_real_test
```

KV choices are `bf16`, `int8`, `fp8`, `nvfp4`, `k8v4`, `vq2`, and `k4v2`; backend choices are
`none`, `mtp`, `dflash`, and `dflash2`, requiring an artifact with the selected component. Batch
defaults to 2; `INFERNIX_TEST_DRAFT_TOKENS` overrides the default MTP3 or DFlash7 block. The
DFlash2-specific integration executable, the n-gram lifecycle test (fifth argument) and the hybrid
prefix test (`INFERNIX_HYBRID_KV_DTYPE`) accept the same KV names and reject unknown ones.

Continuation and pressure recovery have dedicated entries:

```bash
INFERNIX_TEST_ARTIFACT=$PWD/out/qwen3_8_27b_nvfp4.ninfer \
  ctest --test-dir build -R infernix_qwen3_5_agent_continuation_real_test --output-on-failure

INFERNIX_TEST_ARTIFACT=$PWD/out/qwen3_8_27b_nvfp4.ninfer \
INFERNIX_TEST_BACKEND=dflash2 \
  ctest --test-dir build -R infernix_qwen3_5_preemption_real_test --output-on-failure

INFERNIX_TEST_ARTIFACT=$PWD/out/qwen3_8_27b_nvfp4.ninfer \
  ./build/tests/infernix_qwen3_5_native_transactions_test dflash2
```

The agent entry checks multi-turn continuation and branching. The preemption entry exercises
Replay recovery and cancellation while paused or replaying; `INFERNIX_PREEMPTION_REAL_SCENARIO`
selects `all`, `replay`, `cancel-paused` or `cancel-replay`. Native transaction tests cover pause and
Replay with the exact ledger splits, grammar-row failure and replayed sampling counts; their
positional backend is `none`, `mtp`, `dflash` or `dflash2`. Each backend requires an artifact containing that component.
Public-HTTP latency and output gaps are measured separately by the
[TTFT campaign](../tools/bench/ttft/README.md).

`infernix_qwen3_5_tools_real_test [none|mtp|dflash|dflash2] [graph|eager|basic|replay|cancel] [concurrency]`
uses `INFERNIX_TEST_ARTIFACT` for strict tools, thinking, raw continuation, mixed batches, and
call/result prefix reuse, and required-tool → JSON continuation with committed constraint observations.
`basic` checks default constraints with open/complex schemas, continuation,
streaming and mixed strict/basic/free rows. The `replay` mode forces resource pressure and
validates the completed argument value after recovery; `cancel` interrupts the paused request.
`INFERNIX_TEST_TOOL_REPORT` appends schema/output/timing JSONL.
`python3 tests/models/qwen3_5/test_tool_schema.py` checks the native Qwen grammar and decoder against
`jsonschema` (dependencies in `tests/text/requirements.txt`), including string pattern/Unicode-length
intersections. The JSON Schema oracle also covers finite-value filtering, reference/union siblings,
closed-object and positional-array intersections, draft-07 tuples, recursive conjunctions,
numeric endpoints and JSON publication rounding. HTTP parsing tests check schema-number precision
before protocol adapters serialize the schema.

`infernix_qwen3_5_grammar_real_test [none|mtp|dflash|dflash2] [graph|eager] [concurrency] [vision]` uses
`INFERNIX_TEST_DRAFT_TOKENS` to override the default draft count of three,
`INFERNIX_TEST_NGRAM_DRAFT_TOKENS` to add n-gram copy rounds, `INFERNIX_TEST_DRAFT_TREE_NODES` to give
DFlash2 a tree width for every batch size (constrained rounds still verify chains), and
`INFERNIX_TEST_ARTIFACT` to check GBNF/JSON/schema/choice/regex content, sampling, thinking, continuation, prefix reuse
and mixed batches. `infernix_regex_choice_test` checks literal-set prefix masks and regex edge cases;
`python3 tests/text/test_regex_choice.py` compares regex membership with independent fullmatch semantics.
Set `INFERNIX_TEST_CONSTRAINT=grammar` or `json_schema` on the preemption test to
check matcher continuity through Replay and cancellation. `infernix_grammar_test` and
`infernix_json_schema_test` cover CPU language semantics. `infernix_json_schema_oracle_test` compares
supported schemas with the independent Python `jsonschema` validator; install its dependency with
`python3 -m pip install -r tests/text/requirements.txt` in the selected test environment.
The sampling and speculative Op tests qualify masks against independent mathematical oracles.

The capability-evaluation coordinator has its own environment and unittest entry point:

```bash
PYTHONPATH=eval eval/.venv/bin/python -m unittest discover \
  -s eval/tests -p 'test_*.py'
```

Run the serving contract manually after starting a resident server in another terminal:

```bash
./build/apps/infernix-serve out/qwen3_6_27b.ninfer \
  --host 127.0.0.1 --port 18080 --vision
```

```bash
python3 -m tools.smoke.serve_contract \
  --base-url http://127.0.0.1:18080 --model qwen3.6-27b
```

This smoke check is intentionally not a CTest: it needs the real artifact, a supported GPU, and a
server process that remains alive while the client exercises OpenAI Responses/Chat, Anthropic,
state, streaming, and multimodal requests.

The thinking-preservation fixture starts and stops its own server, submits a fixed two-step tool
history, compares stripped and preserved closed-turn prompt lengths, and verifies compatible
prefix reuse, speculative execution, frontier bounds and Responses inheritance:

```bash
python3 tools/smoke/serve_thinking_preservation.py \
  --artifact out/qwen3_6_27b.ninfer --backend mtp

python3 tools/smoke/serve_thinking_preservation.py \
  --artifact out/qwen3_6_35b_a3b.ninfer --backend dflash
```

The shared messages are in
[`fixtures/serve/qwen3_6_thinking_preservation.json`](fixtures/serve/qwen3_6_thinking_preservation.json).

## What belongs here

A permanent test should protect one current risk, such as:

- exact artifact bytes, geometry, object binding, or conversion transform;
- a numerical operator contract with an independent oracle;
- model Frontend or Program frontier, prefix, MTP, or multimodal behavior;
- generated-token commit/stop/cancel consistency;
- public benchmark or OpenAI/Anthropic observable behavior;
- a reproduced supported bug.

Performance-only assertions belong in benchmarks and profiler review. Source scans,
implementation-shape assertions, trivial getters/configuration, retired command surfaces, and
broad additions without a concrete regression risk do not belong in the permanent suite.

## DFlash2 Engine integration

The DFlash prefill regression checks actual KV contents after a StateImage fork and a conflicting
decode binding, including shortened chunks and oversized local/full KV appends. It uses native
Program storage and the production prefill route; select the draft component stored in the artifact:

```bash
cmake --build build -j --target infernix_qwen3_5_dflash_prefill_real_test
INFERNIX_TEST_ARTIFACT=out/qwen3_8_27b_nvfp4.ninfer \
  build/tests/infernix_qwen3_5_dflash_prefill_real_test dflash2
INFERNIX_TEST_ARTIFACT=out/qwen3_6_35b_a3b.ninfer \
  build/tests/infernix_qwen3_5_dflash_prefill_real_test dflash
```

The Engine test uses an artifact containing DFlash2 and checks output budgets, speculative activity,
forced thinking-control append, penalty-enabled sampling, compact batches with unequal budgets,
same-route same-seed replay, retained/fresh prefix behavior and absence of a full backend KV pool.
A shared DFlash/DFlash2 fixture starts decode at token 63, verifies across the page boundary, stops
after one target column at token 64, and checks the exact retained frontier and subsequent generation
with and without reuse.
The KV Store test checks exact mapping and reservation accounting for the same transition.
K>=7 also exercises a stop inside a licensed block; K=15 additionally checks oversized prefill,
local ring wrap, and the logical context-capacity tail. Optional Vision runs image/video capture
and prefix restore. Zero extra Device StateImage slots exercise Host snapshot/restore.

```bash
cmake --build build -j --target infernix_qwen3_5_dflash2_real_test
INFERNIX_TEST_ARTIFACT=out/qwen3_8_27b.ninfer \
  build/tests/infernix_qwen3_5_dflash2_real_test 15 1 1 8
INFERNIX_TEST_ARTIFACT=out/qwen3_8_27b.ninfer \
  build/tests/infernix_qwen3_5_dflash2_real_test 7 1 0 2 bf16 1 0
INFERNIX_TEST_ARTIFACT=out/qwen3_8_27b_nvfp4.ninfer \
  build/tests/infernix_qwen3_5_dflash2_real_test 2 0 0 2 int8
```

Arguments are K, Graph enabled, optimized head enabled, maximum B, target KV (`bf16`, `int8`,
`fp8`, `nvfp4`, `k8v4`, `vq2` or `k4v2`), Vision enabled, and extra Device StateImage slots. Defaults are
`15 1 1 8 bf16 0 3`. An optional eighth argument, a `--draft-tree-nodes` table such as
`16,12,12,10` or `auto`, runs the fixture through DFlash2 tree verification (with `auto`, the
same-seed replay check is skipped because the chosen widths follow measured round time); CTest
runs a table on INT8, K8V4, NVFP4, VQ2 and K4V2 as `infernix_qwen3_5_dflash2_tree_<kv>_real_test`. Run
GPU integration tests serially. The
individual Op suites remain the numerical/state-transition oracle; the fixed Engine fixture does
not define bit parity across arbitrary floating-point routes.

The tree greedy test decodes six prompts greedily (160 tokens each) with a 16-column tree and
compares every committed token with a fresh one-token greedy prefill of the exact committed prefix,
scoring each disagreement on a CausalScoring Engine. A context built by decode rounds rounds
differently from one prefill, so chain decoding disagrees with this oracle too (about 3 % of the
positions, at low-confidence tokens). A state or compaction error after a side branch would leave
the target's choice at most later positions; the test fails above 10 % disagreements or when a
committed token is more than 6 nats below the oracle's.

```bash
cmake --build build -j --target infernix_qwen3_5_dflash2_tree_greedy_real_test
INFERNIX_TEST_ARTIFACT=out/qwen3_8_27b_nvfp4.ninfer \
  build/tests/infernix_qwen3_5_dflash2_tree_greedy_real_test int8 16
```

Arguments are the target KV, the tree table (`auto` for automatic widths, `0` runs chain decoding
against the same oracle), output tokens per prompt and K; defaults are `int8 16 160 7`. CTest
runs it on INT8, K8V4, NVFP4, VQ2 and K4V2.
