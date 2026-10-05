# Linear benchmark contract and preset suites

## Scope

This document is the current authority for `bench/ops/linear_bench.cu`. It defines the pure Linear
benchmark's commands, timing contract, metrics, preset suites and extension rules. The benchmark does
not change the production Linear route and does not reselect any route winner.

The benchmark measures only:

```text
ninfer::ops::linear(x, w, out, policy, workspace, stream)
```

This is a permanently retained public benchmark. The measured calls of single, sweep, suite and profile
must all go through the public entry point above; fixtures can construct inputs by the public weight
format, but the benchmark must not include Linear private launcher/plan/dispatch headers, must not call
`ninfer::ops::detail`, and must not offer candidate, kernel or route forcing options.

Q4/Q5/Q6/Q8 LinearAdd, LinearSwiGLU, LinearPair and other fused Ops are not part of this benchmark. They
continue to be measured independently by their own benchmarks.

Q4/Q5/Q6/Q8 and BF16 use the existing A16 route. The following NVFP4 exact problems support both the A16
and A4 policies and are used as a permanent development surface; they are not added to the model
suites:

```text
[14336,5120]  attention input
[16384,5120]  GDN input
[34816,5120]  MLP gate/up development parent
[ 5120,6144]  attention/GDN output
[ 5120,17408] MLP down
```

`--policy a4` measures the complete public call, and the production resolver chooses the already
qualified A16 or A4 route from the exact geometry and T. The benchmark neither duplicates nor infers
that private choice.
The tuning ranges, the 512/1024 large-T anchors, the key points and the final report format are defined
uniformly by the [Linear tuning and reporting specification](linear-tuning.md).

## 1. Use cases

The benchmark serves four concrete needs.

### 1.1 Single-point performance

Explicitly specify the weight type, `N`, `K` and `T` to measure one production Linear point:

```bash
./build/bench/ninfer_linear_bench \
  --qtype q4 --policy a16 \
  --n 4096 --k 5120 --t 8
```

The numeric `(N,K)` is the main entry point. The benchmark does not duplicate the production selector's
complete shape admission table; unsupported points are rejected by the public `linear()` and its format
selector.

The command for a BF16 decode exact point is:

```bash
./build/bench/ninfer_linear_bench \
  --qtype bf16 --policy a16 \
  --n 14336 --k 5120 --t 1
```

The permanent NVFP4 A16 decode point is:

```bash
./build/bench/ninfer_linear_bench \
  --qtype nvfp4 --policy a16 \
  --n 14336 --k 5120 --t 1
```

Its main W4A4 MMA point is:

```bash
./build/bench/ninfer_linear_bench \
  --qtype nvfp4 --policy a4 \
  --n 14336 --k 5120 --t 1024
```

The other four permanent NVFP4 problems use the same numeric entry point, for example:

```bash
./build/bench/ninfer_linear_bench \
  --qtype nvfp4 --policy a4 \
  --n 16384 --k 5120 --t 1024

./build/bench/ninfer_linear_bench \
  --qtype nvfp4 --policy a4 \
  --n 34816 --k 5120 --t 1024

./build/bench/ninfer_linear_bench \
  --qtype nvfp4 --policy a4 \
  --n 5120 --k 6144 --t 1024

./build/bench/ninfer_linear_bench \
  --qtype nvfp4 --policy a4 \
  --n 5120 --k 17408 --t 1024
```

The workspace is allocated before the timed region according to the public capacity query; all launches
and traffic of activation quantization and GEMM are inside one timed `linear()`. A GEMM-only result after
pre-quantization is not a production metric of this benchmark.

FP8 uses the same entry point, for example:

```bash
./build/bench/ninfer_linear_bench \
  --qtype fp8 --policy a8 \
  --n 14336 --k 5120 --t 1024
```

`--execution graph` measures the complete Op inside a CUDA Graph; `--graph-calls N` puts `1..64` calls
into one timed graph and reports the time per call. The default is eager. Both timing modes call the
same public Op.

### 1.2 NCU single point

Profile mode must accept exactly one exact point:

```bash
ncu --profile-from-start off ... \
  ./build/bench/ninfer_linear_bench \
  --qtype q4 --policy a16 \
  --n 4096 --k 5120 --t 8 --profile
```

The program completes allocation, payload initialization, warmup and L2 flush outside the capture, then:

```text
cudaProfilerStart()
    one public linear() call
    stream synchronize
cudaProfilerStop()
```

NCU therefore sees the one or more kernel launches that host launcher actually issues, without also
capturing weight initialization, the copy roofline probe, the Tensor Core probe, warmup or repeated
measurements.

The following distinctions must still be kept:

- a benchmark point is one public Linear call;
- the selector returns a host launcher;
- a host launcher can issue one or more kernel launches;
- what NCU shows are concrete kernel template instances;
- the benchmark does not mislabel a schedule, template instance or host launcher as a kernel.

### 1.3 Small-T sweep

Fix `(qtype,N,K,policy)` and scan a T range continuously:

```bash
./build/bench/ninfer_linear_bench \
  --qtype q4 --policy a16 \
  --n 4096 --k 5120 \
  --sweep 1:32:1 \
  --csv-out profiles/bench/q4_4096x5120_t1_32.csv
```

The Vision step domain can be expressed directly as:

```bash
--sweep 4:512:4
```

Each T outputs the median latency change from the adjacent point. The sweep is used to find
unreasonable latency steps, observe existing production route seams, and provide leads for separate
route measurement tasks; every point still calls only the public Linear. It does not compare forced
candidates, set automatic performance gates, or duplicate selector, crossover or candidate-legality
tables inside the benchmark.

### 1.4 Representative model suites

To avoid hand-writing `(qtype,N,K,T)` every time, the benchmark registers a small number of
representative 27B/35B points. One suite call runs all points in order:

```bash
./build/bench/ninfer_linear_bench --suite qwen3_6_27b
./build/bench/ninfer_linear_bench --suite qwen3_6_35b_a3b
./build/bench/ninfer_linear_bench --suite all
```

Running without arguments still prints usage and does not implicitly start a heavy suite. A suite is an
explicit convenience, not a correctness matrix, a complete artifact inventory or a second production
admission registry.

## 2. Representative suites

### 2.1 Default T sets

Suite entries use two fixed sampling classes:

| T class | Default T | Purpose |
|---|---|---|
| `Continuous` | `1,16,128,1024` | Observes the small-T bandwidth phase, route transitions and the large-T compute phase together |
| `VisionStep4` | `4,128,1024` | Conservative common sampling that is legal for all current Vision geometries |

These points are not a copy of the production route boundaries and do not try to cover every selector
case. Complete continuous seam checks are done by an explicit `--sweep`.

A suite generates the weight only once per `(qtype,N,K)` and allocates the activation/output once for
that entry's maximum T. Different T values reuse the same allocations, and large weights are not
rebuilt.

### 2.2 `qwen3_6_27b`

The 27B core suite registers only the main Text/MTP/head geometries the current model actually uses
through the pure public Linear:

| Label | QType | `(N,K)` | T class | Actual role |
|---|---|---:|---|---|
| `27b.output_head` | Q6 | `(248320,5120)` | Continuous | full output head |
| `27b.draft_head` | Q4 | `(131072,5120)` | Continuous | optimized proposal head |
| `27b.gdn_output_gate` | Q5 | `(6144,5120)` | Continuous | GDN Z row view |
| `27b.mtp_input` | Q8 | `(5120,10240)` | Continuous | MTP input projection |
| `27b.mtp_attention` | Q8 | `(14336,5120)` | Continuous | packed MTP Q/K/gate/V projection |
| `27b.mtp_gate_up` | Q8 | `(34816,5120)` | Continuous | MTP gate/up |
| `27b.mtp_down` | Q8 | `(5120,17408)` | Continuous | MTP down |

The 27B Vision suite registers the seven geometries actually called through the public Linear:

| Label | QType | `(N,K)` | T class | Actual role |
|---|---|---:|---|---|
| `27b.vision_patch` | Q6 | `(1152,1536)` | VisionStep4 | patch projection |
| `27b.vision_qkv` | Q4 | `(3456,1152)` | VisionStep4 | attention QKV |
| `27b.vision_attn_out` | Q5 | `(1152,1152)` | VisionStep4 | attention output |
| `27b.vision_fc1` | Q4 | `(4304,1152)` | VisionStep4 | MLP expansion |
| `27b.vision_fc2` | Q5 | `(1152,4304)` | VisionStep4 | MLP contraction |
| `27b.vision_merger_fc1` | Q8 | `(4608,4608)` | VisionStep4 | merger expansion |
| `27b.vision_merger_fc2` | Q8 | `(5120,4608)` | VisionStep4 | merger output |

The packed/fused parents of the basic Text Attention, GDN input and MLP are not added to the pure Linear
suite just because they exist in the artifact. They are currently owned by independent semantic Ops such
as Attention/GDN/LinearSwiGLU/LinearAdd and should be measured in the corresponding Op benchmarks.

### 2.3 `qwen3_6_35b_a3b`

The 35B-A3B core suite registers the main head, MTP and DFlash geometries that are really used through
the pure public Linear today:

| Label | QType | `(N,K)` | T class | Actual role |
|---|---|---:|---|---|
| `35b.output_head` | Q6 | `(248320,2048)` | Continuous | full output head |
| `35b.draft_head` | Q4 | `(131072,2048)` | Continuous | optimized proposal head |
| `35b.mtp_projection` | Q8 | `(2048,4096)` | Continuous | MTP input/output projection geometry |
| `35b.dflash_feature` | Q8 | `(2048,16384)` | Continuous | DFlash conditioning projection |

The 35B-A3B Vision backbone shares the first six geometries with 27B, and the merger output goes into the
2048 hidden:

| Label | QType | `(N,K)` | T class | Actual role |
|---|---|---:|---|---|
| `35b.vision_patch` | Q6 | `(1152,1536)` | VisionStep4 | patch projection |
| `35b.vision_qkv` | Q4 | `(3456,1152)` | VisionStep4 | attention QKV |
| `35b.vision_attn_out` | Q5 | `(1152,1152)` | VisionStep4 | attention output |
| `35b.vision_fc1` | Q4 | `(4304,1152)` | VisionStep4 | MLP expansion |
| `35b.vision_fc2` | Q5 | `(1152,4304)` | VisionStep4 | MLP contraction |
| `35b.vision_merger_fc1` | Q8 | `(4608,4608)` | VisionStep4 | merger expansion |
| `35b.vision_merger_fc2` | Q8 | `(2048,4608)` | VisionStep4 | merger output |

The 35B routed expert `[262144,2048]` gate/up and `[524288,512]` down are not added to the pure Linear
suite. They belong to the `sparse_moe` execution contract and are currently not geometries registered
with the pure Linear selector either. Likewise, Attention/GDN/DFlash layer projections remain the
responsibility of their fused Op benchmarks.

### 2.4 `all`

`all` is the union of the two model suites. A completely identical `(qtype,policy,N,K,T)` point is
executed only once; a shared Vision point is not timed twice just because both targets use it. A result
can keep multiple role labels, but there is only one timing row.

## 3. Request pipeline

```text
CLI
 |
 +-- exact point: qtype + policy + N + K + T
 |
 +-- sweep:       qtype + policy + N + K + T range
 |
 +-- suite:       curated target entries + entry-local default T class
 |
 v
expand to BenchPoint(qtype, policy, N, K, T, labels)
 |
 v
group points by (qtype, policy, N, K)
 |
 v
allocate and initialize one packed weight
allocate x/out for max T in the group
 |
 v
for each T:
    bind [K,T] and [N,T] Tensor views
    warm up
    flush L2 outside timed interval
    time public ops::linear()
 |
 v
derive model bytes, useful FLOPs, fixed-spec ratios and measured-read ratio
 |
 v
compact table and optional CSV
```

Suite expansion only provides convenience points; execution still goes through exactly the same
fixture, public API, timer and metric computation as an explicit single point, with no suite-specific
dispatch.

## 4. Theoretical traffic

Let:

```text
K_pad = align_up(K, 128)
groups = N * K_pad / group_size
```

The format weight bytes are the sum of the storage planes the kernel needs to consume, excluding gaps
between planes that exist only to align start addresses and are never read:

| QType | group | bytes/group | weight bytes |
|---|---:|---:|---:|
| q4_g64_fp16 | 64 | `32 code + 2 scale` | `groups * 34` |
| q5_g64_fp16 | 64 | `32 low + 8 high + 2 scale` | `groups * 42` |
| q6_g64_fp16 | 64 | `32 low + 16 high + 2 scale` | `groups * 50` |
| q8_g32_fp16 | 32 | `32 code + 2 scale` | `groups * 34` |
| NVFP4 | 16 | `8 code + 1 E4M3 scale` | `groups * 9` |
| BF16 | — | direct BF16 | `2 * N * K` |
| fp8_e4m3fn_row_bf16 | one row | `K code + 2 scale` | `N * (K + 2)` |

The theoretical minimum traffic of one Linear is:

```text
model_bytes = weight_bytes + 2*K*T + 2*N*T
effective_GB/s = model_bytes / seconds / 1e9
dram_spec_pct = effective_GB/s / 1792 * 100
read_ceiling_pct = effective_GB/s / 1674.5 * 100
```

This is a model floor derived from the public representation, not the physical DRAM traffic a profiler
observes. It does not count:

- cache-line transaction amplification;
- route-private rereads of the same weight tile;
- intermediate traffic of the workspace, split/reduce or composite launchers;
- the DRAM traffic reduction from cache hits.

These questions need NCU to answer. The benchmark no longer derives `weight_replay_lower_bound_bytes`
from the launcher column tile.

`1792 GB/s` remains the fixed hardware specification, used for the existing `DRAM_%` and the fixed-spec
roofline. The additional `READ_%` uses `1674.5 GB/s`, the result of the 4 GiB `uint4` pure-read run of
`tools/hbm_bandwidth_probe.cu` on the RTX 5090, which is the read-only limit this machine has been
measured to sustain. It only provides an actually achievable utilization for read-dominated Linear; it
does not replace the fixed-spec metric or change the `model_bytes` basis of one logical weight read. The
copy probe reads and writes at the same time and is not the applicable limit for this kind of GEMV.

## 5. Mathematical work and route-neutral metrics

All weight types use the same mathematical GEMM work:

```text
useful_flops = 2*N*K*T
useful_TFLOP/s = useful_flops / seconds / 1e12
```

Dequantization, bit decoding, padding, split-K repeated work and tile rounding are not added to
`useful_flops`, and `executed_tflops` is not derived from a private schedule. This guarantees that
different implementations are compared using the same mathematical work.

`AllowA4` is a permission, not the actual activation-compute profile; under the same policy, different
exact geometries and T values can choose different routes. The uniform fixed-spec reference is the
memory floor; only points explicitly associated with an actual MMA path output `tensor_profile`,
`tensor_peak_tflops` and `tensor_peak_pct`, and the rest are left empty.
These fields cannot be inferred from the activation policy alone:

```text
memory_floor_us = model_bytes / 1792 GB/s
memory_floor_pct = memory_floor_us / median_us
```

## 6. Timing contract

Ordinary single points, sweeps and suites use the same cold-cache timing contract:

1. allocation, payload initialization and the first public call are not timed;
2. before each sample, a fixed 256 MiB buffer evicts the L2;
3. the flush completes before the CUDA event start;
4. the events enclose exactly one complete public Linear call;
5. the primary result uses the median, and min and p95 are also recorded;
6. the defaults are warmup `3` and repeat `20`, which can be overridden explicitly;
7. no extra warm-cache measurement set is run for each point.

If an independent hot-cache question really arises in the future, a clearly named opt-in mode should be
added; the doubled cold/warm work must not be made the default behavior of every command again.

## 7. Output

The console header always prints:

```text
gpu=RTX 5090
dram_spec=1792 GB/s
sustained_read=1674.5 GB/s
cache=cold
```

A single-row result keeps:

```text
label qtype policy N K T median_us min_us p95_us
effective_GB/s DRAM_% READ_%
useful_TFLOP/s tensor_profile tensor_peak_pct memory_floor_pct t1_linear_extrapolation
```

A sweep additionally outputs the `delta_%` between adjacent T values. The CSV can add the weight/x/out
byte breakdown, warmup, repeat and environment versions, but does not restore the following fields:

- measured stream-copy ceiling;
- measured Tensor Core ceiling;
- candidate;
- kernel variant;
- inferred weight replay;
- inferred executed FLOPs;
- duplicated launcher tile metadata.

The benchmark does not need to claim which kernel was selected. A single-point NCU run shows the real
kernel instances, and the production selector source is the only executable authority for the host
launcher route.

## 8. Registration rules

### 8.1 New production shape

Adding a new production Linear shape does not require modifying the benchmark:

1. the selector and the public numerical test complete the registration;
2. it can be measured immediately with the numeric `--n/--k/--t/--qtype`;
3. it is added to a model suite only when it becomes a high-frequency or architecturally representative
   geometry of 27B/35B.

A suite entry contains only:

```text
label, qtype, policy, N, K, T class
```

It must not carry launcher, schedule, kernel, tile, Full/Predicated, workspace or route-boundary
metadata.

### 8.2 New route

Adding or replacing a host launcher does not modify the benchmark. Single, sweep and suite always call
the public Linear, so they naturally measure the production route the selector currently returns.

Candidate selection follows the recommended workflow defined in
[`op-development.md`](op-development.md#7-performance-evidence):

1. within the development task scope of that format or route, write a temporary benchmark/sweep, calling
   the private launcher directly if necessary;
2. determine a fixed winner or crossover over the candidates' common legal domain, with the same inputs
   and cache/timing conditions;
3. write the decision only into the production selector;
4. revalidate correctness and final performance through the public Linear;
5. delete the temporary sweep, candidate forcing and private entry points kept only for comparison.

Generic candidate forcing, private legal domains, crossover mirrors, launcher catalogs or the plan layer
must not be reintroduced into the long-term pure Linear benchmark. The public `--sweep` only observes the
overall behavior of the final selector and does not take on candidate selection.

### 8.3 New weight/activation compute type

A new benchmark type is added only when all of the following are complete:

1. the public policy corresponds to a real, reachable production path;
2. the packed-weight fixture supports that persistent format;
3. the model-byte formula is explicit;
4. the public numerical suite has qualified the routes actually reachable under that policy.

A policy is only a permission, and the long-term benchmark does not pass off the permission itself as
low-precision execution. All current preset suites explicitly use `A16Only`; NVFP4 AllowA4 is kept as
explicit points of numeric geometry.

## 9. Recorded verification and measurements

The completed verification and RTX 5090 measurement records are kept below, with the timing conditions
listed with each item:

1. the `ninfer_linear_bench` Release target builds;
2. the 49 points of the 27B suite and the 37 points of the 35B-A3B suite all execute through the public
   Linear;
3. `all` merges into 68 unique points, and shared Vision exact points execute only once;
4. the Text `T=1..3` sweep and the Vision `T=4,8` step-4 sweep both output continuous results;
5. NCU `--profile-from-start off --launch-count 1` captures only the `q4_rowsplit_gemv_kernel` instance
   produced by one public Linear call;
6. the weight-byte result of Q4 `[4096,5120], T=1` is `11141120`, equal to
   `4096*5120*(4/8+2/64)`; the console and CSV use the same computation;
7. the BF16 exact point above executes through the public Linear; the median of 500 cold-cache samples
   was `95.520–97.536 us` across repeated runs, i.e. `1505.5–1537.3 GB/s`;
8. relative to the `1674.5 GB/s` pure-read probe, this range is `89.91%–91.80%` of the actually
   achievable read bandwidth;
9. the DRAM read of the final single NCU capture is `146825728` bytes, while one logical read of the
   weight plus activation is `146810880` bytes; the extra read is only `14848` bytes, with no weight
   replay;
10. the output keeps route-neutral useful TFLOP/s and memory-floor metrics and does not infer the actual
    activation-compute profile from AllowA4;
11. the A4 Linear of all five NVFP4 exact problems passes the same exact-decode/naive-FP64 oracle directly
    at `T=17`, a representative cp.async point and the main `T=1024` TMA point;
12. on RTX 5090, CUDA 13.1, cold cache, the complete quantization + GEMM results of the five NVFP4 A4
    `T=1024` problems are:

    | `[N,K]` | Median | Useful throughput | Dense FP4 peak |
    |---:|---:|---:|---:|
    | `[14336,5120]` | `152.576 us` | `985.24 TFLOP/s` | `58.79%` |
    | `[16384,5120]` | `174.784 us` | `982.92 TFLOP/s` | `58.65%` |
    | `[34816,5120]` | `390.112 us` | `935.81 TFLOP/s` | `55.84%` |
    | `[5120,6144]` | `72.992 us` | `882.62 TFLOP/s` | `52.66%` |
    | `[5120,17408]` | `197.888 us` | `922.42 TFLOP/s` | `55.04%` |

    `[34816,5120]` uses 5 warmups and 40 samples; the other four rows use 5 warmups and 30 samples.

The benchmark does not take responsibility for numerical correctness; each weight/activation-compute
profile remains the responsibility of the public Linear conformance suite and the unified CPU FP64 GEMM
oracle.
