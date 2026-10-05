# Hybrid prefix cache: design and decisions

The hybrid prefix cache (HPC) is `ninfer-serve`'s default prefix-cache mode
(`ContextCacheOptions::mode = ContextCacheMode::Hybrid`). It reuses prompt prefixes through
content-addressed 64-token KV blocks and sparse recurrent-state snapshots, kept on the Device and in
one pinned Host tier. The original system
([Resource scheduling and context cache](resource-scheduling-and-context-cache.md)) stays available
with `--use-original-prefix-caching` (`ContextCacheMode::Original`, still the Engine option
default); the two modes coexist by explicit owner request. Throughout this document "Legacy" names
that original mode.

This document is the authority for the Hybrid mode. §0–§11 describe the design as implemented,
marking the parts of the original design that were not built; §12–§15 cover optional features,
tests, configuration and risks; the [decision record](#16-decision-record) (§16) lists the decisions
taken while building and operating it, with the measurements behind them and the alternatives that
were tried and reverted.

Scope: an alternative to NInfer's prefix-reuse, checkpoint-retention and cache-pressure
system for `Qwen3_5ForCausalLM` / `Qwen3_5MoeForCausalLM` on one RTX 5090 (`sm_120a`), with
`max_concurrency` 1..8, every KV profile (BF16, INT8-G64, FP8-E4M3FN-row256, NVFP4-G16, K8V4, K4V2,
VQ2),
and every speculative backend (none, MTP, DFlash, DFlash2). `Qwen4ExpForCausalLM`
(Qwen3.8-Flash-Next) has its own binding of the same index and frontend keys (§17).

Coexistence rules:

- The mode is fixed at Engine construction. The original-only option `--device-state-slots`
  requires `--use-original-prefix-caching`; Hybrid-only options are rejected with it.
  `--host-context-mib` applies to both modes (in Hybrid mode it sizes the Host slab pool);
  `--no-prefix-reuse` disables the cache, runs the original manager disabled, and is rejected
  with `--use-original-prefix-caching` and with every Hybrid option.
- Hybrid mode reuses the active-execution machinery unchanged: the Engine core and scheduler,
  incremental unit reservation, preemption and Replay recovery, the binding transaction,
  `LogicalKVPageStore`, `KVAddressSpaceStore`, execution rows, the `StateImageStore` images and the
  prefill/decode/speculative paths. It replaces only admission-source selection, retention, capture
  and cache eviction. Tree nodes hold non-writer references to logical pages; active address spaces
  map those pages exactly as a checkpoint prefix fork does.

---

## Status

Hybrid mode configures itself: the only capacity a deployment chooses is `--host-context-mib`
(default 8192, 0 = Device only; Qwen3.8-Flash-Next: default 4096, must be positive, §17); every
other value is derived from the rest of the configuration (§14.2).

| area | state |
|---|---|
| §5 index, §9 eviction (device LRU, host superseded-first then GDSF, dead KV), §7.1 tap planner | implemented; host-only unit tests (`ninfer_prefix_cache_index_test`) |
| §5.4 automatic Device sizing | implemented: in Hybrid mode the Main pool is not clamped to `C·L`, and `ninfer-serve` defaults `--kv-capacity` to `auto`, so free VRAM becomes Device block cache |
| §5.4 unified Host slab pool | implemented: KV blocks, snapshot images (split over slabs) and snapshot tails share one pinned pool; GDSF and the dead-KV sweep decide the split at run time |
| §6 admission as the Engine's binding transaction | implemented: staging reserves the first unit's Device pages and the state slot, Host restores run on a dedicated restore stream, and completion forks the lane at once; its Device work queues behind the copies it reads, layer by layer (§6.4, §6.5, §12.1). Later units reserve incrementally; preemption recovers by Replay (§9.5) |
| §6.6 prefetching the blocked head | implemented: Host-only path blocks are copied into spare Device cache while the FIFO head waits |
| §7.1 taps | implemented: exact taps (explicit, generation opener, structural) split the chunk; flexible taps (prompt tail, ladder) are realized at chunk boundaries at no extra forward pass |
| §7.4 tap publication | implemented: deferred until the anchoring blocks commit (a frontier inside a block, or an MTP backend one token behind); exact-frontier tails are copied into cache-owned pages |
| §7.6 block publication, write-through | implemented: blocks join the tree at every commit; a lane's blocks are written to Host in one batch when it releases them |
| §7.7 endpoint snapshots at finish and consistent abort | implemented |
| §6.2 cost model | implemented: the Engine's context-cost coefficients (prefill and Host-to-Device transfer) rank sources and value snapshots |
| §5.5 persistence across restarts | implemented (opt-in `--prefix-cache-file`) |
| §12.2 in-flight prefix coalescing | implemented: requests sharing a new prefix with a lane still prefilling wait for its snapshot at the divergence instead of prefilling the prefix again |
| §7.2–§7.3 zero-split GDN state tap and phase alignment | not implemented: an exact tap costs one prefill split (about 15 ms per turn on 27B); flexible taps avoid it, and requests resuming from an endpoint skip the opener tap (§7.1) |
| §11.2 KV transfer Op | copy-engine path only (`cudaMemcpy2DAsync` runs over consecutive pages and slabs) |
| §6.3 persistent backfill proof | not issued: a blocked FIFO head is never overtaken (a proof with a growth reserve was tried and reverted, §16.3) |
| §12 optional features other than 12.1 and 12.2, §13.2 Op qualification | not implemented |
| §13.3 real-artifact scenarios | `ninfer_qwen3_5_hybrid_prefix_real_test`: Host vs Device restore exactness (with and without MTP), generation-opener and system-block reuse, Device-only mode, protocol cache hints, Vision, persistence across a restart, cancellation mid-prefill (with and without DFlash2); `NINFER_HYBRID_KV_DTYPE` runs them for every KV storage (bf16, int8, fp8, nvfp4, k8v4 pass). `ninfer_ngram_concurrent_real` runs on Hybrid with `NINFER_NGRAM_TEST_CONTEXT_CACHE=hybrid` |

## 0. Summary

| | Legacy mode | Hybrid mode |
|---|---|---|
| Unit of reuse | owner/checkpoint (private continuation, shared prefix, 5 checkpoint kinds) | content-addressed 64-token KV block + sparse state snapshot |
| Identity | per-owner token ledgers, session index, shortlist keys, markers | exact token path in one radix tree (hash only for lookup) |
| KV sharing across sessions | only through an optional shared-prefix publication that must beat a private baseline | unconditional: identical blocks are one physical page |
| State checkpoint capture | prefill chunk split at the frontier + copy | a copy of the committed state at a chunk boundary; only exact taps (client breakpoints, generation opener, structural boundaries) split a chunk (§7.1) |
| Admission | bounded heuristic search over complete targets; up to 250 ms planning allowance per admission boundary | O(prompt blocks) lookup + arithmetic capacity check, no search |
| Pressure | monotone degradation graph, joint post-state projection, ordered stage peaks, one global transaction | independent eviction per tier: device LRU and host GDSF (GreedyDual-Size-Frequency) |
| Host tier | variable-extent arena with allocator geometry and a separate StateImage slot count | one pinned slab pool; KV blocks and snapshots share it with no fixed split between them |
| Device ↔ host | demote on pressure (copy before release) | asynchronous write-through when a lane releases its blocks; device eviction never copies |
| Code | ≈25 k lines in `runtime/engine/context_cache`, `program/{planning,transactions,storage}`, host arena, resource contracts | ≈6.1 k lines in `runtime/prefix_cache`, `program/prefix` and the Engine's `HybridResourceManager`, plus ≈1.8 k lines of integration |

What the design changes for users:

- **More reuse.** Every committed full block of every request becomes reusable, including
  cancelled and partially prefilled requests. Snapshots are deduplicated by content, so one
  snapshot serves every session that shares its prefix. Tap density is limited only by host memory,
  not by per-continuation anchor counts or catalog slots.
- **Lower TTFT.** No planning search. Host restores never hold an admission: they overlap the
  restored request's first prefill pass layer by layer while other lanes keep decoding (§6.5).
  Only exact taps split a prefill chunk, and the template's rewrite frontiers no longer do.
- **Throughput unchanged.** Kernels, page layout, block tables and CUDA Graph keys are unchanged.
  A flexible tap costs one Device copy of the state image (about 0.25 ms) at a chunk boundary.
  Cold prefill and decode match Legacy at the same pool size (§16.1).
- **Better memory efficiency.** Shared prefixes are stored once. KV that no snapshot can reach
  ("dead KV") is reclaimed first. The host pool has no fragmentation and no split between KV and
  state.

---

## 1. Why a second design

Evidence is taken from the Legacy sources and documents.

1. **Ownership instead of content.** Reuse depends on which *owner* holds a checkpoint: private
   continuation, shared prefix, SessionIndex binding, or one of five checkpoint kinds
   (`SessionEndpoint`, `TurnClosure`, `ResponseReplay`, `LongAnchor`, `SharedStablePrefix`). It is
   also bounded by catalog slots (`max_private_continuations`, `max_shared_prefixes`,
   `max_long_anchors_per_continuation`). Two sessions with the same 30 k-token system+tools prefix
   share KV only if a shared prefix was published. Publication must be *strictly better than the
   private-only baseline* (resource-scheduling §7.2, §8.3).
2. **Planning cost on the critical path.** Materialization runs a bounded heuristic search over
   complete pressure targets. The Program projects joint post-states, stage peaks and Host
   allocator geometry for each target (§7.3–§8.7). Every admission boundary has a 250 ms
   planning allowance. That time sits directly in TTFT for the head request and
   stalls the worker for everyone else.
3. **Split chunks.** A checkpoint at an arbitrary frontier splits a prefill chunk (`text.cpp`
   `prefill_split_frontier_`), producing a short, inefficient chunk. `rewrite_execution_frontiers`
   adds more splits so replay matches the original decomposition.
4. **Host fragmentation and partitioning.** Host KV is a variable-extent arena whose feasibility
   depends on extent geometry, not free bytes (paged-kv §5.3). Host StateImages are a separate
   slot count. `--host-cache-mib` resolves a *static* split between them at startup.
5. **Fragility.** Correctness depends on invariants spread across the ResourceManager and Program
   boundary: capability generations, resource revisions, owner edges, claims, seal windows,
   borrowed-read ownership, and absolute ResourceResult adoption. Recent fix history
   (planning races, shared catalog saturation, lease reclaim) shows the maintenance cost.
6. **Size.** About 25 k lines: `context_cache/` 7.0 k, `planning/` pressure/recovery/request 6.6 k,
   `transactions/` 4.5 k, `storage/` 5.2 k, Host arena + prefix identity + resource contracts
   1.8 k.

The fundamental constraint does not change. Qwen3.5 is hybrid: 48/64 (27B) or 30/40 (35B-A3B)
layers are Gated DeltaNet with a recurrent state that cannot be rolled back. A prefix is therefore
reusable only at a position where the complete recurrent state exists. HPC keeps that rule and
builds everything else around making such states **cheap to create, cheap to store, and exactly
addressable**.

---

## 2. Prior art used

| Source | Idea adopted |
|---|---|
| vLLM automatic prefix caching | Full blocks keyed by (parent, tokens, extra keys). Reference-counted. Freed blocks go to an LRU queue in reverse order, so tail blocks are evicted first. Evictable blocks count as free capacity. |
| vLLM hybrid/Mamba prefix caching | Recurrent state is cached only at block-aligned positions. A hit is the deepest aligned position that has state. |
| SGLang RadixAttention / MambaRadixCache | Radix tree over token blocks. Recurrent-state entries attach to tree nodes with their own eviction. Leaf-first eviction. |
| SGLang HiCache | Host tier with write-through, batched or kernel-based non-contiguous transfers, and layer-ordered loading. |
| llama.cpp context checkpoints | A few full-state checkpoints per sequence for recurrent/SWA models. |
| LMCache | Layer-wise pipelined KV loading overlapped with compute. |
| GreedyDual-Size-Frequency (Cherkasova, 1998) | Size- and cost-aware eviction with O(1) aging via an inflation value `L`. |

NInfer-specific additions in the design (1 and 2 are not implemented, §7.2):

1. The GDN **state tap**: chunked GDN already produces the state at every 64-token intra-chunk
   boundary, so exposing it costs one extra state write.
2. **Phase alignment** by identity-transition padding, so taps land on absolute page boundaries.
3. One pinned **slab pool** for both KV and state.
4. **Dead-KV** accounting: KV that no snapshot can reach has zero reuse value.

---

## 3. Model facts that size the design

### 3.1 KV bytes

Per token, all full-attention layers, K+V including scales (from paged-kv §4.3):

| profile | per token/head | Qwen3.6/3.8-27B (16 FA × 4 KV heads) | page (64 tok) | 35B-A3B (10 × 2) | page |
|---|---:|---:|---:|---:|---:|
| BF16 | 1024 B | 64.0 KiB | 4.00 MiB | 20.0 KiB | 1.25 MiB |
| INT8-G64 | 528 B | 33.0 KiB | 2.06 MiB | 10.3 KiB | 0.64 MiB |
| FP8-row256 | 516 B | 32.3 KiB | 2.02 MiB | 10.1 KiB | 0.63 MiB |
| K8V4 | 402 B | 25.1 KiB | 1.57 MiB | 7.9 KiB | 0.49 MiB |
| NVFP4-G16 | 288 B | 18.0 KiB | 1.13 MiB | 5.6 KiB | 0.35 MiB |
| K4V2 | 196 B | 12.3 KiB | 0.77 MiB | 3.8 KiB | 0.24 MiB |
| VQ2 | 132 B | 8.3 KiB | 0.52 MiB | 2.6 KiB | 0.16 MiB |

MTP adds one layer's worth of pages to the bundle (+1/16 for 27B). A DFlash draft with
full-attention layers adds its own BF16 pool. K4V2 and VQ2 also keep an exact recent-key window
in every StateImage (paged-kv §9.3): 1088 × KV heads × 536 B per attention layer, about 39.7 MB
for 27B with MTP (17 layers × 4 heads), so their checkpoint images are that much larger than the
other profiles'.

### 3.2 State image

Fixed per-sequence state (qwen3_5-model "State and numerical boundaries", dflash "Backend storage"):

| component | 27B | 35B-A3B |
|---|---:|---:|
| GDN recurrent, FP32 `[128,128,Hv]` × GDN layers | 144 MiB | 60 MiB |
| conv history, `(2·16·128 + Hv·128)·3` BF16 × GDN layers | 2.81 MiB | 1.41 MiB |
| continuation hidden | 10 KiB | 4 KiB |
| DFlash2 local rings (official config) | 40 MiB | config-derived |
| **total (DFlash2)** | **186.8 MiB** (matches the 195,897,344 B Host StateImage in README) | — |

One 27B snapshot equals about 91 INT8 pages (≈5.8 k tokens of KV), 166 NVFP4 pages, or 47 BF16
pages. **Snapshots dominate storage cost. Their placement is the main policy decision.**

### 3.3 RTX 5090 transfer and compute budget

These are the design's planning values. The implementation ranks and values with the Engine's
context-cost coefficients (§6.2, §11.4). On the measuring machine, whose GPU link runs at PCIe 5.0
x8, restores move 24–27 GB/s.

| operation | value used for design |
|---|---|
| HBM D2D copy of one 186.8 MiB image (read+write) | ≈ 0.25 ms |
| PCIe 5.0 x16 pinned H2D / D2H (practical) | ≈ 50 GB/s each direction |
| H2D snapshot | ≈ 3.9 ms |
| H2D 100 k tokens INT8 KV (3.38 GB) | ≈ 68 ms |
| Prefill, 27B, ~121 k context (README median) | ≈ 4.2 k tok/s, so 100 k tokens ≈ 24 s |
| GDN tap write (144 MiB) inside a chunk (zero-split tap, not implemented) | ≈ 0.09 ms |
| Write-through rate at 4.2 k tok/s INT8 prefill | ≈ 140 MB/s (0.3 % of PCIe) |

Consequences:

- A host-resident prefix is about 300× cheaper to restore than to recompute. **The host tier
  carries most of the reuse value.** The device tier is a latency optimisation.
- A state snapshot is cheap to create (a Device copy of the image takes about 0.25 ms) and cheap
  to move (4–8 ms over PCIe). **Snapshots can be dense. The limit is host bytes, not time.**

---

## 4. Architecture

```text
                 Frontend (gateway threads)
   tokens, media digests, boundary hints, per-block chained hashes
                          │ PreparedPrompt
                          ▼
┌──────────────────────── Engine worker (single mutation owner) ─────────────────────────┐
│ Scheduler (unchanged FIFO) ─ HybridResourceManager (the ResourceManager surface)       │
│      │ quote / reserve / progress / reclaim / prefetch / terminal settlement           │
│      ▼                                                                                 │
│ Program                                                                                │
│  ├─ PrefixIndex (model-agnostic, src/runtime/prefix_cache/)                            │
│  │    block tree ─ nodes: 64 tokens + extra key, device page ids, host slab, pins      │
│  │    snapshots attached to nodes (+ optional tail)                                    │
│  │    device LRU (tail-first)           host eviction (superseded, GDSF, dead KV)      │
│  │    TapPlanner                         CacheCostModel (prefill and H2D costs)        │
│  ├─ Qwen3.5 binding (src/models/qwen3_5/program/prefix/)                               │
│  │    HybridPrefixCache: Host slab pool, residency, restore and write-through copies   │
│  │    admission, taps, block commit, finish; StateImage parts; persistence             │
│  ├─ Device typed KV pools (unchanged layout) + active block tables (unchanged)         │
│  ├─ Device state pool: C active slots + D snapshot slots                               │
│  └─ streams: restore (dedicated), write-through (the Program's transfer stream)        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

Ownership against AGENTS.md boundaries:

- **Core** owns physical primitives: the paged KV cache's record-addressed Host copies, which
  copy runs of consecutive pages and slabs as one strided transfer per plane.
- **Ops** gain nothing: the zero-split GDN tap (§7.2–§7.3) and a KV page transfer kernel (§11.2)
  are not implemented. Taps are Device copies of the committed StateImage.
- **Program** owns the whole cache: exact identity, physical residency, refs, eviction, taps.
  There is no cross-layer logical/physical split, because the cache's policy decisions are
  local physical facts. In Hybrid mode `ResourceManager`, `MaterializationPlanner`,
  `SharedCapturePlanner`, portfolio value and the target/pressure machinery are not constructed
  or called; they remain the Legacy-mode implementation.
- **Runtime** keeps the scheduler and the common request contracts. `HybridResourceManager`
  presents the ResourceManager surface to the Engine core (§8.2), and `runtime/prefix_cache/`
  holds the model-agnostic index, tap planner and cost model.
- **Frontend** supplies tokens, media digests, boundary hints and block hashes. It no longer
  produces cache "candidates", evidence flags or session keys for the cache.

---

## 5. Data model

### 5.1 Block key and exact identity

Block `b` of a prompt covers absolute positions `[64b, 64b+64)`.

```text
BlockKey(b) = ( parent NodeId,
                tokens[64b .. 64b+63],               // TokenId, exact
                extra = media_key(b) )               // 0 when no Vision item overlaps block b
media_key(b) = xxh3_64 over, for each Vision item whose token span overlaps block b and every
               earlier item (Vision changes later MRoPE positions via rope_delta):
               (content_digest[32], modality, grid, span begin/length)
lookup_hash(b) = xxh3_64(lookup_hash(b-1) || tokens || extra), lookup_hash(-1) = seed
```

- `lookup_hash` only selects a child in the parent's child map. A hit always compares
  `tokens` and `extra` exactly. A forced hash collision must not produce a false hit (tested,
  §13.1).
- Everything else that affects KV or state is fixed for the Engine lifetime: model artifact,
  KV profile, speculative backend, RoPE/YaRN options. None of it is part of the key. Positions are
  a function of tokens and media, so they are covered by the key.
- Thinking mode, templates, tools and roles are all rendered into tokens, so they need no key
  field.
- The Frontend computes `lookup_hash` for every full block during preparation, off the worker, and
  stores it in `PreparedPrompt::block_hashes`.

### 5.2 Nodes

The records below list the state the design keeps; `PrefixIndex`
(`src/runtime/prefix_cache/prefix_index.h`) holds it in its own layout.

```cpp
struct BlockNode {                      // host-side; stored in a slab-allocated arena
    NodeId   id; uint32_t generation;   // generation-checked handles
    NodeId   parent;
    uint32_t depth;                     // block index b; the node covers [64b, 64b+64)
    uint64_t lookup_hash, extra;
    TokenId  tokens[64];                // exact tokens (256 B)
    ChildMap children;                  // small open-addressed map keyed by lookup_hash
    // Residency
    int32_t  device_page[kMaxPools];    // per enabled pool; -1 when not device resident
    int32_t  host_slab;                 // -1 when not host resident
    Residency device_state;             // Absent | Resident | Filling(event)
    Residency host_state;               // Absent | Resident | Filling(event)
    // Liveness
    uint32_t active_refs;               // active sequences whose table maps this node
    uint32_t restore_pins;              // in-flight transfers reading/writing this node
    uint32_t live_snapshots_below;      // snapshots whose path includes this node
    uint64_t last_access_tick;
    IntrusiveLink device_lru;           // present iff active_refs==0 && device Resident
    IntrusiveLink host_dead;            // present iff live_snapshots_below==0 && no refs && host Resident
    SnapshotId first_snapshot;          // snapshots anchored here (intrusive list)
};
```

A node is only inserted when **all enabled pools** have committed all 64 positions of the block
(§7.6). A node may be device-only, host-only or both. A node with neither copy is deleted, along
with its subtree (§9.4).

### 5.3 Snapshots

```cpp
struct Snapshot {
    SnapshotId id; uint32_t generation;
    NodeId   anchor;                    // node of the last full block below frontier (root if F<64)
    uint32_t frontier;                  // F: the snapshot state is after tokens [0,F)
    uint8_t  tail_len;                  // F - 64*(anchor.depth+1), 0..63
    TokenId  tail_tokens[63];           // exact; only tail_len are meaningful
    // Tail KV (only when tail_len>0): one private, immutable page bundle
    int32_t  tail_device_page[kMaxPools];  int32_t tail_host_slab;
    // State image
    int16_t  device_slot;               // index into the D device snapshot slots, or -1
    SlabList host_slabs;                // ceil(image_bytes / slab_bytes) slabs, or empty
    Residency device_state, host_state;
    uint32_t restore_pins;
    SnapshotKind kind;                  // Tap | Endpoint | OutputBoundary | Boundary (§9.3)
    bool superseded;                    // §9.3
    // Host eviction (GDSF)
    double   gdsf_h; uint32_t freq;
    uint64_t last_hit_tick;
};
```

- **Tap snapshots** have `tail_len == 0`: frontier on a page boundary, KV fully in the tree.
- **Endpoint snapshots** are taken at request finish or cancel at an exact frontier. They own the
  partial last page, which moves from the finishing sequence without a copy.
- A snapshot is **valid** iff every node on `path(anchor)`, the tail bundle (if any) and the image
  each have at least one resident copy (device or host). Validity is maintained eagerly:
  removing the last copy of a node deletes its subtree's snapshots (§9.4).
- There is at most one snapshot per `(anchor, tail tokens)`. Publishing a duplicate is a no-op
  that refreshes recency.

### 5.4 Memory pools

- **Device KV pools**: unchanged geometry, plane order, block-table matrix and consumer views
  (paged-kv §3–§4, §6.4, §10–§11). A page group is in exactly one state: `Free`, `Active`
  (owned by one sequence, private), `Cached` (owned by a node or snapshot tail), or `Filling`
  (transfer destination). `available = free + evictable_cached`.
- **Main pool size**: automatic sizing keeps the capacity curve. In Hybrid mode the upper clamp
  `M_max = C·L` is removed (only the int32 token representation bounds it): pages beyond the active
  leases are cache-only device capacity. `ninfer-serve` defaults `--kv-capacity` to `auto` in
  Hybrid mode. Active admission still requires `M ≥ max(L, C)`.
- **Device state pool**: `C` active slots plus `D = device_snapshot_slots` snapshot slots in the
  existing slot-indexed layout (`ssm_states [128,128,Hv,C+D]` per layer, and so on). Snapshot slots
  are also the staging destination for taps (§7.4). Default `D = C + 1` with a Host tier (one
  resident snapshot per live conversation plus one staging slot) and `C + 2` without one.
- **HostSlabPool**: one pinned allocation of `host_cache_bytes`, divided into slabs. A slab holds
  one block bundle: the Main page packed with its Host page layout, then the backend page at the
  next 4 KiB boundary; `slab_bytes` is that record rounded up to 4 KiB. A snapshot image uses
  `image_slabs = ceil(image_bytes / slab_bytes)` slabs, not necessarily contiguous (the image's
  packed byte `o` lives in slab `o / slab_bytes`), and a snapshot with a tail one slab more. The
  free list is an index stack, O(1). There is no fragmentation and no geometry check: capacity is
  `free_slabs`. A nonzero budget smaller than one snapshot plus one block is rejected at startup.
  The pool is pinned in chunks of at most 4 GiB of whole slabs (one very large pinned allocation can
  fail or stall under WDDM); a slab never crosses chunks. A strided copy run joins consecutive
  pages only while their slab records share a chunk and advance by one pitch no larger than the
  device's maximum copy pitch. Any two records form a candidate run, so records in different
  chunks, or gigabytes apart in one, must be split explicitly. A 52 GB Host tier otherwise produced
  an invalid `cudaMemcpy2DAsync` that aborted the server.

### 5.5 Persistence (opt-in)

With `persistent_file` (`--prefix-cache-file`), the Host tier outlives the process. `ninfer-serve`
resolves the path to an absolute one at launch. It rejects a directory, a missing parent
directory, or `--host-cache-mib 0`, so an unusable location fails before any caching:

- **Save**, in the worker's orderly stop, which `Engine::stop()` or the Engine's destruction
  starts. `ninfer-serve` calls `stop()` on a confirmed Ctrl+C (a second press within 5 s,
  `serve/stop_control.h`), Ctrl+Break, `SIGTERM` or console close. Running and queued requests
  then fail as Unavailable at the next unit boundary instead of holding the stop until they
  finish; the worker answers them before the cleanup that saves, so no answer waits for the
  save. One more Ctrl+C exits at once without saving: through the shared
  `PrefixCacheSaveControl` (`HybridPrefixCacheOptions::persistent_save`) the save stops before
  its next slab or before the rename, deletes its temporary file and ends, and `ninfer-serve`
  waits up to 2 s for that before exiting. A save not yet begun never begins, and one still in
  the Device drain is not waited for. The console-close handler blocks while `main` unwinds, but
  Windows ends the process about 5 s after a close, which leaves the temporary file until the
  next save. Either way an unfinished save leaves the previous file in place.
  `Program::shutdown_cleanup` releases every lane first, so requests still in flight write their
  committed blocks through, and saves before the cleanup drops the cache. Every Host write
  lands, then every retained (not superseded, §9.3) Host-resident snapshot whose anchor path is
  Host-resident, and exactly those paths, are written parents first with their slab bytes (a
  temporary file renamed into place). Dead KV, superseded snapshots and Device-only content are
  not saved. A worker that ended on a fatal error saves
  nothing, so the previous file stays in place. A worker recovery (OOM or recoverable logic
  error) empties the cache like every cleanup, so a later save holds only what was cached after it.
- **Load**, at Engine construction before any request: the file is used only when its fingerprint
  (absolute artifact path, size and modification time, prefill signature, KV storage, speculative
  backend, RoPE scaling, and the product binary's identity — build id plus executable size and
  time, so any rebuild invalidates it) and its Host geometry equal the running Engine's, and its
  size matches its tables. The file (format 3) holds the header, the block table, the snapshot
  table and only then the slab bytes, so the loader chooses before it reads any slab. A Host tier
  that holds the whole file restores all of it. A smaller one restores whole snapshots, greedily
  by the GDSF density of §9.3 without its time-dependent base — (1 + hits) × the prefill a
  restore saves from the root, per byte of image, tail and anchor path — each with the path
  blocks it needs and no others, and reads only those slabs (`plan_host_restore`); the startup
  log then warns `prefix cache partly restored` with the kept and saved counts and the Host tier
  the file needs. A tier that holds none of the snapshots restores nothing, since a restore keeps
  only the blocks its snapshots resume through; the log then warns `prefix cache not restored`
  with the tier the file needs, and that the save at shutdown replaces the file. Entries are rebuilt Host-only, parents before children, without evicting
  anything, and their first requests restore blocks and images through the ordinary Host restore
  path. A damaged file loads nothing.
- The startup log shows the read of an accepted file as a progress phase (`loading prefix cache`,
  then `prefix cache read | bytes | time | rate`), then reports what was restored or why nothing
  was; the shutdown log reports the save.

---

## 6. Admission (lookup, choice, capacity)

### 6.1 Lookup

`PrefixCache::match(prompt)` runs on the worker:

1. Walk from the root. For block `b < floor((n-1)/64)`, find the child by `block_hashes[b]`,
   then compare tokens and extra exactly. Stop at the first miss. This gives the matched path
   `N_0..N_{k-1}`, `k` full blocks.
2. Collect candidate snapshots: every snapshot anchored on the matched path (walk each node's
   list) whose tail tokens equal `prompt[64·(anchor.depth+1) .. F)`, with **`F ≤ n − 1`**. At
   least one prompt token is always prefilled, so logits, continuation hidden and MTP/DFlash
   bridges come from the normal path. No full-hit path exists.
3. Frontiers inside a Vision token span are not valid resume points. Tap placement never creates
   them, and match rejects them defensively.

Complexity is O(k + snapshots on path). A 240 k prompt is 3,750 hash probes and exact 256 B
compares: tens of microseconds.

### 6.2 Choice

Candidates: root, the deepest valid snapshot `s*`, and the deepest valid snapshot whose image and
path are fully device resident, `s_dev`. At most three. For each:

```text
cost(s) = restore_seconds(s) + prefill_seconds(base=F_s, tokens=n-F_s)
restore_seconds = (host_path_bytes + host_image_bytes + host_tail_bytes) / measured_h2d_bw
                  + per_batch_latency
```

`prefill_seconds` is the existing calibrated machine model (`context_cost.cpp`: linear tokens
plus `B·S + S(S+1)/2` attention pairs, keyed by `prefill_signature`). Pick the minimum. Ties
favour fewer transferred bytes. This is O(1) and deterministic, and the cost model only ranks
feasible choices.

### 6.3 Capacity

```text
need_pages(s) = host_only_path_blocks(s)          // H2D destinations (shared once restored)
              + (s.tail_len>0 ? 1 : 0)            // private COW copy of the tail
              + growth_pages(first unit, base F_s) // the first prefill chunk (or a resumed
                                                   // request's replay coverage)
need_state_slot = 1 active slot (a lane implies a free active slot)
```

A quote keeps `s` when `need_pages(s) ≤ free + evictable`, where evictable pages exclude pages
pinned by this source's own path; otherwise it tries the next cheaper-to-materialise candidate:
`s_dev`, then root. Root is always quoted. Later units reserve their own growth as they run
(§9.5), so admission no longer holds a whole output window up front. When staging cannot make
room from the cache, the shortage goes back to the Engine, which pauses a younger request or keeps
the request queued, exactly as for a checkpoint source.

**Backfill proof** (engine-architecture §5.2) uses the same arithmetic:
`need(borrower) + need(head root) ≤ free + all_evictable + Σ donor entitlements released`.
Hybrid mode does not issue it: a blocked FIFO head is never overtaken.

*Tried and not adopted (2026-09-27).* A proof with this arithmetic plus a growth reserve was
implemented and measured (the change record near the top of this document). Without the reserve,
a borrower can take the pages a long answer would grow into, and that answer then ends early
(bounded completion, paged-kv §6.2). The reserve kept every active sequence, the borrower and the
head able to grow by 16K more tokens:

```text
Σ_active (entitlement + G) + entitlement(c) + G(c)                      ≤ capacity   // now
Σ_borrowers (entitlement + G) + entitlement(c) + G(c) + root(h) + G(h)  ≤ capacity   // donors ended
G(x) = min(pages to x's lease ceiling beyond its entitlement, pages of 16K tokens)
```

It was safe (no answer cut short), but it moved latency rather than removing it. Queued short
requests finished in 1.6 s instead of 35 s. The long request already decoding was slower by about
their prefill time, and the blocked long head started that much later. For workloads dominated by
long main-agent requests that is a loss where it matters, so it was reverted. A version worth
revisiting needs a clear win, for example:

- bound the borrower's cost to the donor: only borrowers whose whole service (prefill and expected
  answer) is small against the donor's remaining time, so the head's delay stays negligible;
- keep donor decode rounds running while a borrower prefills (prefill in short slices between
  rounds), since most of the donor's slowdown was the borrower's prefill blocking its rounds;
- make the reserve unnecessary by suspending a borrower to the Host tier when a donor needs its
  pages (a product change: active-request preemption).

Every unpinned Device-resident cache entry (node or snapshot tail) owns its page bundle
exclusively, so evicting one returns exactly one page to each pool and the arithmetic is exact.
Exact-frontier tap tails are copied into their own pages for this reason (§7.4). Candidates with a
block another admission is still filling are skipped.

### 6.4 Admit

Admission is the Engine's ordinary binding transaction. `hybrid_sources` quotes the request's
sources in preference order (the chosen snapshot path, then a root start); the Engine passes each
to `start_binding` until one reserves, and `poll_context` completes it. Staging
(`start_hybrid_binding`) runs in the `start_binding` call:

1. Re-validate the quoted snapshot and path. Pin every path node up to `F_s` (removed from the
   device LRU) and pin `s`.
2. Plan the binding's first unit exactly as a checkpoint binding does (a prefill chunk, or for a
   resumed request its replay coverage). Evict from the device LRU until the pools hold every page
   the binding needs: the first unit's growth beyond `F_s`, the private tail page, and one per
   host-only path block and host-only tail. When the cache cannot supply them, the pins are
   dropped and the shortage goes back to the Engine, which reclaims (§9.5) or pauses a younger
   request and tries again. Reserve the lane's StateImage destination slot.
3. If anything is host-only, open a restore batch on the dedicated restore stream (ordered after
   the compute stream's queued work, since destinations may have been read by earlier kernels):
   - host-only blocks: materialize destination pages, adopt them as cache references, mark the
     nodes `Filling`, and copy the slabs in runs of consecutive pages and slabs;
   - host-only tail: the same into a cache-owned page (the snapshot's tail is Device-resident
     again afterwards);
   - host-only image: copy the image slabs straight into the reserved destination slot.
   The batch is enqueued in the order a forward pass reads it (§6.5).
4. Prepare the page-prefix forks: path pages are shared by reference, the growth is reserved, and a
   partial tail gets a private destination page.

Completion (`complete_hybrid_binding`, in the next `poll_context`) builds the lane without waiting
for the copies: the compute stream waits for the restore's prelude, the partial tail is D2D-copied,
the forks commit, and the state is reset, D2D-copied from the snapshot's device slot, or restored in
place. The lane takes over the path pins and the snapshot pin is dropped; a restore batch holds its
own pins on the nodes and snapshot it copies until it lands.

Admission never evicts anything that another sequence has pinned. Evictions come from the device
LRU (§9.1) and complete synchronously: dropping a device copy is bookkeeping, since the data is
already backed up or deliberately discarded.

### 6.5 Restoring

A restore batch never holds its admission. `submit_restore` enqueues it on the restore stream in
the order a forward pass reads it, recording an event after each part:

1. the **prelude**: the snapshot tail bundle, every restored block's backend page, and the image's
   continuation hidden and DFlash local state;
2. then, for each model layer in forward order, that layer's state: a full-attention layer's KV
   planes of every restored block, or a linear-attention layer's conv and recurrent state.

`land_restore` makes the compute stream wait for the prelude (activation copies the tail and the
MTP bridge reads the backend pages and hidden) and hands the lane a ticket for the per-layer
events. The lane's first prefill pass waits for each layer's event just before that layer
(`TextContext::run_layers`), so it computes the early layers while later ones are still arriving;
later passes are stream-ordered behind it. The chunk function itself looks the events up by
ticket (`PrefillContext::take_layer_ready`) just before its pass, because `poll()` retires a
landed batch and recycles its events, and program code between chunks polls: a batch already
retired has landed, so the pass has nothing to wait for. A lane released before its first pass makes the compute
stream wait for the whole batch, because the state slot it returns may still be a destination.

The index keeps the batch's nodes and tail `Filling`, and the batch keeps them and its snapshot
pinned, until `poll()` (every admission quote, and the block publication after each KV commit)
or `drain()` sees the last event. So no eviction,
dead-KV reclaim or Host write can free a slab or page a copy still reads or writes. Other admissions
skip `Filling` candidates for that short window.

Consequences:

- A Host revisit no longer waits for the host to notice the copy finished: under concurrency that
  was the rest of the other lane's decode round (~12 ms measured), and with an idle Engine a timer
  tick.
- The copy overlaps the restored request's own first prefill chunk: for a long context, the prefill
  of new tokens hides most of a multi-GB restore.
- Other lanes' decode rounds queue behind that prefill chunk as for any prefill. When the chunk is
  much shorter than the copy (a few new tokens after a very long restored context), they can wait up
  to the copy's remaining time once.

Failure: a CUDA error in a restore job is Engine-wide, as today. A request cancelled while staged
(before activation) waits for the restore stream, returns every destination and reservation, and
releases its pins.

### 6.6 Prefetching the blocked head

When the Device pool cannot hold every conversation, each admission restores most of its path
from the Host tier (the production agent log of 2026-09-26: 2.59 TB, a median 2.7 GB per turn).
For a short turn the first prefill pass waits for those copies, and other lanes' decode queues
behind it. The FIFO head usually waited seconds to minutes for a lane or for pages before that,
so its restore can happen during the wait instead.

- When the head stays blocked (hybrid mode does not backfill, §6.3), the Engine asks the Program to
  prefetch for it. The Program matches the prompt as a quote would, takes the source admission
  would choose, and copies up to 256 of its Host-only path blocks into Device pages. They are
  ordinary cached blocks: `Filling` until the batch lands, then unpinned at the MRU end of the
  device LRU. Admission later finds them Device-resident and restores only the rest.
- Pages come from the free pool and from host-backed cached blocks, least recently used first.
  The head's path is pinned meanwhile. A prefetch never drops a block's last copy, never waits for
  a transfer, and never runs beside an open admission or another prefetch. Active leases may evict
  the prefetched blocks like any cache.
- One batch is about 550 MB for 27B INT8 KV (about 20 ms of PCIe). The Engine retries for the same
  head only while the last attempt copied blocks or the spare room grew, because matching walks
  the whole prompt path.
- A quote whose candidates cross a landing prefetch waits for it (`settle_prefetch`) instead of
  skipping those candidates for a shallower source. A lane's own restore batches are never waited
  for synchronously (§6.5).

In the production log about 57 % of the restored bytes could have been copied this way. The rest
did not fit beside the active sequences while the head waited.

---

## 7. Prefill, taps, decode, finish

### 7.1 Tap planning

`plan_taps(n, base F_s, hints, existing frontiers, Vision exclusions)` returns taps sorted by
position, each `{p, placement}` with `F_s < p ≤ n − 1`, `p` not strictly inside a Vision token span
(a candidate inside one moves to the span start), and no existing or already-planned snapshot at
or near `p` on this path.

Candidates, in priority order:

| priority | source | placement | rationale |
|---|---|---|---|
| 1 | explicit markers (OpenAI breakpoints, Anthropic `cache_control`, C++ `PromptInput` markers) | exact | the client names reuse points |
| 2 | start of the final assistant generation opener | exact | the next turn re-renders the history after it (for example, stripped thinking) |
| 3 | end of tools, end of the leading System/Developer block | exact | compaction and new sessions sharing the preamble |
| 4 | `n − 1` ("prompt tail") | flexible | regenerate / retry of the same prompt |
| 5 | geometric ladder: for k = 0,1,…, `n − G·2^k` (G = `tap_ladder_tokens`), snapped back to a message boundary within the minimum gap | flexible | edits deep in history. Spacing grows with distance, so there are about log2(n/G) taps |

- **Exact** taps split the prefill chunk at `p` (one more forward pass each) and snapshot there.
  Semantic boundaries are where later prompts diverge, so the exact position is worth the split.
- **Flexible** taps never split: the snapshot is taken at the end of the chunk that reaches `p`, or
  at the start of the prompt's final chunk when `p` lies in it (the prompt tail always does). A
  chunk boundary costs nothing beyond the snapshot copy.
- Semantic boundaries (generation opener, structural) closer than 64 tokens form one cluster and
  only its earliest position is kept: a short user turn puts the system-block end next to the
  opener, and a snapshot at the system-block end serves new conversations sharing it while the
  next turn recomputes fewer than 64 tokens.
- A snapshot serves every prompt that diverges after it, so proximity only makes a *later*
  candidate redundant: a non-explicit candidate is dropped when an accepted tap, an existing
  snapshot on the path or the base lies at most 64 tokens before it. Ladder candidates keep
  `tap_min_gap` (default `max(1024, chunk)`) on both sides instead. The opener therefore absorbs
  the prompt tail in chat traffic, leaving one split per request.
- Only markers with explicit evidence (a client-named breakpoint) are priority 1. Protocol-automatic
  markers (OpenAI default caching, Anthropic automatic `cache_control`) are planned as structural
  boundaries, but they mark the conversation's latest turn rather than a prefix conversations
  share.
- Explicit and structural taps (not protocol-automatic ones) are published as `Boundary`
  snapshots: conversations may share them, so a lineage moving past one supersedes it only while
  no other conversation has continued from it (§9.3). A cluster keeps this from any member it
  drops.
- A request that resumed from an endpoint snapshot proves its client echoes generated turns token
  for token (agent loops, preserved reasoning), so its own endpoint serves the next turn and the
  generation opener is not tapped. Measured on a tool-calling agent trace (27B NVFP4, INT8 KV,
  `--preserve-thinking`), the opener split cost about 15 ms of a ~70–130 ms turn TTFT. Requests
  that resumed from a tap keep the exact opener: their client re-renders history.
- Hints are the Frontend's existing `message_boundaries` plus the structural boundaries it already
  computes.
- At most `max_new_taps` (default 8 with a Host tier, 2 without) taps are kept per request.

Tap placement is a function of the prompt, so it is deterministic. Endpoint snapshots are exact
(§7.7).

Exact taps are the only prefill splits in Hybrid mode. The chat template's rewrite execution
frontiers (after each assistant header, after `<think>`, after the reasoning close) split Legacy
prefill for its rewrite checkpoints and execution provenance. Hybrid captures nothing there, so a
Hybrid lane ignores them. Each split is a whole extra pass over the model. A short chat turn used
to prefill as four passes (reuse point, opener, `<think>`, reasoning close). Now it takes two (to
the opener, then the rest), or one when the opener is dropped, unless a client breakpoint or
structural boundary falls inside the new tokens.

### 7.2 Phase-aligned chunked GDN

**Not implemented** (§7.2 and §7.3 together form the zero-split tap). The implemented tap copies
the lane's committed StateImage at a chunk boundary (≈0.25 ms D2D), so only exact taps pay a split.
Build this only if measurement shows the remaining exact-tap split matters.

The chunked GDN kernel's intra-chunk grid is relative to the call start. To make every internal
boundary an absolute multiple of 64, the Op gains a `phase = base % 64` parameter:

- The kernel treats the call as if it were left-padded by `phase` virtual tokens with
  `k = 0, v = 0, g = 0, β = 0`. For such a token, `α = exp(0) = 1` exactly and
  `delta = 0 · (v − αS·k) = 0` exactly, so `S ← 1·S + outer(0, k) = S` **bit-exactly**.
- Padded outputs are not written.
- Padding is by predicated loads, with no workspace copy.
- `phase = 0` is the current behaviour, bit for bit.

Prefill chunks remain `--prefill-chunk` tokens long. The **first** chunk after a non-aligned base
is `chunk − phase` tokens long, so every later chunk starts page-aligned. This is one boundary
change in `TextContext::prefill`. In Hybrid mode no `prefill_split_frontier_` or
`rewrite_execution_frontiers` split is requested (both remain for Legacy).

### 7.3 Tap Op contracts

**Not implemented** (the design of the zero-split tap, with §7.2).

For each prefill unit with taps `T_u = T ∩ (unit_begin, unit_end]`, `|T_u| ≤ taps_per_unit`
(default 4, bounded by free snapshot slots):

1. **GDN state tap**: `gated_delta_net(..., StateTapSet taps)`. For each tap `j`, the kernel writes
   the FP32 state after absolute position `p_j` for every value head into
   `snapshot_slot[j].ssm_state[layer]`.
   - Stores use `st.global.cs` (streaming) to avoid displacing prefill working sets in L2.
   - Oracle: the FP64 naive recurrence evaluated to `p_j` (op-development §numerical).
   - Exactness criterion: the tapped state is **bit-identical** to the final state of a separate
     call on the same inputs ending at `p_j` with the same phase. Same chunk grid, same
     arithmetic.
2. **Conv tap**: per GDN layer, copy the `kernel_width − 1 = 3` pre-conv projected columns
   `[p_j − 3, p_j)`, taken from the current unit's projection output or the previous conv history
   when `p_j − unit_begin < 3`, into the slot's conv history. Exact copy, exact oracle.
3. **Hidden tap**: the continuation hidden at column `p_j − 1`, stored for completeness. It is
   never used for a full hit (§6.1).
4. **DFlash/DFlash2 ring tap**: after the unit's context K/V materialisation, and *before* the
   ring is overwritten, assemble `slot.ring = ring(positions [p_j − S, p_j))`. Each position comes
   from the pre-unit ring if `< unit_begin`, otherwise from the unit's freshly materialised K/V.
   The materialiser writes this unit's context K/V to a unit staging buffer first; the ring commit
   and the tap gathers both read from it. Exact copy.
   - The DFlash Full (paged) pool is committed through the chunk at unit commit, so its pages
     belong to the tree like Main pages.
5. **MTP**: MTP KV is paged. Tap snapshots carry no MTP state. A resume performs the Program's
   existing MTP bridge from the resumed frontier, as for any Program-resumed frontier today.

If no snapshot slot is free when the unit starts, the tap is skipped and counted
(`taps_skipped_no_slot`). In practice D2H of a slot (≈4 ms) finishes long before the next unit
(≥0.3 s).

### 7.4 Tap publication

At the chunk boundary the tap takes a staging device snapshot slot, copies the lane's committed
StateImage into it (including the continuation hidden of column `p − 1`, which prefill keeps while
taps remain) and waits as a **pending tap** of the lane. It is published as soon as the blocks it
anchors on are committed tree nodes:

- the anchor `p/64 − 1`, which for an MTP backend (one token behind the text frontier) commits
  only after the next chunk even when `p` is page-aligned;
- for `p % 64 ≠ 0`, block `p/64` as well, whose first `p % 64` columns are copied into a
  cache-owned tail page, so every cache entry keeps an exclusive page (§6.3).

Publication creates a `Snapshot{Tap}` with `device_slot` = the staging slot and enqueues D2H
write-through of the image (and tail) into host slabs; `live_snapshots_below++` along the path.
Once write-through completes, the slot becomes an **evictable device snapshot** (§9.2). A finishing
lane publishes its remaining pending taps from its own last page, or drops them.

### 7.5 Recomputed region

When `F_s < 64k`, blocks `[F_s/64 .. k)` already exist in the tree but are recomputed, because the
recurrent state must be rebuilt. The sequence writes them into its own private pages. At block
commit (§7.6), each such block finds the existing node, so the private page stays private
(`Duplicate`) and is freed at finish. The tree is not modified. KV-write elision, which maps the
existing pages and discards those writes, is an optional optimisation (§12.4).

### 7.6 Block commit (prefill and decode)

After any commit that advances every enabled pool's committed frontier across a block end
`64(b+1)`:

- If `child(N_{b−1}, key(b))` exists with a Device copy: the sequence's page stays private
  (`Duplicate`). If it exists host-only, it adopts the sequence's page as its Device copy. Otherwise
  **insert** a node owning the sequence's page. The node stays pinned for this sequence.
- D2H write-through of the lane's Device-only nodes is enqueued in one batch when the lane releases
  its path (finish, abort or cancel), before the nodes become evictable. Batching coalesces runs of
  consecutive pages and slabs into one strided copy per plane, and pinned nodes need no backup yet.

The block-table row is unchanged, so no GPU work is needed.

Only committed tokens count. Speculative provisional lead, rejected columns and MTP/DFlash lag are
excluded by requiring every enabled pool's *committed* frontier (paged-kv §9.1).

Decode blocks become cacheable the moment they fill. A later request that shares a long generated
output, such as an agent re-sending the assistant turn, can resume from the endpoint snapshot
(§7.7).

### 7.7 Finish and cancel

At terminal settlement (Finish, or Cancel after at least one committed unit):

1. If the request is cache-enabled, every pool has committed the frontier (the backend up to its
   own frontier, `F_end − 1` for MTP) and `F_end − (deepest snapshot on path) ≥ 64`:
   - The active StateImage itself becomes the snapshot image (frozen, no copy) in a device snapshot
     slot; if none is free, one is evicted (§9.2), otherwise the endpoint is skipped.
   - The partial last page (`F_end % 64 ≠ 0`) moves to the snapshot as its tail. There is no COW,
     because the writer is terminating. The backend page keeps its own committed extent.
   - Publish `Snapshot{Endpoint}` and enqueue D2H write-through of the image and tail.
2. For every pinned node: `active_refs--`. A node that reaches 0 goes to the device LRU tail
   (§9.1), inserted in **reverse block order**, so deeper blocks are evicted before shallower
   ones.
3. Free `Duplicate` and growth pages. Release the active state slot.

`Disabled` requests (Serve warmup, `--no-prefix-reuse`) never insert nodes or snapshots.
Their pages are freed at terminal.

---

## 8. Engine and API

### 8.1 Program API

```cpp
struct SourceCandidate {                    // the binding source shared with checkpoint mode
    // ... checkpoint fields ...
    std::shared_ptr<const detail::HybridQuoteImpl> hybrid;   // set for a hybrid source
    PrefixReusePath reuse_path;             // HybridEndpoint | HybridSnapshot for a hybrid hit
};
[[nodiscard]] std::vector<SourceCandidate> hybrid_sources(const RequestBasePlan&,
                                                          std::uint32_t maximum_frontier); // §6.1-6.3
[[nodiscard]] bool hybrid_reclaim(runtime::ContextResourceUsage shortage);                // §9.5
[[nodiscard]] std::optional<std::uint32_t> hybrid_prefetch(const RequestBasePlan&);       // §6.6
[[nodiscard]] std::uint32_t hybrid_prefetch_room() const noexcept;
void set_hybrid_cost(const runtime::prefix_cache::CacheCostModel&);                       // §6.2
void set_hybrid_coalesce_wait_limit(double seconds);                                     // §12.2
[[nodiscard]] HybridCachePersistence attach_hybrid_cache_file(...);                      // §5.5
[[nodiscard]] std::optional<HybridCachePersistence> hybrid_shutdown_save() const;
[[nodiscard]] HybridPrefixCacheStats hybrid_stats() const noexcept;
void shutdown_cleanup() noexcept;           // orderly stop: write through, save, then clean up
// Shared: start_binding (dispatches on SourceCandidate::hybrid), poll_context, reserve_units,
// start_pause, advance_prefill, advance_replay, decode, commit, finish, abort.
```

A hybrid request plans no capture groups, so the Program never offers a checkpoint capture; taps are
internal to prefill. `finish` publishes the endpoint into the tree and returns no checkpoint;
`abort` publishes it too when the lane's committed state is consistent.

### 8.2 Engine

The Engine core is a template over its resource manager, chosen at construction from
`ContextCacheMode`: the original `ResourceManager`, or `HybridResourceManager`, which presents the
same surface, so scheduling, binding, incremental unit reservation, preemption and commit
orchestration are shared by both modes. Retention policy lives in the Program's index:

- `candidates` returns the Program's hybrid sources; empty while a prefilling sibling is about to
  publish the snapshot the request should resume from (§12.2), which leaves the request queued.
- `reclaim` evicts cached Device blocks (`hybrid_reclaim`); when nothing is evictable it reports
  `Blocked`, and the Engine pauses a younger request as in checkpoint mode.
- No checkpoint is published, captured, adopted or kept: `host_victims` offers nothing and the
  Program keeps no Host context arena, so a paused request recovers by Replay, from the deepest
  source the tree then holds.
- While the FIFO head waits, the Engine asks for a prefetch of its Host-only path (§6.6).
- `HybridPrefixCacheStats` feed the `throughput` record's `context_cache.hybrid` object.

### 8.3 Frontend and protocol mapping

- `PreparedPrompt` gains `block_hashes` and `tap_hints` (sorted `{position, priority}`), filled in
  both modes (cheap). Hybrid mode ignores `PreparedContextCache` session key, retention and
  opportunities, and `PromptIdentity` rewrite fields; Legacy ignores the new fields.
- `ContextCacheHints` is unchanged. In Hybrid mode:
  - explicit-evidence markers map to priority-1 tap hints, automatic protocol markers to
    `Automatic` hints (planned as structural, published as ordinary taps);
  - `session_key`, `retention`, `allow_engine_automatic_shared_prefixes` and
    `update_session_index` are ignored. Content addressing finds session chains without them.
- External protocol behaviour is identical in both modes:
  - OpenAI `prompt_cache_*` and breakpoints, and Anthropic `cache_control` validation, limits
    (four distinct breakpoints) and error cases are unchanged.
  - In Hybrid mode, breakpoints mean "tap here" (priority 1). Protocol write-policy modes
    (`mode:"explicit"`, Anthropic `cache_control`-only requests) do not suppress automatic taps:
    taps are cheap and content-deduplicated, so markers can only add reuse.
  - `cached_tokens` and `cache_read_input_tokens` report `reuse_tokens`. The streaming
    `message_start` timing is unchanged, because the quote fixes the value at admission.

---

## 9. Eviction

### 9.1 Device KV (LRU, tail-first)

- One intrusive list of `Cached` device pages whose owner has no refs and no pins: nodes with
  `active_refs == 0 && restore_pins == 0`, and snapshot tails with `restore_pins == 0`.
- On a hit, touched nodes move to the MRU end.
- On release, nodes are appended in reverse block order.
- Allocation pops from the LRU head:
  - **Host-backed** entries (host `Resident`): drop the device copy. Free.
  - **Unbacked** entries, which occur only when the host is disabled, full, or write-through is
    still filling: the node loses its last copy, so apply §9.4.

Two lists are kept, *backed* and *unbacked*, and backed entries are always consumed first.

### 9.2 Device snapshot slots

Slots hold at most `D` images. A sequence about to snapshot past its previous snapshot (the one
it resumed from, or its previous tap in the same prompt) supersedes it first (§9.3), so the slot
comes from its own lineage when it can. A tap or endpoint takes a slot in this order:

1. a free slot;
2. a slot owned by a superseded snapshot, oldest supersession first; the snapshot keeps its Host
   copy or, without one, is deleted;
3. a slot owned by a Device-only snapshot (no Host copy: the Host tier is full, disabled or
   evicted it): the one with the lowest GDSF priority (§9.3), and only when the new snapshot's
   estimated priority (no hits, and the blocks since the sequence's previous snapshot its own) is
   at least as high. The snapshot is lost, and `L` rises to its priority;
4. the least recently hit Host-backed snapshot's slot, which loses only its Device copy.

Otherwise the tap or endpoint is skipped. A Device-only owner's slot is its last copy, so value
decides there rather than recency: least-recently-hit eviction let a short request's snapshots
push out the snapshot a long conversation resumes from, and taking Host-backed slots first left
owners that Host eviction had made Device-only in their slots for good (§16.5).

### 9.3 Host (superseded snapshots first, then GDSF; dead-KV first)

**Supersession.** In a conversation the next request resumes from the newest snapshot of its
lineage, not from the one the current request resumed from. When a sequence is about to publish a
deeper snapshot on its path, the snapshot it resumed from becomes *superseded*, and so does the
previous tap it captured in the same prompt when the new snapshot is a tap, before the new snapshot
takes a Device slot or Host slabs: only a request diverging before the newer snapshot (a retry, an
edit) can still use them. The endpoint leaves the prompt's last tap retained, since a next turn
whose template re-renders the reply (stripped thinking) resumes from that tap. A hit on a superseded
snapshot retains it again, until that lineage moves on. A `Boundary` snapshot (a tap at a client
breakpoint or structural boundary, or at the divergence of coalesced requests, §12.2) may be shared
by conversations, so it is superseded only while no other conversation has continued from it: while
the tree below it is a single chain. Until then it is just its lineage's snapshot.

Allocation of `k` slabs:

1. **Dead KV sweep**: pop host-resident nodes with `live_snapshots_below == 0`,
   `active_refs == 0` and no pins, leaves first. These can never produce a hit.
2. Otherwise evict the oldest superseded snapshot entirely (both copies; `L` is unchanged), then
   repeat step 1.
3. Otherwise evict the retained snapshot with minimum `H`, then repeat step 1 (its exclusive path
   has just become dead). Update `L := H(victim)`. A victim still complete on the Device (image in
   a device slot and tail resident) only gives up its host copy and stays restorable.

A snapshot image write stops at step 3 when the victim is worth more than the new snapshot
(`H(victim) > H(new)`): GDSF never inserts what it would evict next, and the new snapshot stays
Device-only (§9.2). KV block writes (write-through at release) are not bounded this way.

```text
H(s)  = L_s + F(s) · C(s) / Z(s)                // L_s: L when s was published or last hit
C(s)  = prefill_seconds(base=F_a, tokens=F_s−F_a) − restore_seconds(image)   // clamp ≥ 0
        where a = nearest retained (not superseded) snapshot above s on its path, whatever its
        tail (or root, F_a = 0)
Z(s)  = image_bytes + tail_bytes + exclusive_path_bytes(s)
        exclusive_path_bytes: host bytes of the nodes on path(s) below which s is the only
        retained snapshot
F(s)  = 1 + hits(s)                             // hits since publication
```

- Each snapshot records its nearest snapshot above it on its path (`ancestor`) and the snapshots
  that record it (`dependents`); a new snapshot between them takes over the dependents below it,
  and a removed one hands its dependents to its ancestor. `C` and `Z` are recomputed for the
  snapshot and for those whose nearest retained ancestor or exclusive path it changes, when it is
  published, hit, superseded, retained again or removed.
- Valuing against the nearest *retained* ancestor makes a conversation's newest endpoint carry the
  whole conversation-specific prefill (its superseded predecessors go first), and a shared
  boundary its own prefix. Valuing only against aligned (`tail_len == 0`) ancestors, as before,
  priced every tailed endpoint as a full re-prefill from the root: stale endpoints of long
  conversations crowded out shared boundaries and fresh endpoints, which were evicted before the
  next turn (the production agent log of 2026-09-26: 68 turns re-prefilled 758K tokens behind a
  stuck resume point, 24 subagent turns prefilled from the root beside a cached 19.8K-token
  shared prefix).
- `C` uses the same calibrated model as §6.2, so eviction and admission agree about value.
- Aging comes from `L`, so snapshots that are no longer hit fall behind new ones without timers.
- Victim selection scans the snapshot table, O(snapshots); valuation walks at most one path per
  affected snapshot. Snapshot count is bounded by `host_cache_bytes / image_bytes` (about 1,300
  for a 230 GiB Host tier with 27B DFlash2 images), so the policy is independent of the tier size:
  a small tier reaches step 2 and 3 often, a large one rarely.
- Pinned objects (`restore_pins`, `active_refs`, in-flight fills) are never evicted.

### 9.4 Losing the last copy

A node whose device and host copies are both gone is deleted together with its subtree. Every
snapshot anchored in the subtree is deleted, and `live_snapshots_below` is decremented along the
paths. A node with `active_refs > 0` always has a device copy, so this never affects an active
sequence. The subtree walk runs only in the unbacked cases of §9.1 and §9.2, and is bounded by
the subtree size.

### 9.5 Active guarantee

- Active growth is the Engine's incremental unit reservation, shared with checkpoint mode: each
  prefill, decode or replay unit reserves its pages before it runs.
- Cached pages never count against it: a shortage first evicts from the device LRU
  (`hybrid_reclaim`), so active growth always wins over cache. Only when nothing is evictable does
  the Engine pause a younger request (Replay recovery).
- Nothing active is ever evicted.

---

## 10. Invariants

1. Each device page group and each host slab has exactly one owner state:
   Free/Active/Cached/Filling, or Free/Node/SnapshotImage/SnapshotTail/Filling.
2. A tree node is immutable: its tokens, extra key and payload never change after insertion.
3. Every node on an active sequence's mapped path has `active_refs > 0` and a device-resident
   copy until the sequence releases it.
4. A snapshot exists only while it is valid (§5.3). Match returns only valid snapshots, and only
   with `F ≤ n − 1`.
5. Exact identity: a match implies exact equality of tokens and extra keys for every block and
   tail token. Hashes never decide equality.
6. A Filling destination is never mapped into a block table, and its readers wait on its event.
   A Filling source is never freed.
7. Taps are bit-identical to an unsplit run ending at the tap, for the same phase.
   Identity-padding for phase alignment is bit-exact.
8. `Disabled` requests leave no nodes or snapshots behind.
9. The worker is the only mutator. Transfers change residency only in `poll_transfers()` or at
   commit.
10. Cost models only rank choices. Capacity and validity checks never depend on predictions.

---

## 11. RTX 5090 / sm_120a optimisation requirements

### 11.1 Streams

- Restores run on a dedicated non-blocking restore stream, ordered after the compute stream's
  queued work (destinations may have been read by earlier kernels), with an event after the
  prelude and after each layer (§6.5).
- Write-through runs on the Program's transfer stream. Every write shares that stream and every
  restore the restore stream, so each list completes in order.
- Stream priorities are not set. The design gave restores the highest and write-through the
  lowest priority; measurement has not shown the need.

### 11.2 KV page transfer

Device plane order is unchanged: one page is `planes` contiguous slices, 64 planes for 27B INT8
plus MTP. The Host slab layout packs a block's Main page with its Host page layout and the backend
page at the next 4 KiB boundary, so a transfer is a pure copy with no requantisation, and restore is
bit-exact for every KV profile.

Implemented: the copy engines. Runs of consecutive device pages whose slab records advance by one
constant pitch (within one pinned chunk and the device's maximum copy pitch) move as one
`cudaMemcpy2DAsync` per plane; a restore orders the planes by model layer.

Not implemented, kept as options if restore bandwidth or SM interference ever matters: an SM copy
kernel reading pinned slabs through mapped pointers (16 B vector accesses, a capped grid so other
lanes' decode keeps its SMs, layer-major work items), and `cudaMemcpyBatchAsync` over coalesced
runs. The design's acceptance bar was ≥ 85 % of the contiguous `cudaMemcpyAsync` peak for 2 MiB
copies in each direction; the copy-engine path reaches 24–27 GB/s on the measuring machine's
PCIe 5.0 x8 link.

### 11.3 Other rules

- A snapshot image copy is a set of Device copies over the image's parts (48 SSM + 48 conv +
  hidden + ring), about 0.25 ms for a 27B DFlash2 image.
- CUDA Graph keys, decode graphs and block-table publication are unchanged. Prefill is not
  graphed.
- All cache bookkeeping is host-side O(1) or O(path). Index arenas are reserved at startup from
  `M`, `D` and `host_cache_bytes`.

### 11.4 Cost coefficients

Admission choice (§6.2) and Host valuation (§9.3) use the Engine's context-cost model: prefill
chunk, token and attention-pair coefficients measured for the artifact's prefill signature, and the
Host-to-Device transfer coefficients of the RTX 5090 class (a local preset can replace them).
Uncalibrated terms keep the index's generic defaults. No separate calibration runs at startup.

---

## 12. Optional features (each gated by its own measurement)

1. **Layer-pipelined restore** (implemented, §6.5). The first prefill pass's layer `l` waits on
   the restore's event for layer `l` (`cudaStreamWaitEvent`), so the restore overlaps compute; the
   admission itself never waits for the copies.
   - Gate: TTFT reduction ≥ 30 ms p50 on host-restore requests of ≥ 50 k tokens.
2. **In-flight prefix coalescing** (implemented, `hybrid_await_sibling`). An admissible FIFO head
   whose prompt shares more tokens with a lane still prefilling than the cache can serve now
   reports `TemporarilyBlocked`, holding no pins, until that lane's snapshot at the divergence
   publishes; it is then admitted from it.
   - Target: the block boundary `⌊d/64⌋·64` below the divergence point `d` (at most `n-1`),
     added to the sibling's remaining plan as an exact tap. A frontier inside a block would
     publish only once the sibling completes that block. The sibling's deepest planned exact tap
     at or below `d` serves instead when it is at least that deep, or when prefilling the tokens
     between them costs no more than one chunk's fixed cost.
   - A snapshot the sibling has captured in the shared prefix but not yet published (its block or
     MTP backend page is incomplete) is also waited for. It publishes within the sibling's next
     chunk.
   - It waits only when the prefill it saves exceeds the split it may add, and when twice the
     sibling's predicted prefill up to the target (other lanes' decode rounds interleave) is within
     half the Engine's queue timeout (`pending_timeout_ms`). A waiting head stays in the FIFO.
   - Vision placeholder tokens are equal for different media, so the shared prefix ends at the
     first media item the two prompts do not carry identically (content digest, grid, timing,
     token placement). A request whose shared prefix covers a sibling's image waits for the
     snapshot past it and encodes nothing for that image (ignis ADR 0035's fan-out case).
   - Every sibling prefill boundary re-arms admission, and the head stops waiting once the sibling
     passes the target, finishes or is cancelled. So a missing snapshot never blocks it for long.
   - Gate: total prefill tokens reduced on the concurrent-shared-prefix trace, with no TTFT
     regression for other requests.
3. **Output-boundary snapshots** (not implemented). The Frontend's output parser signals structural boundaries
   (after `</think>`, before `<tool_call>`), and the Program snapshots the state at that exact
   committed frontier.
   - Without speculation: D2D of the active slot at the round boundary.
   - With speculation: a two-segment ReplaySSM Fold, `S0 → slot` over the first `j` records, then
     `slot → active` over the rest. This is bit-identical by the Fold contract. Conv and ring taps
     are handled as in §7.3.
   - Gate: reuse gain on thinking-stripped agent traces.
4. **KV-write elision** (not implemented) for the recomputed region (§7.5): map existing pages and send those KV
   writes to a scratch page.
   - Gate: measurable page savings under pressure.

---

## 13. Test specification

The test design follows. §13.1 and §13.3 are implemented (`ninfer_prefix_cache_index_test`,
`ninfer_qwen3_5_hybrid_prefix_real_test`, see Status); §13.2 applies to the unimplemented Ops;
§13.4 was the acceptance plan, and §16 records what was measured.

### 13.1 Host-only unit tests (`tests/test_prefix_cache_index.cpp`)

- **BlockTree**
  - insert/match with exact tokens and extra keys;
  - a forced constant `lookup_hash` must not produce a false hit;
  - different media digest with the same tokens must miss;
  - tail snapshot matches on exact tail tokens only;
  - `F ≤ n − 1` is enforced;
  - duplicate snapshot publication is a no-op.
- **Liveness**
  - `live_snapshots_below` stays correct under interleaved publish, evict and subtree deletion;
  - losing the last copy deletes exactly the subtree's snapshots.
- **Device LRU**
  - tail-first order after release;
  - hits move the whole path to MRU;
  - backed entries are consumed before unbacked;
  - pinned entries are never returned.
- **Host GDSF**
  - hand-computed cases: two siblings, nested ancestor/descendant, frequency;
  - dead KV is reclaimed before any snapshot;
  - the victim's exclusive path is freed;
  - `L` is monotonic.
- **Supersession and admission**
  - a lineage's previous snapshot is superseded, and a hit retains it again;
  - a `Boundary` snapshot is superseded only while no other conversation continued below it;
  - Device slots go to superseded owners, then to the least valuable Device-only owner for a new
    snapshot worth as much, then to the least recently hit Host-backed owner;
  - a Host image write displaces only snapshots worth no more than the new one.
- **HostSlabPool**
  - exhaustion, reuse and chunked allocation;
  - a slab never crosses chunks.
- **TapPlanner** (property tests over random prompts and boundaries)
  - all taps 64-aligned, in `(F_s, n−1]`, outside Vision spans;
  - deduplicated against the path;
  - priority order and budget respected;
  - the ladder stays within about log2(n/G) + 1 taps;
  - deterministic.
- **Admission arithmetic**
  - candidate fallback order;
  - Blocked vs Infeasible classification;
  - backfill proof on constructed states;
  - quote staleness via `cache_epoch`.

### 13.2 Op qualification (op-development.md contract, real 27B and 35B-A3B shapes)

- **GDN tap and phase**
  - FP64 naive-recurrence oracle at every tap, phases 0..63, T in {1, 63, 64, 65, 2048, 4096};
  - **bitwise** equality of each tap with a truncated call on the same phase;
  - `phase > 0` with zero taps matches the unpadded oracle within the existing GDN criterion;
  - `phase = 0` is bit-identical to the pre-change kernel.
- **Conv, hidden and ring taps**: exact byte comparison against a host reference gather, including
  `p − unit_begin < 3` and ring wrap (`p > S`, `unit length > S`).
- **`paged_kv_transfer`**
  - exact bytes for BF16, INT8, FP8, NVFP4 and K8V4, with Main+MTP and Main+DraftFull bundles,
    random page IDs, both directions;
  - bandwidth ≥ 85 % of the contiguous peak;
  - a concurrent-decode interference measurement is recorded.

### 13.3 Real-artifact integration (the design's scenario set)

Each scenario runs for `--kv-dtype` in {bf16, int8, fp8, nvfp4, k8v4} × backend in {none, mtp,
dflash2}. Scenarios marked "full matrix" run for every combination; the rest use int8 × {none,
dflash2}.

| scenario | assertion |
|---|---|
| resume-equivalence (full matrix) | resumed vs root: first-token logits within tolerance; greedy top-1 agreement over 256 tokens ≥ the chunk-decomposition baseline (root with a different `--prefill-chunk`); CausalScoring NLL difference within the same baseline |
| multi-turn exact endpoint | turn k+1 reuses `F_end(k)` exactly, and the tail was COW'd |
| re-rendered history (thinking stripped) | reuse ≥ `floor64(n_k − 1)` |
| edit deep message | reuse = the deepest ladder tap below the edit |
| shared preamble across 8 sessions | device pages for the preamble counted once; all 8 hit |
| concurrency 8, identical prompts arriving together | correct outputs; no double restore of a Filling node |
| host-only restore (D snapshot slots evicted) | same outputs as device resume; `Restoring` observed |
| cancellation mid-prefill | taps and completed blocks published; next identical request reuses them |
| Vision: same image vs different image, identical tokens | hit vs miss; no resume inside a span |
| pressure with active growth | active request extends by evicting cache; bounded completion unchanged |
| host cache 0 | device-only cache works; unbacked eviction deletes subtrees correctly |
| `Disabled` request | tree and snapshot counts unchanged |
| protocol usage | `cached_tokens` / `cache_read_input_tokens` equal `reuse_tokens`; streaming `message_start` exact |

Fault injection covers:

- a failed host allocation at startup, which is a clear startup error;
- a forced CUDA error in a transfer job, which gives Engine-wide failure and cleanup with no leaked
  pins (checked by `prefix_cache_stats` after restart);
- `compute-sanitizer --tool memcheck` on one concurrency-8 scenario, for the lifetime claims.

### 13.4 Performance acceptance plan

The baseline is Legacy mode, run on identical traces and flags, with `--host-cache-mib` equal in
both runs.

| metric | requirement |
|---|---|
| prompt tokens served from cache, agent trace (README workload) | ≥ baseline, target ≥ +5 points |
| admission planning time (quote + admit, host) | p99 < 1 ms (baseline: up to 250 ms allowance) |
| TTFT p50 / p95 on the agent trace | lower than baseline; report the breakdown (planning, restore, prefill) |
| cold-request prefill tok/s | within ±1 % |
| decode tok/s at c = 1, 2, 8 (with and without write-through active) | within ±1 % |
| tap overhead per 4 k-token chunk | < 0.5 % of chunk time |
| restore bandwidth | ≥ 85 % of measured PCIe peak |
| device memory | shared-preamble trace: ≥ 7× fewer preamble pages at c = 8 |


---

## 14. Original cache and configuration

### 14.1 Shared code

Both modes share the Engine core, the scheduler, the binding/pause/replay transactions and the
logical KV page and StateImage stores. The hybrid cache adds its own index and Host slab pool, the
page-prefix fork on `KVAddressSpaceStore` and copied-destination activation on `StateImageStore`.
The original cache's checkpoints, prefix index, capture transactions and Host context arena are not
used in Hybrid mode.

### 14.2 Configuration surface

```cpp
enum class ContextCacheMode : std::uint8_t { Original, Hybrid };

struct HybridPrefixCacheOptions {                           // used only when mode == Hybrid
    std::optional<std::uint32_t> device_snapshot_slots;     // --device-snapshot-slots
    std::optional<std::uint32_t> max_new_taps;              // --cache-taps-per-request
    std::optional<std::uint32_t> tap_ladder_tokens;         // --cache-tap-ladder (G)
    std::optional<std::uint32_t> tap_min_gap_tokens;        // --cache-tap-min-gap
    std::filesystem::path persistent_file;                  // --prefix-cache-file (§5.5)
    std::string persistent_identity;                        // set by the product binary
    PrefixCacheSaveControl persistent_save;                 // abandons the save (§5.5)
};

struct ContextCacheOptions {
    bool enabled;
    ContextCacheMode mode = ContextCacheMode::Original;   // ninfer-serve selects Hybrid by default
    HybridPrefixCacheOptions hybrid;
    std::optional<std::uint32_t> device_state_slots;      // Original only (--device-state-slots)
    std::optional<std::size_t> host_capacity_bytes;       // --host-context-mib: Hybrid slab pool
};
```

Engine construction (`normalize_engine_options`) resolves every unset Hybrid value, and
`Engine::options()` reports the effective ones. With `C = max_concurrency`, `chunk =
prefill_chunk` and a Host tier present when the budget is nonzero:

| value | default | reason |
|---|---|---|
| `host_capacity_bytes` | 8 GiB (`kDefaultHybridHostCacheBytes`); 0 disables the Host tier | the original cache's Host KV default |
| `device_snapshot_slots` | `C + 1` with a Host tier, `C + 2` without | one resident snapshot per live conversation, plus staging; without Host the slots are the only snapshot storage |
| `max_new_taps` | 8 with a Host tier, 2 without | without Host, taps would evict other conversations' snapshots |
| `tap_ladder_tokens` | `max(4096, 2·chunk)` | ladder taps land on chunk boundaries; a finer ladder only duplicates them |
| `tap_min_gap_tokens` | `max(1024, chunk)` | the same |
| KV capacity (`ninfer-serve`) | `--kv-capacity auto` unless given | free VRAM becomes Device block cache |

`device_state_slots` is rejected in Hybrid mode: the extra Device StateImages are the snapshot
slots.

---

## 15. Risks and mitigations

| risk | mitigation |
|---|---|
| Exact taps cost one prefill split each | Only semantic boundaries are exact (explicit, generation opener, structural); the prompt tail and ladder are flexible and cost no split. The proximity rule leaves one split per chat request, and none after an endpoint resume. The zero-split GDN tap (§7.2–§7.3) removes the rest if measurement shows it matters. |
| Snapshots crowd KV out of host memory | GDSF weighs bytes against recompute seconds on the same scale for both. Dead-KV sweep runs first. |
| Write-through competes with decode | Copy engines only, batched when a lane releases its blocks; decode rounds/s were unchanged with prefetch and write-through active (§16.3). Write-through can be throttled when decode is active without changing correctness. |
| Unbacked device eviction deletes subtrees | Only happens with no or full host cache. Backed entries are always preferred. |
| Large pinned allocations on Windows | Chunked allocation; the startup ledger reports the resolved size. |
| Hybrid mode ignores session keys | Content addressing reproduces chain reuse; the protocol surface is unchanged in both modes. |

---

## 16. Decision record

Every measurement here ran on one RTX 5090 (Windows, CUDA 13.4) with Qwen3.8-27B NVFP4. The
Host-to-Device link of the measuring machine runs at PCIe 5.0 x8 (the GPU supports x16), so
restores and prefetches move about 25 GB/s. A change that did not improve its target metric, or
that regressed another, was reverted and is recorded here with the reason.

### 16.1 Hybrid against Legacy

`ninfer-serve --max-context 32768 --max-concurrency 2 --kv-dtype int8 --kv-capacity auto
--prefill-chunk 2048 --host-cache-mib 12000`, plus `--use-original-prefix-caching` for Legacy, a
fresh server per workload:

| workload | Legacy | Hybrid |
|---|---|---|
| 4 conversations × 6 turns, thinking, shared ~5.9K-token system prompt: prompt tokens from cache | 83.0% | 90.9% |
| same: TTFT p50 / p90 / mean | 0.177 / 0.962 / 0.331 s | 0.149 / 0.283 / 0.200 s |
| same: TTFT median of later turns | 0.173 s | 0.129 s |
| 8 distinct ~11.4K-token prompts overflowing the Device pool, revisited: tokens reused on revisit | 0 of 8 | ~11,400 of each |
| cold ~8.6K-token prompts: server prefill / decode tok/s (median of 6) | 10.1k / 84.9 | 10.2k / 84.6 |
| 64-token prompt, 2048 output tokens: decode tok/s | 86.3 | 86.0–86.1 (86.1–86.3 with `--kv-capacity 65536`) |

With `--spec dflash2 --draft-tokens 7` and `--spec mtp --draft-tokens 3` the conversation workload
served 90.8% and 90.9% of prompt tokens from cache with no errors.

### 16.2 Decisions while building

1. **A second mode, not a replacement.** The owner asked for both systems to stay available, so
   Hybrid reuses the active-execution machinery and replaces only admission-source selection,
   retention, capture and pressure (coexistence rules above). `ninfer-serve` defaults to Hybrid;
   the Engine option default stays Legacy.
2. **Snapshots copy the committed state at chunk boundaries; the zero-split GDN tap was not
   built.** A tap copies the lane's StateImage into a snapshot slot (about 0.25 ms D2D), so only
   exact taps split a prefill chunk, at about 15 ms per split on 27B. §7.2–§7.3 remain the design
   for removing that split if measurement shows it matters.
3. **Hybrid ignores the chat template's rewrite frontiers.** Legacy splits prefill after each
   assistant header, after `<think>` and after the reasoning close for its rewrite checkpoints.
   Hybrid captures nothing there, so a short chat turn prefills in two passes (to the opener, then
   the rest) instead of four (§7.1).
4. **No opener tap after an endpoint resume.** On a tool-calling agent trace (27B NVFP4, INT8 KV,
   `--preserve-thinking`) the opener split cost about 15 ms of a ~70–130 ms turn TTFT. A request
   that resumed from an endpoint proves its client echoes generated turns token for token, so its
   own endpoint serves the next turn (§7.1).
5. **Free VRAM becomes Device block cache.** Removing the `C·L` clamp and defaulting
   `--kv-capacity` to `auto` makes the pool several times larger. At the Legacy pool size decode is
   within noise; the automatic pool costs about 0.2–0.35% decode throughput, attributed by the
   capped runs to KV placement across the larger pool, not to per-round cache work (§16.1).
6. **One pinned Host slab pool, pinned in chunks.** KV blocks and snapshots share one pool with no
   fixed split (§5.4). A single very large pinned allocation can fail or stall under WDDM, so the
   pool is pinned in chunks of at most 4 GiB; strided copy runs never cross chunks, after a 52 GB
   Host tier produced an invalid `cudaMemcpy2DAsync` that aborted the server.
7. **In-flight prefix coalescing** (§12.2). With `--max-concurrency 4`, four requests arriving
   together with a new ~13.9K-token system prompt served 74.9% of prompt tokens from cache (the
   three followers reuse 13,888 tokens each) instead of 11.0%. Mean TTFT fell from 3.52 s to
   1.48 s and the burst's wall time from 5.56 s to 1.80 s.
8. **Restores never hold the admission, and the first prefill pass waits layer by layer** (§6.5).
   Copies run at 24–27 GB/s. Revisits of 8 ~11.4K-token prompts under two clients, each restoring
   its 147 MiB state image, have a median TTFT of 45.7 ms, against 60.0 ms when the admission
   waited for the copy. Revisits of two alternating ~90K-token conversations, each restoring
   2.6–2.8 GiB, have a TTFT of 268 ms with a short question (was 301 ms) and 1186 ms with a
   3.4K-token tool output (was 1208 ms). An Nsight Systems trace of a 44K-token revisit (1.49 GiB
   restored in 60 ms) shows the pipelining at work: the first prefill pass runs its
   linear-attention layers at once and waits ~2.2 ms at each attention layer for its KV, finishing
   as the copy does. The rest of that TTFT was not the restore: the 45 new tokens ran as four
   passes over the model, split at exact snapshot and template boundaries (~12–15 ms each), which
   decision 3 removed, and the prompt-attention kernel took ~2.3 ms per attention layer over the
   44K context in the one pass that used it.
9. **Short prefill passes over long contexts use split-KV attention.** Single-row passes of 17–64
   new tokens were then routed to the chunked split-KV attention once the context held 64 visible
   keys per new token (80 for 16 query heads), 4–12× faster than the prompt kernels at 16K–180K
   keys (32 new tokens at 180K keys: 1.06 ms per layer instead of 9.5 ms). With ~90K cached tokens,
   one client:

   | revisit | before | after |
   |---|---|---|
   | Device-resident, short question | 204 ms | 91 ms |
   | Device-resident, 3.4K-token tool output | 1101 ms | 1050 ms |
   | Host restore (2.6–2.8 GiB), short question | 268 ms | 170 ms |
   | Host restore, 3.4K-token tool output | 1186 ms | 1067 ms |

   Under two clients, ~11.4K-token Host revisits went from a 45.7 ms to a 30.1 ms median TTFT. Cold
   prefill was unchanged (10.1k vs 10.2k tok/s). Causal attention has since been reorganized per
   KV format: its grouped split-KV routes take single-row passes of up to 256 new tokens (BF16,
   INT8), 192 (NVFP4) or 80 (FP8, K8V4) directly, so no cache-specific routing remains. The
   numbers above were measured on the earlier kernels.

### 16.3 Changes from a production agent log (2026-09-26)

A production request log (889 requests over 70 min, 27B NVIDIA NVFP4 + DFlash2, INT8 KV, C=2,
`--host-cache-mib 52000`, 247K-token Device pool) of an agent harness with about ten concurrent
conversations (17-tool subagents of 30K–208K tokens and a 20-tool orchestrator) served 94.9 % of
prompt tokens from cache but prefilled 5.02M tokens (994 s). It showed three problems:

1. **Host eviction kept the wrong snapshots.** The Host tier was full from minute 4. The GDSF value
   of a snapshot counted only aligned ancestors, so tailed endpoints were priced as full
   re-prefills from the root: about 100 stale chain endpoints (~19 GB) stayed, while shared
   system/tools taps and fresh endpoints went first. 68 turns re-prefilled 758K tokens because
   their resume point was stuck at an old aligned tap while the suffix grew to 42K tokens, and 24
   subagent turns prefilled from the root next to a cached 19.8K-token shared prefix.
2. **Only one lane ran for half the wall time.** Two 150K–200K contexts do not fit in the Device
   pool together, and Hybrid mode never issued a backfill proof: the average decode batch was
   1.27 of 2.
3. **Every turn restored its whole context from the Host tier**: 2.59 TB in total, 2.7 GB per turn
   (median), while the FIFO head had usually been waiting seconds to minutes.

| change | section | A/B result | status |
|---|---|---|---|
| Superseded snapshots evicted first; valuation against the nearest retained ancestor | §9.3 | prompt tokens prefilled −10.8 % (seed 42) and −3.4 % (seed 43); main-session turns that re-prefilled the whole prompt 3 → 1 (seed 42); TTFT without queue wait −6 % and −3 % | kept |
| Persistent backfill proof with a 16K-token growth reserve | §6.3 | targeted scenario: four 12K-token requests queued behind a blocked 90K-token head finish in 1.6 s instead of 35–36 s; the long request already decoding takes 4.7 s (12 %) longer, about the borrowers' prefill time, and the head starts that much later (its answer ends at the same time); no answer is cut short. Agentic workload: 0–9 backfills per run, effect within sampling noise | reverted: a trade-off, not a win (below) |
| Prefetch of the blocked head's Host-only blocks | §6.6 | admission Host restores −22 % (seed 42, 62.4 → 48.6 GB) and −25 % (seed 43, 68.4 → 51.3 GB), 10–11 GB prefetched per run; TTFT without queue wait −3 % and −6 %; decode rounds/s unchanged (57.8 vs 57.8, one request decoding) | kept |

The prefetch was measured on top of the backfill proof. The build that combined all three against
the base: at the 16 GB Host tier, prompt tokens prefilled −13.9 % (seed 42) and −2.7 % (seed 43),
TTFT without queue wait 0.82 → 0.74 s and 0.75 → 0.73 s; at 52 GB (seed 42), prompt tokens
prefilled −9.2 % (623,853 → 566,189), served from cache 89.1 → 90.1 %, no turn prefilled from the
root beside cached blocks (3 before), TTFT without queue wait 0.68 → 0.60 s (−12 %), workload wall
time −7.8 % for the same output volume (130K tokens). That 52 GB arm's mean TTFT including queue
wait rose from 2.1 to 2.8 s: in a closed loop, turns that finish sooner re-queue sooner, and the
queue-wait increase was spread over requests that no change touches. Backfill triggered at most
twice in it. The two kept changes together (without the backfill proof) ran the pressure workload at
scale 0.3 (54 requests, no failures, 1.2 GB prefetched); that combination was not A/B tested on its
own.

**Why the backfill proof was reverted.** Backfill lets a shorter request start beside a long one
while a longer FIFO head waits for pages. The GPU time the borrower uses comes out of the long
requests: the one decoding slows by about the borrower's prefill time, and the blocked head starts
that much later. Short requests gain a lot, long ones lose a little, and aggregate throughput hardly
moves. When long main-agent requests dominate, the proof rarely passes (two long contexts never
fit, and the growth reserve leaves room only for small borrowers; in the production log about
225 of the 2,122 s spent with one lane decoding and requests waiting), and when it passes the
latency cost falls on those long requests. Hybrid mode therefore still issues no backfill proof.
A version worth revisiting has to remove the trade-off rather than move it; see §6.3.

Measurement: `bench/agentic_ab` (3 interleaved sessions and 11 subagents, 130 requests; seeds 42
and 43) on the production flags with the Device pool and Host tier scaled down to that workload
(`--kv-capacity 160000`, `--max-context 160000`, `--host-cache-mib 16000`; `--vram-headroom-mib`
dropped, since it requires `--kv-capacity auto`). Each change is compared with the build before it
at a 16 GB Host tier, where eviction matters, and all three with the base at 52 GB, where it rarely
does. The policy does not depend on the tier size (`--host-cache-mib` 0 to 230 GiB): snapshot count
is bounded by `host_cache_bytes / image_bytes` (about 1,300 at 230 GiB), and victim selection is a
scan over snapshots.

Run notes:

- Seed 42, first run: the superseded-first arm and the second half of the prefetch arm ran under
  outside CPU load; tokenization, which none of these changes touch, was up to 2.3× slower, and
  their decode host time per round (204 and 338 µs against 80 µs) was not the builds'. Both arms
  were rerun with per-process CPU logging and the rerun values are the ones above; their
  tokenization matched the base.
- Backfill needs a blocked FIFO head with a shorter request behind it while a long answer decodes.
  The workload's lock-step sessions rarely queue that way (one lane decoding with requests waiting
  for 3–13 % of a run, against 50 % in the production log), so a targeted scenario measures it: a
  100K-token request answering with a count to 2,400, then a 90K-token request, then four 12K-token
  requests, greedy, `max_tokens` 64000, two repetitions per build, identical to within a second.
  In seed 43 the two long requests that borrowers overtook waited 6 and 8 s longer (23 s for one in
  the prefetch arm) while the borrowers saved about 1 s each; output lengths differ between arms,
  so this is not attributable, but it is worth watching in production.

### 16.4 Fixes from operation (2026-09-27)

None of these were A/B tested; each is a correctness, behavior or log fix.

| change | section | reason and evidence |
|---|---|---|
| A Host tier smaller than the saved file restores its most valuable snapshots and only their paths (file format 3, tables before slab bytes), with a startup warning | §5.5 | Format 2 restored every block before any snapshot, in file order, and still reported the file restored. For the production file (15,807 blocks and 98 snapshots of 91–92 slabs of 2.06 MiB, 49.97 GiB, saved at `--host-cache-mib 52000`), any tier below about 32,800 MiB filled with blocks and kept no snapshot, so nothing was resumable; between that and about 51,200 MiB the snapshots kept depended on file order |
| A Host tier that keeps none of the file's snapshots warns `prefix cache not restored` instead of `partly restored … the most valuable snapshots were kept` | §5.5 | A restore keeps only the blocks its snapshots resume through, so keeping no snapshot restores nothing, yet the line claimed the most valuable snapshots were kept; it now names the tier the file needs and that the save at shutdown replaces the file |
| A Host restore's per-layer events are looked up immediately before the first prefill chunk, not before the MTP bridge | §6.5 | Reported by funguf (Wallawalla47/ninfer-custom pull request 2) with an AddressSanitizer trace and 9 production crashes in `cuStreamWaitEvent` in one day (`--spec mtp`, Host tier): the view was taken before the MTP bridge, whose KV commit polls the cache, and a batch that had landed meanwhile was freed under it, so the first chunk waited on whatever the freed memory held. The report's fix copied the handles into the prefill context; looking them up at the chunk instead never hands CUDA an event the cache has recycled, needs no allocation, and skips the waits once the copies are complete. DFlash2 and no-speculation lanes run no bridge and were not exposed |
| The chunk function takes the restore's layer events itself (`PrefillContext::take_layer_ready`), so program code never holds the view | §6.5 | Hardening. A deterministic test of the fix above was asked for, but the retired-batch path is timing-dependent and a dangling view reading a recycled handle can pass a token-equality check, so a test would need a product seam or AddressSanitizer. Moving the lookup inside the chunk call instead leaves no place for program code to take the view early |
| A stop fails running and queued requests and answers them before the save (`Engine::stop()`; `ninfer-serve` stops on Ctrl+C pressed twice within 5 s) | §5.5 | The production log's stops at 07:48 and 07:54 left 10 and 3 requests unfinished and wrote no file: Ctrl+C waited silently for them and a second Ctrl+C killed the process. Console test, one streaming and one queued request: before, generation ran on 68.6 s after Ctrl+C; after, one press only prompts, and a confirmed pair fails both with 503 within 0.2 s, saves 99 blocks (503 MiB) in 0.1–0.2 s and exits about 0.7 s later. Answering them only after the cleanup that saves made each 503 wait for the whole save (408 blocks and 4 snapshots, 1,438 MiB: both 503s 392 ms after the stop, as the 0.3 s save ended); they are now answered first (8 ms after the stop, 447 ms before the save ended), and the `persist` real test checks that the file is not yet saved when the running generation is answered |
| One Ctrl+C during the stop exits without saving and deletes the unfinished file (`PrefixCacheSaveControl`); the line reads `Press Ctrl+C again to exit without saving` | §5.5 | Leaving during the save needed another confirmed pair of presses, and `_Exit` left a partial `.tmp` of up to the Host tier's size beside the previous file until the next save |
| A cancellation, and a failed admission's rollback, wait for the Device before the lane's pages, state image and execution row return to the pools or move into the index (2026-10-01, upstream `75a89050`) | §7.7 | A non-final prefill step returns without waiting, so a cancel can arrive with the lane's next chunk still queued. The block-table shadow already waited for its own queued copies; the wait now covers everything the lane releases. Not reproduced as a failure. The `cancel-prefill` and `cancel-prefill-dflash2` real scenarios cancel after the first reported chunk (the cancel landed at 1,024 of 3,172 prompt tokens): the lane's next request matches a run without the cancellation, and a request extending the cancelled prompt resumes from its published prefix and matches an uncached run |

### 16.5 Changes from upstream's TTFT campaign (2026-09-30)

Upstream's public-HTTP TTFT campaign (`tools/bench/ttft`, its cache profiles translated to this
cache's options at equal Device state slots and pinned Host bytes) ran against the upstream port of
this cache in WSL2 on the same RTX 5090. Five of its cases resumed a conversation from the root, or
far behind its newest snapshot, where upstream's checkpoint-catalog cache resumed from the conversation's own
checkpoint. Measured with one sample per case on the upstream port; the eviction code is the same
in both trees.

| change | section | result | status |
|---|---|---|---|
| A sequence supersedes the snapshot it resumed from before its new snapshot takes a slot | §9.2, §9.3 | `session-alternating` (2 Device snapshot slots, no Host tier; two sessions sharing a 4K prefix): A2 and B2 TTFT 305 → 22 ms (upstream 49 ms). Before, each session's endpoint took the slot of the other session's endpoint, not of the shared tap it had resumed from | kept |
| Device-only slots evicted by GDSF priority, only for a new snapshot worth as much, and before a Host-backed slot gives up its Device copy | §9.2 | `resume-after-interference-device` (same profile; two short requests between a 7.7K conversation's turns): resume 600 → 22 ms (upstream 78 ms); `resume-after-interference-catalog` 600 → 22 ms (632 ms) | kept |
| A snapshot image goes to the Host only by displacing snapshots worth no more; protocol-automatic markers published as ordinary taps rather than `Boundary` snapshots | §7.1, §8.3, §9.3 | `private-state-working-set-shift` (2 Device slots, 384 MiB Host, DFlash2; four 2K conversations in two phases): the second phase's continuations 204–215 → 40–49 ms (upstream 48 ms). New images had displaced the snapshots the waiting conversations resume from | kept |
| A `Boundary` snapshot is superseded like any other while no other conversation has continued from it | §9.3 | same case: B1 207 → 39 ms. Each conversation kept its system-block boundary beside its newer snapshot, so the other conversation's snapshot lost its slot. All eight continuations now take 20–40 ms (upstream 48–56 ms) | kept |
| A deeper tap supersedes the sequence's previous tap of the same prompt (the endpoint does not supersede the prompt's last tap) | §9.3 | `session-alternating-64k-host-swap` (4.5 GiB Host for two 64K sessions whose KV takes 4 GiB): A2 2381 → 176 ms, B2 164 → 113 ms (upstream 242 and 171 ms). Valued against the tap below it, each ladder tap was worth more per byte than the prompt's last tap, so GDSF kept the four ladder taps and evicted the last tap until A resumed from 48K | kept |
| Flexible taps (prompt tail, ladder) as a separate snapshot kind, evicted before every other retained snapshot | §9.3 | broke a regenerate-after-edit resume: with DFlash2 and a 1 GiB Host tier, the prompt-tail tap a request diverging late in the previous prompt needed went first (`restore-exact-dflash2` real test) | reverted |
| The same kind starting from half a hit in its GDSF value | §9.3 | no change on the two cases it targeted | reverted |
| Host admission by value (`F·C/Z`) instead of priority `H` | §9.3 | meant for small tiers, where `L` rises by a whole snapshot's value per eviction and almost any new image outranks the rest: stale snapshots worth more per byte then kept the tier from new ones, and the index test's Host backup found no slabs | reverted |

The full campaign after these changes (3 samples per case) gives a geometric-mean TTFT ratio of
0.658 against upstream's cache over 259 (case, role) observations: 172 faster by more than 10%, 18
slower. The slower roles are not eviction. The 55K rotation (six sessions resumed round-robin with
Device KV for about four) restores every session from Host, about 101 ms each, because Device KV is
LRU over a cyclic working set larger than the Device; upstream's cache keeps three sessions resident
(72 ms) and restores the other three in 125–245 ms, so its per-round mean is higher (128 against 99
ms) while three roles are faster. A repeat of an identical prompt prefills its last few tokens
(about 20 ms), where upstream's cache samples from the hidden state stored with its checkpoint (6
ms): the prompt's last tap sits before the generation opener, and a snapshot at the prompt's exact
end would cost a whole state image per request for byte-identical repeats only. The other slower
roles are arrival races between concurrent requests and timing effects in profiles that run with the
cache off.

### 16.6 Port onto upstream's continuation/checkpoint engine (2026-10-05)

Upstream (`abb7f14f8`) replaced its context cache, Engine core and Program storage: incremental unit
reservation with resource-pressure preemption and Snapshot/Replay recovery, a binding transaction
for every lane start, logical KV page and StateImage stores with replicas, and a single
`--host-context-mib` Host budget. The hybrid cache was ported onto it rather than kept on the
superseded storage:

- Admission is a binding source (`SourceCandidate::hybrid`) bound by the shared `start_binding` /
  `poll_context` transaction (§6.4), so preemption, replay and the scheduler work unchanged in both
  modes. The old private materialization transaction and the lease-entitlement admission are gone:
  the first unit's growth is reserved, later units reserve incrementally, and a shortage evicts
  cached blocks before the Engine pauses anything (§9.5).
- `KVAddressSpaceStore` regained the page-prefix fork and `StateImageStore` the copied-destination
  activation; everything else (logical pages, transfer destinations, segmented StateImage and
  record-addressed KV Host copies) maps onto the new stores directly.
- In Hybrid mode the Host budget is the slab pool and the Program keeps no Host context arena, so a
  paused request recovers by Replay; the tree usually still holds its prefix.
- `--use-original-prefix-caching` now selects upstream's new continuation/checkpoint cache; the
  old catalog cache and its flags (`--host-cache-mib`, `--host-state-slots`, `--host-kv-mib`,
  catalogs, long anchors) were removed with it. `--host-context-mib` replaces `--host-cache-mib`.
- The per-request `materialization` admission diagnostics were dropped with the old transaction;
  the `throughput` record keeps the hybrid gauges and counters.

---

## 17. Qwen4Exp binding (Qwen3.8-Flash-Next)

`Qwen4ExpForCausalLM` keeps every routed expert in pinned Host RAM and streams them into a Device
expert cache, so VRAM and pinned RAM are budgeted against the experts. Its prefix cache reuses this
design's index (`runtime/prefix_cache`: tree, snapshots, GDSF, superseding, `plan_taps`) and the
frontend's keys and hints (`block_hashes`, `block_extras`, `tap_hints`) unchanged; the physical
binding and its orchestration are its own (`src/models/qwen4_exp/program/prefix/`). The model-level
rules, measurements and the build record are in the
[Flash-Next design](qwen3_8-flash-next-design.md) §19.3.1; this section states the contract.

| area | Qwen4Exp |
|---|---|
| Configuration | Hybrid only: `normalize_engine_options` resolves the Host tier per architecture (default 4 GiB, `kDefaultQwen4ExpHybridHostCacheBytes`; 0 is refused), no Device snapshot slots (`--device-snapshot-slots` is refused), the common tap budget (8, ladder max(4096, 2 x chunk), gap max(1024, chunk)). The original prefix cache (`ContextCacheMode::Original` enabled) is refused: a Qwen4Exp Engine runs with the hybrid cache or with the context cache disabled |
| Snapshots | Host-born only: a capture reserves Host slabs and copies the lane's state image (GDN recurrent and conv state, PLE state, QSA tails, the MTP drafter's saved residual) and its partial page, one decoder layer at a time on the transfer stream; the next call of the lane waits per layer. Restores are one batch per admission on the restore stream, waited on per layer by the first call |
| MTP drafter | Cells and anchors follow rules MR1-MR6 (design §19.3.1): blocks record `mtp_next`, snapshots `mtp_written`/`mtp_next`; a resume whose continuation differs rewrites the anchor's last cell in a private copy |
| Admission | Two sources per request: the affordable snapshot chosen without a lane, then the root. `start_binding` re-selects for the actual lane (crediting a lane-resident snapshot) and returns no reservation when the choice changed, so the Engine falls back to the root; otherwise it reserves the request's whole KV extent, so every execution unit fits and a Qwen4Exp request is never paused. A prefetch landing over the prompt's path is waited for before matching, and pending Host writes are drained before a cached source is given up (as §6.6). A blocked FIFO head's Host-only blocks are prefetched into free and Host-backed Device pages, never growing the pool into the expert frames (`RuntimeStats::hybrid_prefetched_blocks`). A fresh request whose prompt shares a prefix (media agreeing, as §12.2) with a lane still prefilling waits for that lane's snapshot at the divergence when the predicted wait is within half the queue timeout: an exact tap is planned there (splitting the lane's pending call) and published as a Boundary. Shortages are reclaimed by evicting unpinned cached Device blocks |
| Taps | `plan_taps` with Vision exclusions, then the call planner (`prefix/call_plan.h`): boundary exact taps (Explicit, Structural) always split; the generation opener and Automatic taps split only when the split adds at most 0.1 s of call cost (a CPU-served opener tail costs ~35 ms), else become flexible. Taps planned as boundaries (Explicit, Structural, coalescing) are published as `SnapshotKind::Boundary`. The opener is kept after endpoint resumes; a resume from an opener snapshot whose capturing request had resumed from an endpoint counts `endpoint_mismatch_fallbacks` |
| Prefill calls | The plan's calls from the resume frontier: without exact taps `F, F + chunk, ...` (for F = 0 the cold grid). Calls of at most 8 columns are CPU-served and promote experts at the decode rate |
| Vision | A resume frontier strictly inside an item is never chosen; items ending at or before the frontier are not encoded again (the encode window is planned from the frontier); suffix M-RoPE positions come from the prompt |
| Statistics | `RuntimeStats::hybrid_*` and the request log's `cached_prefix_tokens` / `restored_host_bytes` as in Qwen3.5 |
| Persistence | As §5.5, own format (`prefix/persist.cpp`, `NINFQ4PC` version 1): the block table also stores each block's `mtp_next`; snapshot meta travels inside each image's header slab and is validated by the state-image layout fingerprint, which the file's geometry also records. Saved at the Engine's stop after every lane is released; loaded into the empty cache at startup |
| Not built | Device snapshot slots, preemption (pause and Snapshot/Replay recovery: the whole extent is reserved at admission instead), prefill CPU assist and C > 1 extras (design P7, P8) |
