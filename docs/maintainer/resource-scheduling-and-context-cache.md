# Infernix Resource Scheduling and Context Cache

This document explains how Generation coordinates request execution, prefix reuse and preemption
recovery within limited device memory and Host RAM, and it is the design authority for these
policies. The prefix cache itself (block index, snapshots, taps, eviction, the Host slab pool) is
specified in the [Hybrid prefix cache](hybrid-prefix-cache-spec.md); the global execution and
publication relationships are in [Engine architecture](engine-architecture.md); the physical KV
layout, page tables and consumer constraints are in [Paged KV](paged-kv-cache.md).

## 1. Core idea

Infernix runs on a single GPU with a single resident model and 1–8 execution slots fixed at startup.
For a Hybrid model to reuse a prefix, it needs the complete recurrent state at that position and
contiguous KV coverage. KV can be shared page by page, while recurrent state is a complete image of
one position. The prefix cache therefore keeps content-addressed 64-token KV blocks in one radix
tree and sparse StateImage snapshots at chosen positions of it, on the Device and in a pinned Host
tier; an admitted request starts from the deepest snapshot its prompt reaches.

Ordinary execution reserves resources incrementally for the next unit. When the working set grows
beyond what can be resident together, cached blocks are evicted first; then younger requests are
paused and older requests advance first. A paused request gives up its lane and its device state
and later recovers by Replay, from the deepest state the cache then holds. A request's committed
history and output semantics are preserved across a pause; the device binding is rebuilt.

<a id="ownership"></a>
## 2. Ownership and call relationships

| Part | Facts and decisions it owns |
|---|---|
| Frontend | Template rendering, token/position/media identity, block hashes, tap hints at exact semantic boundaries |
| Engine request record | Original ticket, input and output objects, budget, cancellation, first-time stamps, pause and recovery state |
| Scheduler | Execution membership, fresh scans and bounded bypass, prefill/replay order (least remaining, aged), preemption and resume gating |
| `HybridResourceManager` | The Engine's connection to the Program's prefix index: admission candidates, reclaim by eviction, blocked-head prefetch, queue holds, cache statistics |
| Native Program | Model ledger and its execution splits, backend frontiers, work-unit demand, state commit, the prefix index and its retention policy |
| Stores inside the Program | Actual occupancy of State and KV, references, source pins, reservations and transfer destinations |

```mermaid
flowchart TD
    F[Frontend: exact input, block hashes, tap hints] --> Q[Request record]
    Q --> S[Scheduler: this round's work set]
    Q --> R[HybridResourceManager: candidates, reclaim, holds]
    R --> N[Program: prefix index, bind, execute and commit]
    S --> N
    N --> P[Stores: actual allocations and references]
    N --> E[Engine: output and lifecycle publication]
```

The runtime queries the Program for actual occupancy and releasable amounts. A cached block or
snapshot can be shared by many requests, and evicting one entry may release zero bytes. A source pin
protects a page while a fork or copy reads it.

One Engine control thread advances policy, transactions and reference settlement. A Program owns its
own mutable state, workspace, context stores and CUDA Graphs and shares no mutable allocation with
other Programs.

<a id="capacity"></a>
## 3. Capacity and execution permits

### 3.1 Startup capacity

The Device prepares Main KV, the selected backend KV, StateImage, workspace and Graph space according to
the model's actual layout. A Main KV page is 64 tokens; the physical byte size of each KV kind is
determined by its layer count, geometry and storage format.

| Setting | Current meaning |
|---|---|
| `max_context` | The logical context limit of one request |
| `kv_capacity` | Token-equivalent capacity of the Main KV physical pool, rounded to pages; can also be solved automatically from the startup memory budget, which then turns free VRAM into Device block cache |
| `max_concurrency` | The limit C on simultaneously held execution bindings, range 1–8 |
| `context_cache.hybrid.device_snapshot_slots` | Device StateImages beyond the C lane images, holding cached snapshots and tap staging |
| `context_cache.host_capacity_bytes` | The pinned Host slab pool shared by cached KV blocks and snapshot images |
| `context_cache.ssd_capacity_bytes` | The opt-in SSD tier below it (0: none), a scratch file of Host-slab-sized records |

The page lower bound of the Main KV capacity is `max(ceil(max_context / 64), C)`, which satisfies one
request exclusively using the maximum context and the geometric requirement of C minimum pages.
Without the prefix cache the upper bound is `C × ceil(max_context / 64)`; with it, pages beyond the
active leases are cache-only capacity. Native also computes the layout of the selected backends,
state, workspace and Graphs; explicit or automatic capacity must fit the available device memory.
Capacity is not divided evenly among lanes. The defaults of the cache settings are resolved at
Engine construction ([spec §14](hybrid-prefix-cache-spec.md#14-configuration)); with the cache
disabled both are 0.

Required request input, the token ledger, prepared media, Responses storage and the Frontend media
cache each have their own lifecycle and existing count, length or media budget limits. The Host slab
pool does not cover this memory and does not limit total process RAM.

### 3.2 Execution and resume permits

The Program computes the typed KV coverage of Prefill, Replay, Decode/Verify, Control or Normalize from
the actual binding state, including the speculative peak and backend needs. State and the private
copy of a cached partial tail page are acquired at bind time.
The requests to multiple typed pools for one unit succeed together; on failure the partial
reservation of this attempt is returned and the specific shortfall is reported.

The Engine acquires resident unit permits with the service turn's owner first (§5.2), then in original
ticket order, forming the runnable subset; one row temporarily lacking resources does not stop other
permitted rows from executing. Fresh admission uses the remaining headroom and cannot spend existing
permits. Ordinary unit
settlement releases unused reservations, and later growth keeps requesting on demand. Operators do not
select cache victims internally.

A pause resume acquires, at once, coverage for the complete State and "rebuild to the old frontier +
the first real new unit". Native holds the real, not-yet-materialized reservation across Replay
chunks; reaching the old frontier or completing the bridge does not lift the protection. The resume
permit ends only when a new prefill commits, generation/control commits a new token, or the request
enters a terminal state.

One request can temporarily hold most of the KV. If, after evicting cached blocks and pausing other
requests, the only request still cannot acquire a legal unit, the Engine reports a capacity contract
error instead of waiting for a releaser that does not exist. The error names the request, the unit
kind and the KV pages still short. No part of the unit has run, so it is a
`RecoverableExecutionError` (engine architecture, failure handling): the residents and paused
requests fail with it and the Engine keeps serving, unless a binding or materialization is in flight
or the Program cannot recover, when everything fails.

<a id="sources"></a>
## 4. Admission sources and binding

An admission or Replay resume asks the prefix index for its candidates: the deepest snapshot the
prompt reaches (a previous generation's endpoint or a prefill tap), and the root. The Program orders
them by "restore transfer cost + remaining prefill cost" with the Engine's context-cost
coefficients ([spec §6](hybrid-prefix-cache-spec.md)). Queries produce no use heat or cache hits.

A waiting request retains no source: queue holds only order eviction, and a request quotes the tree
again when it is next considered. A source with parts only on the opt-in SSD tier is read into the Host
tier before it binds: the quote returns no source and reports the read (`hybrid_source_pending`), the
request keeps its place as a waiting request (not an unadmittable one), and the Engine scans the queue
again when the read lands (`hybrid_poll_storage`, every cycle; [spec §5.7](hybrid-prefix-cache-spec.md#57-ssd-tier-opt-in)). A quote that went stale binds as an invalid source, so the Engine
moves to its next candidate. When a request shares a new prefix with a lane still prefilling, it can
wait for that lane's snapshot at the divergence instead of prefilling the prefix again.

A binding is one context transaction: it reserves the first unit's Device pages and the state slot,
shares the cached full pages by reference, copies a partial tail page and the snapshot into private
destinations, and restores Host-only blocks on the restore stream; the first prefill pass waits for
those copies layer by layer. After a successful binding no other source is tried.

<a id="scheduling"></a>
## 5. Scheduling, preemption and resume

### 5.1 Ordinary cycle

Each cycle first processes context transactions, terminal states and cancellation, ends a service
turn that has served its slice (§5.2), and while a turn is held tries a complete resume of its owner
if it is paused. It then requests resident unit permits row by row, the turn's owner first and then
original ticket order, keeps the runnable subset, and, while no turn is held and nothing is paused or
recovering, uses the remaining headroom to try fresh admission.
It executes the permitted compact Control/Decode set and one Prefill/Replay chunk; rows committed by
Control reacquire Decode permits in the next cycle. Graphs use the actual member count and execution
profile. With the `idle-wide-prefill` fast path a Prefill permit reserved while no other lane is
generating covers the startup-planned idle width (a multiple of the chunk) instead of one chunk,
when its extra KV pages are free or evictable cache; a held permit keeps its width until its step
runs. A binding's first step takes it only when the free pages alone hold it (and every reused
block, counted as restored); Replay and recovering requests keep one chunk
([model quality §6.7](model-quality.md#67-fast-path-registry)).

Each cycle runs one Control/Decode batch before its prefill turn, so a decoding lane beside a long
prompt waits one prefill unit between rounds. Qwen3.8-Flash-Next shortens that wait without
changing the units: while another lane decodes, each layer-walk step enqueues about 4,096 tokens
of one-call work (`kWalkStepTokensBesideDecode`) instead of one call's; a step boundary only
changes when work is enqueued, never a bit (candidates TX-12, measured below). A decode share,
where prefill turns yielded until Control/Decode work had had a share of each prefill unit's wall
time, was measured and removed (2026-11-12). The rule for both is the fastest overall first, and the
fairer only where the whole is generally as fast. FP4-Local served, three lanes, two requests decoding
1,500 tokens beside a fresh 21K-token prompt, ABBA with three repeats a pass:

| Arm | Whole set | Prompt's TTFT | Decoders during the prefill | Decoders' longest pause |
|---|---|---|---|---|
| one call's work a step | 17.31 / 18.66 / 13.30 s | 2.92 s | 1.3 tok/s | 1.63 s |
| 4,096-token steps (default) | 17.03 / 18.26 / 13.09 s | 2.90 s | 2.5 tok/s | 0.74 s |
| decode share 1 (half the wall time) | 18.01 / 18.95 / 13.63 s | 4.27 s | 15.4 tok/s | 1.59 s |
| both | 18.73 / 19.55 / 13.92 s | 5.28 s | 18.9 tok/s | 0.73 s |

The whole set is each repeat's mean of two passes; against the baseline's later pass alone (its first
pass ran cold) the 4,096-token steps are -0.1..+1.3 %. The decode share cost the whole set 1.6-4.0 %
and the prompt's TTFT 47 %, so it is not kept.

The fresh queue keeps FIFO tickets, and each blocked request can be successfully bypassed by at most C
younger requests. Each scan checks the queue head and at most C following candidates; failed checks do
not consume the allowance, and an incomplete scan continues in ticket order. Queue changes, actual
capacity releases and completed transactions trigger a scan; a request that already failed under the
same event is not quoted again, and ordinary decode does not reopen a scan. While the FIFO head
waits, the Engine prefetches its Host-only path into spare Device cache. Under pressure fresh requests
stay queued (their queue deadline still applies) until the admitted requests no longer contend (§5.2).

### 5.2 Pressure and complete resume

```text
unit shortfall → reclaim optional reservations and evict cached Device blocks
              → the blocked request takes a service turn (or, during one, waits for its own)
              → the turn's owner pauses the youngest other resident
```

**Service turns (adopted 2026-11-04, after NInfer `a754fcf3`, which fixed #434: long requests
silent for hours while short ones completed).** Every admitted request, resident or paused, has a
place in a service order kept in admission order. When a resident cannot obtain its unit after the
cache gave what it could, pressure begins and that request owns the first turn. The owner keeps the
next capacity hand-over across reclamation, pause, binding and Replay: it pauses residents for its own
unit (the youngest other one that is not binding, finished or protecting a recovery), it is resumed
first when it is the one paused, and while it is paused or binding no other resident reserves a unit,
so no one borrows capacity released for it. Once it holds its permit, the others use what is left.
During a turn only the owner may take capacity from a resident; another blocked request only records
that it contends.

A turn ends at a cycle boundary once its owner has served one second of execution that committed new
progress for it (prompt or output tokens), or as long as its restore took (Replay time) up to four
seconds; waiting for a grant and Replay never count as service, and a turn never ends while a
recovery is pending. The turn then passes to the next admitted request in the order (a paused one is
resumed, pausing whoever its unit needs), and an unused permit of the previous turn is released.
Pressure ends when a turn ends with nothing paused and no other request contending; a terminal owner
passes its turn at once. Every admitted request therefore advances within a bounded number of turns
of about a second each; the cost is Replay on the turns that pause (Qwen3.5 recomputes from the
deepest cached state; Qwen3.8-Flash-Next restores the snapshot it published), so `--max-concurrency`
and the KV capacity should still leave room for the contexts a workload runs together. While
pressure holds, fresh requests wait in the queue under their queue deadline instead of entering idle
space that a turn's owner would need next. (This replaced an age order, 2026-11-03 to 11-04, under
which a request paused behind long older generations waited for all of them.)

A pause happens at a stable Native commit boundary and does not interrupt a kernel. Other permitted
requests keep executing; the whole batch does not need to fit before running.

At most one request holds a complete resume permit at a time, the turn's owner. It rebuilds its old
history and completes its first real new unit first, while other runnable units keep executing. Native
holds the real capacity across chunks, and the runtime uses `recovery_pending` to decide when
protection ends. Reaching the old frontier or completing the MTP bridge does not count as new
progress; a new prefill, a generation/control commit or a terminal state ends the protection.

Prefill chunk boundaries are where the next chunk's lane is chosen: the resident prefill with the least
remaining prompt, aged by its wait (engine-architecture.md, "Prefill order"). When all lanes are occupied, a new
request still has to wait for some resident to finish, be cancelled or pause under resource pressure;
shortening chunks by itself does not let it enter earlier. A short request may therefore queue across
several chunks of a long request, and analyzing such latency requires checking lane admission and the
time of each execution unit separately.

The Engine advances context transactions in a bounded way at the start of the round, after admission
before execution selection, and after Control/Decode. Completed bindings promptly enter the existing
permit flow; after Control/Decode executes, only Prefill/Replay permits that have not yet executed are
refreshed, and each round is still one compact decode batch and one prefill turn. Incomplete
asynchronous events are not spin-waited on.

A deferred resume of the turn's owner is retried every cycle; while it waits, residents that it does
not need keep their lanes but reserve no new unit. When no real execution or transfer can release
resources, the turn's owner must advance through reclaim and a legal permit, and an error is reported
if the capacity contract is not met (§3.2).

### 5.3 Replay

The request's input, output objects, budget and first-time stamps persist; Native's pause state keeps
the committed ledger, its execution splits, sampling and backend resume information. After the lane
is returned, this information is still held by the request. A pause releases the lane's device
execution state; the blocks and snapshots the lane already published stay in the cache.

The resume binds like an admission, from the deepest cached state on the request's committed
history (often its own prefix), then recomputes to the pause frontier and resumes the original phase.
Replay splits prefill at every split the ledger recorded, so its GDN decomposition and therefore its
state match the original execution exactly. A pause also publishes the lane's state at its frontier
to the cache (both families; on Qwen3.5 since 2026-11-14), so its resume normally finds a snapshot at
the paused frontier itself and continues without recomputation (counted as a snapshot restore). That
capture is the paused request's resume point, not a cache tap: it is taken however close the frontier
lies to the lane's last snapshot, its Host image is admitted over any retained snapshot that is not
pinned or held, and the paused request's queue hold keeps it cached until the resume. The resume is
matched over the request's whole ledger, so a snapshot of generated tokens serves it. A Qwen3.5 lane
whose DFlash2 drafter ring is not whole at the frontier (YE-01: a prefill skips drafter positions no
planned read reaches) publishes nothing there and replays from its deepest snapshot; its resumed
prefill plans taps over the prompt it has still to prefill, so a second pause resumes past the first.

A resume happens during its owner's turn, usually while another resident still holds most of the
pool. Its source is therefore not limited to what fits beside the other residents at that moment: a
resumed request offers its deepest cached state the whole pool holds (its pause's capture when
cached), and its binding reports the shortage, which the turn covers by pausing the other resident
(or the resume waits). When evicting cached blocks cannot cover that shortage, the binding says so
(`BindingReservation::reclaim_cache` false) and the Engine pauses the resident without evicting the
cache first, which would only have evicted the resumed request's own history (2026-11-14: a resume
evicted 1,500 of its own blocks, 90 per attempt, before the pause). Taking a shallower state that
happened to fit would replay the tokens past it on every turn, and on Qwen3.8-Flash-Next with prefill
arithmetic, so the request's output would depend on when the turns paused it (2026-11-07: the
preemption tests' DFlash2-pairs replay on Dense8, and Qwen3.5's cache-replay, which reused its prompt
snapshot once in five resumes).

A fresh Qwen3.5 request follows the same rule for the history it shares with the cache: it takes a
source that fits beside the residents now only when that source leaves at most a prefill chunk of the
preferred source's cached history to compute again. Otherwise it offers the preferred source, which
waits for the residents' release while evicting cached blocks cannot make room; recomputing restorable
history needs the same pages within a few units and costs far more than the restore (2026-11-14: two
agent conversations that did not fit the pool together took sources 65K and 71K tokens short of their
history this way).

Replay does not republish old output, does not charge the budget again and does not resample historical
tokens. Penalty counts are restored from committed output and actually counted forced control; the RNG
keeps its logical position, seed and purpose. Replay advances across chunks and does not increase
cross-request reuse heat.

MTP discards unused drafts and rebuilds them in a legal unit after the resume; DFlash/DFlash2 keep their
respective context and local state rules. Vision keeps prepared media, media identity and MRoPE, and a
stale vision handoff is re-encoded on demand.
These resume constraints use the same model mathematics implementation as ordinary execution.

<a id="transfers"></a>
## 6. Context transactions and cancellation

A Program has at most one context transaction at a time, a bind or a pause; work that already holds a
permit and does not depend on that transaction keeps running. A bind follows:

```text
choose source → pin sources and reserve destinations → submit copies
              → wait for completion events → publish the lane's binding → unpin sources
```

An initial bind also acquires the first unit's permit; a pause resume bind acquires the complete resume
permit. A cancellation before the copies are submitted returns the reservations; after submission,
release waits for the real readers to finish. Owner/generation checks on sequences, snapshots and
pending handles prevent late completions from misusing reused descriptors.
Device errors go to the Engine's unified failure cleanup; ordinary resource shortfalls return to the
policy layer.

<a id="observability"></a>
## 7. Observability and performance limits

Logs and the [TTFT tool](../../tools/bench/ttft/README.md) report request results, hits and real work,
preemption resumes and user-visible latency respectively. Initial prefill and Replay are counted
separately; cancellation counts only work already executed. The `throughput` record's
`context_cache` object reports selections, state operations and occupancy, and its `hybrid` object
the cache's blocks, snapshots, taps, Host traffic and evictions.
`GenerationStart` and the first admitted/first-token events are published only once, and a pause does
not recompute the ingress timeout.

The evaluation order is: first check whether reuse that should exist was lost, then look at severe
regressions and in-stream stalls, and finally analyze the copy, recompute, CPU scheduling and protocol
preparation costs of individual scenarios. A complete hit may still need a lot of H2D; a low TTFT may
still come with an output gap after a long pause. Report the two separately.

Fixed typed pools, the complete resume permit, Replay and Vision re-encoding can all incur real
costs. Throughput, TTFT, completion time and the maximum output gap together describe the result; a
single sample describes only that trajectory and establishes no latency distribution guarantee.

Implementation entry points:

- [Scheduler](../../src/runtime/engine/scheduler.h), [HybridResourceManager](../../src/runtime/engine/context_cache/hybrid_resource_manager.h): membership and policy.
- [EngineCore](../../src/runtime/engine/engine_core.h): transaction orchestration, lifecycle and publication.
- [Program](../../src/models/qwen3_5/program/program.h): model restore and resource operation contracts.
- [Native source organization](../../src/models/qwen3_5/program_sources.cmake): planning, stores, binding, the prefix cache and the implementation of each transaction.
