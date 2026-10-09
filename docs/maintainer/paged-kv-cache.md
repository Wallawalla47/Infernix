# Infernix Paged KV Context Store

This document defines the physical storage and consumer contract of Infernix's growing KV. It is the
maintainer authority for typed KV pools, logical pages, address spaces, reservations, block tables and
GPU consumer views.

Request order and lifecycle are in [Engine architecture](engine-architecture.md); preemption and
resource policy are in [Resource scheduling](resource-scheduling-and-context-cache.md); prefix reuse,
the cache's Host tier and eviction are in the [hybrid prefix cache](hybrid-prefix-cache-spec.md). The KV
Store fulfills
the physical requirements of the selected operations and provides model execution units with stable
direct-access views.

---

## 1. Physical model

Growing KV uses a set of homogeneous pools fixed at startup. Each pool:

- stores all planes that share the same frontier and lifetime;
- uses a fixed token page size, plane order and page-group count;
- has its own physical page-ID namespace and free capacity;
- provides one logical address space per active sequence;
- is addressed directly by consumers through a block table.

A request's KV does not need to be physically contiguous and is not permanently bound to a control
lane. All active and inactive address spaces share the pool capacity. An ordinary reservation
guarantees the next unit's increment; a pause-resume reservation covers the rebuild to the old frontier
and the first real new unit, and is held across chunks.

Paged storage covers KV that grows with the context. DFlash local cyclic state, Vision/query temporary
K/V and other fixed state have different lifetimes and are managed by their own StateImage or workspace
contracts.

---

## 2. Three independent granularities

The KV architecture distinguishes:

| Granularity | Meaning | Current contract |
|---|---|---|
| allocation granularity | How much token payload a pool acquires or releases at once | `P=64` for every growing pool |
| valid-frontier granularity | Which logical position a consumer can read up to | 1 token |
| reusable-state granularity | Which frontier has a complete model continuation | target-defined snapshot |

A page boundary is not an Attention mask boundary and not a prefix hit boundary. A valid frontier can
lie at any offset inside a page.

The KV Store can represent, truncate or protect a prefix at any token frontier; this does not prove the
model can resume from that position.
A reusable frontier requires both a complete StateImage and the target-defined backend state; the
specific rules are in the [hybrid prefix cache](hybrid-prefix-cache-spec.md).

---

## 3. Typed pool set and capacity

### 3.1 Pool set

The model config and the selected speculative backend determine the pool set at startup:

```text
ordinary:
    Main Text

MTP:
    Main Text
    MTP

DFlash / DFlash2:
    Main Text
    Draft Full, when the selected draft config contains full-attention layers
```

Speculative backends are mutually exclusive within one Engine, so there are currently at most two
growing pools. All five layers of the official DFlash2 config are local attention, so it needs only the
Main growing pool, and its draft context uses cyclic storage.

| Pool | Content | Logical frontier |
|---|---|---|
| Main Text | target full-attention K/V and its code/scale planes | target materialized KV frontier |
| MTP | MTP persistent K/V and its code/scale planes | MTP KV frontier |
| Draft Full | the selected draft's full-context K/V | draft context frontier |

Main Text and MTP use the KV profile the Engine selects: BF16, INT8-G64, FP8-E4M3FN-row256, NVFP4-G16,
K8V4, VQ2 or K4V2; Draft Full uses its own BF16 profile. The physical layout under the `BFloat16` name is
BF16 K and FP16 V, and the writer converts BF16 V to FP16 once. K8V4 is a closed asymmetric profile, not
a runtime bit-width combination: K is fixed to FP8-E4M3FN-row256 and V to NVFP4-G16.

VQ2 and K4V2 are vector-quantized profiles: a row is first rotated by a fixed normalized Hadamard H256;
VQ2 stores one 16-bit code word per 8 dimensions (512 trained INT8 magnitude patterns × 7 sign bits, with
the 8th sign given by even parity), and K4V2 stores K as 4-bit Lloyd-Max scalar codes and V as the same
VQ2 codes; each row has one unbiased FP16 scale
\(S=\lVert y\rVert^2/\langle y,c\rangle\). The distance used to choose a code word is computed in the
fixed FMA order specified by [`vq2_codec.cuh`](../../src/ops/kv_cache/vq2_codec.cuh), and ties take the
smallest pattern index; whether append goes through the block kernel or the warp-per-row kernel for wide
calls, the same row gets bit-identical codes, so chunk splitting and prefix-cache reuse do not change
the stored KV. Both carry an exact recent-key window (9.3), which belongs to sequence state
(StateImage) and is not in the page pool.

`PagedKVStorageLayout` resolves the selected closed profile into the K/V data/scale plane schema; the
target planner expands that schema per layer and determines plane ordinals. The common pool
implementation still receives only the expanded `KVPageGeometry`, plane inventory and capacity, and
does not interpret the storage mode.

### 3.2 Main capacity

Let:

- \(S\): the `max_context` of one sequence;
- \(C\): `max_concurrency`;
- \(P=64\): the Main page size;
- \(L=\lceil S/P\rceil\): the logical page capacity of one address space;
- \(M\): the physical page-group count of the Main pool.

The available range is:

\[
M_{min}=\max(L,C)
\]

\[
M_{max}=C\,L
\]

\(M_{min}\) satisfies, respectively, reaching \(S\) when one request is exclusive and the geometric lower
bound of C requests each needing one minimum page.
\(M_{max}\) is the physical upper bound when all active requests reach the per-sequence ceiling at the
same time.

The explicit `kv_capacity` policy resolves to:

\[
M=\left\lceil K_{main}/P\right\rceil
\]

and requires \(K_{main}\ge S\) and \(M\in[M_{min},M_{max}]\).

### 3.3 Automatic capacity

After the weights are loaded, the automatic policy resolves \(M\) once from the currently available
device memory \(F\) and the required headroom \(R\).
The model planner provides an affine `SequenceCapacityCurve` from the bound parameters and the selected
execution domain:

\[
B(M)=B_{min}+(M-M_{min})B_{step}
\]

where \(B(M)\) is the complete runtime Device reservation for that Main capacity, including:

- the Main and selected backend typed pools;
- active and snapshot State storage;
- block tables and fixed persistent state;
- the unified workspace;
- the CUDA Graph allowance.

Automatic selects:

\[
M=
\min\left(
M_{max},
M_{min}+
\left\lfloor
\frac{F-R-B_{min}}{B_{step}}
\right\rfloor
\right)
\]

and requires \(F\ge R+B_{min}\). When \(M_{min}=M_{max}\), that single point is taken directly. The target
uses the same production layout builder to generate \(B_{min}\), \(B_{step}\) and the final layout, and
validates the capacity curve; the resolver does not duplicate the model dimension formulas or guess
capacity by allocation probing.

The final public Main KV capacity is \(M\,P\) token-equivalents. Page rounding only adds physical
padding and does not raise the logical ceiling \(S\) of a single sequence.

### 3.4 Backend capacity

The selected backend's logical page capacity is still \(L\). Its physical capacity is derived from the
Main \(M\):

```text
backend off:
    Main physical pages = M

MTP with draft window K:
    Main physical pages = M
    MTP physical pages  = M + C * ceil((K - 1) / P)

DFlash / DFlash2 with full-attention layers:
    Main physical pages        = M
    Draft Full physical pages = M
```

MTP's extra pages only cover each active row's provisional lead over Main within one speculative round,
and do not raise the logical capacity of any address space. Draft Full has no such provisional lead; a
draft config with no full layer does not allocate this pool.

The pools are physically separate. A free page of one pool cannot become payload of another pool. The
Program establishes the complete typed capacity vector once at startup and does not grow or
redistribute the pool geometry at runtime.

### 3.5 Host capacity

The prefix cache's Host tier is one startup-fixed pinned slab pool that cached KV blocks and snapshot
images share ([spec §5.4](hybrid-prefix-cache-spec.md)); Core's record-addressed Host copies move whole
pages between it and the Device pools. It does not raise the Device capacity or the single-sequence
context ceiling. Input, the request ledger and other CPU data have their own lifecycles and are not
counted against it.

---

## 4. Page groups and physical layout

### 4.1 Grouping invariant

Planes go into the same pool only when all of the following hold:

1. they use the same logical cache ordinal;
2. they share the same committed frontier;
3. they are reserved, materialized, truncated, retained and released together;
4. they use the same page size;
5. they have no semantics of independent release or independent capacity reuse.

The K, V, code and scale of the Main layers can therefore share one page-group ID; Main, MTP and DFlash
Full must belong to different pools.

### 4.2 Page group

One pool-local page-group ID \(g\) selects the payload of that logical block in all grouped planes at
once:

```text
page group g
├── layer/plane 0 slice g
├── layer/plane 1 slice g
├── ...
└── layer/plane n slice g
```

Planes have Engine-lifetime-stable backing but need not form one contiguous blob. A page group is the
smallest Device unit of allocation, reservation, reference and transfer.

Physical IDs can be in any order. The allocator can prefer returning contiguous IDs to improve locality,
but correctness, admission and kernel launch topology do not depend on contiguity. Fixed-size page
groups produce no variable-size external fragmentation and need no Device compaction.

### 4.3 Closed Device plane orders

All registered growing pools use \(P=64\) and choose one of two closed orders.

Main Text and MTP use page-major:

\[
[X,P,H,N_{physical}]
\]

DFlash Full uses a head-major page run:

\[
[X,P,N_{physical},H]
\]

where:

- \(X=D\) for a K/V or quantized code plane;
- the INT8-G64 scale plane uses \(X=D/64\);
- the FP8-E4M3FN-row256 scale plane uses \(X=1\);
- the NVFP4-G16 code plane uses packed U8 \(X=D/2\), and its scale plane uses U8 \(X=D/16\);
- \(H\) is the number of KV heads;
- \(N_{physical}\) is the physical page count of that pool.

For logical position \(p\):

\[
b=\lfloor p/P\rfloor,\qquad o=p\bmod P,\qquad g=block\_table[b]
\]

The page-major address is:

\[
address=base+d\,nb_0+o\,nb_1+h\,nb_2+g\,nb_3
\]

The head-major address is:

\[
address=base+d\,nb_0+o\,nb_1+g\,nb_2+h\,nb_3
\]

Code and scale planes use the same \(g\) but their own Tensor's leading coordinate and strides.
The exact persistent codec is defined by [`kv_cache_append.h`](../../include/infernix/ops/kv_cache_append.h)
and the consumer arithmetic by [`softmax_attention.h`](../../include/infernix/ops/softmax_attention.h); the
allocator interprets only plane bytes, order and page-group identity.

The physical payload per token/head of the D256 Main/MTP profiles is:

| profile | K code + scale | V code + scale | K+V |
|---|---:|---:|---:|
| BF16 | 512 B | 512 B | 1024 B |
| INT8-G64 | 256 B + 8 B | 256 B + 8 B | 528 B |
| FP8-E4M3FN-row256 | 256 B + 2 B | 256 B + 2 B | 516 B |
| NVFP4-G16 | 128 B + 16 B | 128 B + 16 B | 288 B |
| K8V4 | 256 B + 2 B | 128 B + 16 B | 402 B |
| K4V2 | 128 B + 2 B | 64 B + 2 B | 196 B |
| VQ2 | 64 B + 2 B | 64 B + 2 B | 132 B |

The K/V code and scale planes have their own dtype, leading extent and group size; they still share
page-group identity, frontier and lifetime. The capacity curve, the prefix cache's Host copies and
the memory summary are all computed from this typed plane inventory; `2 * vector_bytes` must not be
used in place of K8V4's asymmetric byte count.

### 4.4 Logical position domain

The block table uses the autoregressive cache ordinal:

```text
logical block b covers positions [b*P, (b+1)*P)
```

RoPE, MRoPE, Vision axes and `rope_delta` are Attention input metadata and do not change KV slot
ownership.

### 4.5 Page payload

A pool's logical page-group payload is:

\[
PageBytes=\sum_{plane} PlaneBytesPerToken\cdot P
\]

Startup physical bytes are derived from the complete span and alignment of each plane slab. `PageBytes`
can differ between pools, but all page groups within one pool are equivalent.

`P=64` simultaneously suits the current 32/64-key Attention tiles, 128-token aligned prefill chunks,
bounded block-table metadata and bounded tail slack. Prefix hit granularity plays no part in choosing the
page size. Changing the page size, grouping or closed plane order is an architecture change.

---

## 5. Logical pages

### 5.1 Logical page identity

A Device page ID is only the physical location of a page's payload, not prefix identity. The Program
maintains generation-checked logical pages for each pool:

```text
LogicalKVPage
├── object generation
├── committed columns [0, P]
├── protected columns
├── address-space and prefix-cache references
├── active references and writer state
├── Device page-group lease
└── source pins and transfer-destination state
```

Speculative or not-yet-committed bytes do not extend committed coverage.

The same logical page keeps the same page index in every address space that references it. A
cached-prefix fork shares pages at their original position, COW and growth create new logical pages,
and truncation only deletes the suffix; an address space also never references the same page twice.

### 5.2 Device page leases

A page uses the consumer-native plane layout of section 4. A `DeviceKVPageLease` exclusively owns one
pool-local physical page group; the generation prevents stale handles after release/reuse.

A pool distinguishes, for the same capacity unit:

```text
allocated page lease
reserved but not materialized page
globally available page
```

\[
allocated+reserved+available=capacity
\]

Materialize turns a reservation into a lease; dematerialize returns a lease to the same reservation.
Ordinary unit settlement releases the remainder; a protected resume keeps the remainder needed for
future coverage and returns it after real new progress. Shared prefix references do not occupy
physical pages twice.

### 5.3 Transfer destinations and source pins

A copy into a new page (a cached partial tail copied into a lane's private page, or a Host restore)
materializes a transfer destination that no address space references yet; it is published to its
owner only after the copy completes, and aborting it returns the page to its reservation. A source pin
keeps a page's payload and coverage unchanged while a copy or fork reads it.

Host transfers copy complete page payloads and can merge adjacent physical IDs into fewer transfer
runs. Core chooses a 2D copy from the bound plane geometry: a short PageMajor run can merge planes of
equal width and spacing, and HeadMajor can choose page or head as the outer submission dimension. The
merged path is used only when it reduces the call count and the pitch is legal; the Host layout and
payload do not change.

### 5.4 Descriptor lifetime

A logical descriptor is reclaimed, and its generation advanced, when its last reference is released
and no pin or transfer holds it; its Device lease returns to the pool.

---

## 6. KV history and address space

### 6.1 KV history

A `KVHistory` holds a sequence's Main and optional backend address spaces. The prefix cache's block tree
holds its own references to the full pages a lane commits; a later request shares those pages through
a cached-prefix fork, so two histories can share complete physical prefix pages but never the suffix of
a mutable directory. State and KV of one binding always come from the same actual history.

### 6.2 Address space

```text
KVAddressSpace
├── logical-block -> LogicalKVPage directory
├── committed frontier
├── active/inactive state
├── unit / recovery growth reservation
└── optional execution-row lease
```

The directory maintains ordered membership through shared immutable prefix nodes and a private append
path. The block table is the execution mirror of the active Device mapping. An inactive history occupies
no execution row and is not bound to its original lane.

Each address space records separately:

| Fact | Meaning |
|---|---|
| membership | Logical pages that already belong to this address space |
| committed frontier | The token prefix that has formed canonical content |
| growth reservation | Incremental pages acquired for a unit or complete resume coverage but not yet materialized |

Membership can cover a provisional suffix such as the speculative window, whose bytes cannot serve as a
complete restore point before commit.
The Main/backend frontiers can differ; their semantic relationship is determined by the model schedule.

### 6.3 Execution rows

Each pool establishes a fixed-address Device block-table matrix at startup:

\[
block\_tables[N_{logical},C]
\]

where \(N_{logical}=L\) and \(C=max\_concurrency\). Each active address space leases one row, whose
entries are I32 pool-local physical page IDs. Activation publishes the Device page IDs of the membership
in bulk; a resume can lease a different row.
An execution row does not own logical pages, a frontier or a reservation.

## 7. Finite execution units and lifecycle

### 7.1 Binding

On an initial bind from the root or a cached prefix, the Program prepares State, all enabled KV pools and
the first legal unit together.
A pause resume instead prepares the complete resume coverage: the peak needed for the old frontier,
backend normalization/bridge and the first real new unit.

```text
pin the cached source and acquire destinations
  -> reserve the private tail, the State slot and unit/recovery growth
  -> restore Host-only blocks
  -> fork the shared pages and copy the tail
  -> lease execution rows and publish mappings
  -> publish complete sequence
```

A shortfall in any pool blocks the complete bind. The source pins cover the required transfers and
installation; a failure before publication cleans up the destination and keeps the cached source,
and later exceptions go to Engine failure cleanup. One bind's State and KV are never spliced from
different computation histories.

### 7.2 Unit reservation and materialization

The Program computes the maximum write position in each pool of the selected prefill, Replay, decode,
control or normalization unit.
Existing membership is not reserved again; the full typed demand of one call set is checked first, then
the permit is installed. The permit fixes the unit kind, token parameters and typed frontiers, and
execution must match them. The Engine forms the runnable subset by requesting row by row.

The complete resume permit is separate from the current unit's parameters: the address space holds the
reservation needed up to the end of the resume coverage, and the current Replay chunk materializes only
the part it needs. During the resume, every unit must fall within this already acquired coverage, and
other requests cannot take the unconsumed remainder.

`ensure_mapped_to_tokens()` receives the lower bound of coverage needed for this phase: if the existing
membership suffices it returns directly; otherwise it turns that address space's reservation into
physical pages and publishes the table slice. It does not advance the committed frontier and does not
trim a longer speculative mapping. Needing more pages than `membership + reservation` is a permit
violation.

Target prefill/verify secures Main KV; MTP and DFlash Full secure their backend KV respectively. The
DFlash2 local draft context uses fixed cyclic state. Ordinary decode usually materializes only one Main
page when it crosses a page boundary.

### 7.3 Commit and rollback

After a unit succeeds, the Program commits the corresponding State and canonical KV frontiers. The
speculative path first completes the recurrent state, hidden and backend context the accepted prefix
needs, then publishes the frontier. Consumers do not read the rejected suffix; leftover bytes in a
partial page do not extend the valid range.

At settlement, uncommitted tail pages are explicitly truncated and dematerialized, after which an
ordinary unit releases the remainder; during a resume, the pages still needed for the complete resume
coverage are kept. Reaching the old frontier does not release them by itself; a real new prefill/token
commit or a terminal state ends the protection.
`truncate` cannot overwrite the protected coverage of a shared page, nor a partial tail another
reader needs. Merely requesting shorter coverage does not trigger trimming.

### 7.4 Pause, finish and release

On pause or finish, the execution row, active references and unused growth reservation are released,
and the lane's committed full pages stay referenced by the prefix cache. A paused request keeps no KV;
it resumes through Replay. When the last reference to a page disappears, its Device lease is
returned.

### 7.5 Stable boundary

From block-table publication until the GPU unit completes, the selected rows, membership, readable
frontier and accessed payload all stay stable. Mapping updates, frontier commits, truncation and row
recycling complete at GPU boundaries; allocator and transfer ownership never enter a kernel.

## 8. Cached-prefix fork and write protection

### 8.1 Forking a cached prefix

A request admitted from a cached snapshot acquires an empty address space and an execution row and
shares the complete cached pages before the snapshot's frontier. If the frontier falls in a partial
page, the cached tail is first copied to a private destination, and then appending begins.

For example, with \(P=64\) and frontier \(F=1000\): the first 15 complete pages are shared, and one
private tail page is copied for the last 40 valid tokens. The snapshot lies exactly at token 1000;
allocation does not round the hit frontier down to 960.

### 8.2 Write protection

The stores check protected coverage, active references, the writer and source pins together. Shared
full pages carry protected coverage and are never truncated; multiple read-only references can share
one page. A branch that needs to write a shared partial tail first acquires a private COW page. A
refcount by itself does not grant write permission.

---

## 9. Speculative and non-growing KV

### 9.1 Independent pool frontiers

MTP and DFlash backends with full layers use the Program KV Store's backend pool and do not build a
separate allocator.

In one speculative unit, Main and the backend:

- materialize from their own unit reservations;
- can have different mapped/provisional frontiers;
- commit their accepted frontiers separately;
- trim rejected trailing mappings separately.

During an MTP draft, the backend mapped extent can temporarily lead Main; DFlash Full usually lags
behind Main. A provisional lead does not form committed snapshot coverage. Rejected bytes can remain
in a partial page, but must be overwritten by a new canonical write before any later read.

### 9.2 Fixed and transient K/V

The following storage does not enter the growing pools:

| Resource | Owner |
|---|---|
| DFlash/DFlash2 local sliding-window K/V | fixed per-sequence StateImage |
| DFlash/DFlash2 boundary-local snapshot | fixed snapshot StateImage |
| Vision/query temporary K/V | Program workspace |

DFlash/DFlash2 cyclic K/V uses its own `CyclicKVCacheLayerView` and modulo/window semantics; it holds no
page ID, block table or growing reservation. The model config determines its layer count, heads and
window.

### 9.3 VQ2/K4V2 exact recent-key window

For VQ2 and K4V2, every attention layer (the Main Text layers followed by the MTP layer) holds an exact
window in each StateImage slot: 64 sink slots plus 1024 ring slots (the slot of position \(p\) is
\(p<64\,?\,p:64+(p\bmod 1024)\)); each slot, KV head and K/V role has one row of rotated INT8-G64 values,
4 FP16 group scales and a tag. The tag is an FNV-1a hash of (position, the stored code words, the scale
bits), forced to be odd; a zero tag never matches, so zeroing the tags clears the window.

The read rule ([`softmax_attention.h`](../../include/infernix/ops/softmax_attention.h)) depends only on
position: a query at \(q\) reads the exact INT8-G64 row for sinks and for keys with \(j\ge q-768\) (keys
appended by this call read the INT8-G64 of their input row; earlier keys read the slot when the tag
matches the stored codes and otherwise read the codes), and reads codes for all other keys. So the same
context reads the same key representation however it is split into prefill chunks, decode steps or
prefix-cache reuse points. When a CTA's queries cover \([q_0,q_1]\), keys in \([q_0-768,q_1-768)\) are
exact for some queries and codes for the rest. For tiles that intersect this band, the prompt kernel
first does an exact pass and then one masked codes pass (at most 127 keys for a 128-row CTA). The prompt
kernel's key tile is 32 keys (double-buffered INT8 tiles); once the call-visible keys reach 1536 it
switches to 64-key tiles (one INT8 tile per page, with codes prefetch hiding their reads and exact-row
tiles read synchronously): measured 6-9 % faster at 32-128K visible keys, while for short chunks most
tiles are exact rows and 64-key is 3-12 % slower, hence this threshold. The decode/verify kernel instead
lets a row's first few split CTAs process only the sink and recent tiles (the exactly read pairs), and
the remaining split CTAs process only the pairs read from codes; each (key, query) pair is counted in
exactly one CTA, and merge combines all splits. The grouped kernel's key tile is 32 or 64 keys: 64 is
used only for 16-column CTAs and requires one block's shared memory to fit (vq2's 16 columns fit; k4v2
has 130 B of K code per row, and 16 columns need 103296 B, above this target's 101376 B per-block limit,
so it is always 32). Measured, the 64-key tile is 7-11 % faster on vq2 16-column calls with ≤8K keys
(W=16 relative to INT8 drops from 1.89x to 1.71x, B=1, 2K), but 13-44 % slower on 8-column CTAs because
each SM fits one fewer block, so it is not used for 8 columns. Codes CTAs read only keys earlier than any
query's window, so they run in parallel with the append as its programmatic dependent, while window
CTAs first wait for the append to complete. The tag is self-validating, so trim, rollback, rejected
drafts or leftover old content only fall back to codes and never read the wrong row. Calls of width at
most 256 write the slots directly during append; wider prompt calls first write the window rows to
workspace staging and commit the sinks and the last 1024 columns after attention.

The window is sequence-local state; like DFlash local K/V it is held by the StateImage and indexed by
destination state slot:

- a batched view's `window` covers all slots, and `window.slots[b]` is the destination slot of compact
  row \(b\); a single-sequence view binds only that sequence's one slot;
- `zero_slot` zeroes the tags, and `copy_slot` and Host snapshot/restore carry all window bytes;
- a StateImage Fork copies the source window to the destination with `copy_fork_local` before the first
  write;
- tree compaction moves the accepted column's slot to the chain position and recomputes the tag
  (zeroing it when the source slot is already invalid).

So the window restored by a prefix-cache hit is byte-identical to the window at the same frontier on a
miss. The window per slot per layer is \(1088\times H_{kv}\times 536\) B (Qwen3.6/3.8-27B: 17 layers,
\(H_{kv}=4\), about 39.7 MB); it occupies the Device StateImage slot and is also counted in the Host
snapshot image and in fork transfer work.

---

## 10. Consumer contract

### 10.1 Single-sequence view

A single-sequence growing-cache Op uses the non-owning `PagedKVLayerView`:

```text
PagedKVLayerView
├── k_pages / v_pages
├── optional k_scale_pages / v_scale_pages
├── block_table          I32 [Nlogical]
├── head_dim
├── num_kv_heads
├── storage
└── window               VQ2/K4V2 exact window of this sequence's slot (9.3); empty otherwise
```

The view contains only one layer's plane tensors and one block-table row. It contains no allocator
handle, request identity, reservation, ownership or frontier.

### 10.2 Batched view

A batched Op uses `PagedKVBatchLayerView`:

```text
PagedKVBatchLayerView
├── shared layer planes
├── block_tables         I32 [Nlogical, C]
├── head_dim / num_kv_heads
├── storage
├── window               VQ2/K4V2 window planes of every StateImage slot, window.slots I32 [B]
└── table_rows[B]        separate Op input
```

`table_rows[b]` selects the active address-space row for compact row \(b\). Compact batch order, Engine
lane and table row can all differ from one another. Per-row context length, valid columns and positions
are provided by the Op's other typed inputs.

### 10.3 Address translation

With the current \(P=64\):

```text
logical_block = position >> 6
page_offset   = position & 63
physical_page = block_table[logical_block]
```

The element address is then computed with the closed plane order of section 4.3. A batched consumer
first selects the table row with `table_rows[b]`, then performs the same translation.

The wrapper validates Tensor dtype, geometry, closed strides, table shape and the execution envelope.
The caller guarantees that:

- the logical blocks this call may access are materialized;
- the read domain does not exceed the sequence's frozen valid frontier;
- writable K/V code and scale are complete before frontier publication;
- the selected table rows are stable within the GPU unit.

### 10.4 Direct paged execution

Growing-cache Ops consume paged views directly. Page translation is computed and reused at page/tile
granularity, and the inner loop does not interpret request, pool kind or allocator state. Kernel
correctness cannot depend on adjacent logical pages mapping to adjacent physical IDs.

Paging introduces no gather-to-contiguous cache and no staging copy proportional to context length.
Route-specific tile, split, warp and shared-memory schemes can be optimized independently as long as
they keep the same logical Attention, persistent codec and the address contract above. The Op numerical
and performance admission rules are in [Op development](op-development.md).

---

## 11. CUDA Graph and table publication

Plane bases and the block-table matrix base are stable for the Engine lifetime. What changes across
replays is:

- table content;
- `table_rows` selectors;
- positions, context lengths and valid counts;
- model state selectors.

Page IDs, request identity and physical contiguity do not enter the graph key.

Before an execution unit that needs new mappings, the Program:

```text
materialize all required pages
  -> update host-side membership
  -> publish one contiguous table slice
  -> launch/replay consumer on the ordered stream
```

Batched publication is used when several pages are added or a complete membership is reactivated at the
same boundary. A mapping update must precede its consumer, and the same row must not be rewritten while
a replay is in flight.

---

## 12. Core invariants

1. Each Device page-group lease carries at most one logical page in its pool.
2. A pool only groups planes that share frontier, lifetime, page size and allocation semantics.
3. A pool's K/V/code/scale planes use the same page-group ID for the same logical block.
4. Device occupancy counts allocated leases and reservations; logical aliases are not charged twice.
5. Logical page identity and physical page ID are independent of one another.
6. The valid frontier is exact to the token; page boundaries do not change Attention or snapshot
   semantics.
7. Coverage a cached page protects cannot be overwritten by a writer; any logical page has at most
   one writer.
8. Shared full pages are immutable; a non-aligned writable tail first gets a private COW page.
9. A transfer destination is published only after its copy completes.
10. Ordinary unit settlement returns unused capacity; a resume reservation is kept across chunks until
    real new progress or a terminal state.
11. Within one GPU execution unit, membership, block tables, pages and the read frontier are stable.
12. An inactive address space occupies no execution row; an execution row owns no logical pages.
13. Main and backend pools reserve, materialize, commit and truncate separately.
14. Growing-cache consumers access KV only through paged views and block tables, and acquire no allocator
    or ownership authority.
15. Kernel correctness does not depend on physical page ID contiguity and does not build
    request-contiguous KV by gathering.
16. Snapshot reusability is proven by a complete target continuation; the existence of KV pages by
    itself does not constitute a hit.
17. A block-table publication's H2D reads the execution row's pinned shadow only when the stream reaches
    it. A later publication of the same row (including the next owner's publication after release) waits
    for that copy to complete before rewriting shadow entries that overlap a copy still in the queue;
    non-overlapping entries do not wait.

---

## 13. Implementation locations

| Responsibility | Main location |
|---|---|
| Device page pools, reservations and execution tables | `src/core/paged_kv_cache.*` |
| Closed K/V data/scale plane schema | `src/core/paged_kv_storage.h` |
| Logical pages and references | `src/models/qwen3_5/program/storage/logical_kv_store.h` |
| Address spaces, directories and the cached-prefix fork | `src/models/qwen3_5/program/storage/kv_address_space.h` |
| History lifecycle | `src/models/qwen3_5/program/storage/sequence.cpp` |
| Prefix cache pages, Host slabs and transfers | `src/models/qwen3_5/program/prefix/` |
| Unit permits and context transactions | `src/models/qwen3_5/program/planning/request_plan.cpp`, `transactions/` |
| Model pool layout and capacity curve | `src/models/qwen3_5/program/planning/startup.cpp` |
| Public paged consumer views | `src/core/paged_kv_cache.h` |
| Growing-cache Ops | `include/infernix/ops/`, `src/ops/` |

Exact model state and backend mathematics are in [Qwen3.5 model](qwen3_5-model.md) and
[DFlash](dflash.md); the persistent KV codec and the causal consumer numerical contract are defined by
the growing-cache Ops in the table above. The paths locate the current implementation and do not promote
the files or class names themselves to external interfaces.
