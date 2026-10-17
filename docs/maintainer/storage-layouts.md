# Infernix Persistent Storage Layouts

This reference records the persistent tensor layouts and resource encoding used by current
`.ninfer` artifacts, including alignment, byte order, padding, encoded-size rules, and logical
decode. Numeric semantics come from [`tensor-formats.md`](tensor-formats.md); framing and object
ranges come from [`artifact-container.md`](artifact-container.md).

## 1. Registered identities

The storage registry contains exactly these identities:

| Identity | Kind | Compatible numeric formats | Logical shape | Object alignment |
|---|---|---|---|---:|
| `contiguous_le_v1` | tensor layout | `bf16`, `fp32`, `int32` | rank `0..16` | 256 bytes |
| `row_split_k128_v1` | tensor layout | `q2_g64_fp16`, `q4_g64_fp16`, `q5_g64_fp16`, `q6_g64_fp16`, `q8_g32_fp16` | rank 2 `[N,K]` | 256 bytes |
| `block_scale_k16_m128x4_v1` | tensor layout | `nvfp4` | rank 2 `[N,K]`, `N % 128 == 0`, `K % 64 == 0` | 256 bytes |
| `row_scale_v1` | tensor layout | `fp8_e4m3fn_row_bf16` | rank 2 `[N,K]` | 256 bytes |
| `nvfp4_expert_rg16_v1` | tensor layout | `nvfp4_mul` | rank 3 `[E,H,I]`, `H % 16 == 0`, `I % 16 == 0` | 4096 bytes |
| `ggml_rows_v1` | tensor layout | every `ggml_*` format | rank 2 `[N,K]`, `K` whole blocks | 256 bytes |
| `ggml_tiles16_v1` | tensor layout | `ggml_q4_k`, `ggml_q5_k`, `ggml_q6_k`, `ggml_iq4_xs` | rank 2 `[N,K]`, `K` whole blocks | 256 bytes |
| `ggml_expert_iq4_nl_v1`, `ggml_expert_q2_0_v1` | tensor layout | `ggml_iq2_s`, `ggml_iq3_xxs`, `ggml_iq3_s`, `ggml_iq4_xs` | rank 3 `[E,H,I]`, whole blocks | 4096 bytes |
| `fp8_expert_b128_v1` | tensor layout | `fp8_e4m3fn_b128` | rank 3 `[E,H,I]`, `H % 128 == 0`, `I % 128 == 0` | 4096 bytes |
| `raw_bytes_v1` | resource encoding | not applicable | nonempty byte string | 1 byte |

These format/layout pairs define the current codec support. Native consumer requirements are
covered separately in Section 12.

Object alignment applies to the object's payload-relative `offset` in the `.ninfer` JSON. Internal
plane offsets and padding belong to the selected layout. Inter-object padding belongs to the
container and is not included in an object's `bytes`.

The helper used below is:

```text
align_up(x, a) = ceil_div(x, a) * a
```

All formula inputs and intermediate results are nonnegative integers. The container implementation
must reject a shape or calculation that cannot be represented by its file-offset and size types.

## 2. `contiguous_le_v1`

### 2.1 Logical traversal

`contiguous_le_v1` stores direct logical words in C order: the last logical dimension varies
fastest. For shape `[D0, D1, ..., D(r-1)]`, coordinate `[i0, i1, ..., i(r-1)]` has linear index:

```text
index = (((i0 * D1 + i1) * D2 + i2) ... ) * D(r-1) + i(r-1)
```

A rank-zero shape `[]` contains one scalar word. For every other legal shape:

```text
elements = product(shape)
```

The layout performs no reshape, transpose, type conversion, or numerical canonicalization. Those
operations, when required, occur before layout encoding under the model-specific conversion recipe.

### 2.2 Word bytes

Words are serialized least-significant byte first:

| Format | Bytes per element | Stored word |
|---|---:|---|
| `bf16` | 2 | the exact 16-bit bfloat16 logical word, little-endian |
| `fp32` | 4 | the exact 32-bit IEEE-754 binary32 logical word, little-endian |
| `int32` | 4 | the exact 32-bit two's-complement logical word, little-endian |

Signed zero, subnormal, infinity, NaN payload, and integer-word behavior are determined by the
numeric-format contract. The layout only preserves the word bits.

### 2.3 Encoded size

There is no internal prefix, stride table, per-row padding, or trailing padding:

```text
payload_bytes = elements * bytes_per_element(format)
```

The tensor object's JSON `bytes` must equal `payload_bytes` exactly.

## 3. `row_split_k128_v1`

### 3.1 Logical and physical geometry

The logical tensor is a positive rank-two matrix `[N,K]`. The format supplies code width `b` and
group size `G`:

| Format | `b` | `G` | Base bytes per group `B` | High bytes per group `H` |
|---|---:|---:|---:|---:|
| `q2_g64_fp16` | 2 | 64 | 16 | 0 |
| `q4_g64_fp16` | 4 | 64 | 32 | 0 |
| `q5_g64_fp16` | 5 | 64 | 32 | 8 |
| `q6_g64_fp16` | 6 | 64 | 32 | 16 |
| `q8_g32_fp16` | 8 | 32 | 32 | 0 |

The layout extends the last axis to a multiple of 128:

```text
K_pad              = align_up(K, 128)
groups_per_row     = K_pad / G
logical_groups     = ceil_div(K, G)
physical_group_cnt = N * groups_per_row
```

`K_pad` is physical geometry and is not added to the JSON `shape`. Because both registered group
sizes divide 128, `groups_per_row` is integral.

For the final partially logical group, lanes whose column is at least `K` have signed code zero. Its
scale remains the scale of the logical group defined by the numeric-format contract. Any complete
physical group after `logical_groups` has scale word `0x0000` and all codes zero. Physical padding
therefore cannot change a decoded logical value.

### 3.2 Payload planes

The payload contains three conceptual planes in this order:

```text
base-code plane
zero padding to a 256-byte boundary
optional high-bit plane
zero padding to a 256-byte boundary
binary16 scale plane
```

Q2, Q4 and Q8 have no high-bit bytes. They still place the scale plane at the first 256-byte boundary
after the base-code plane. There is no padding after the scale plane inside the object.

Within every plane, traversal order is:

```text
row 0 group 0, row 0 group 1, ..., row 1 group 0, ...
```

Bytes belonging to one group are adjacent.

### 3.3 Base-code plane

For Q4, Q5, and Q6, let `q[i]` be lane `i`'s signed code and let:

```text
u[i] = q[i] modulo 2^b
```

be its unsigned `b`-bit two's-complement word. Each consecutive lane pair occupies one base byte:

```text
base[j] = (u[2*j] & 0x0f) | ((u[2*j + 1] & 0x0f) << 4)
```

Thus the even lane is in the low nibble and the odd lane is in the high nibble. Every G64 group
occupies 32 base bytes.

For Q8, each lane occupies one byte containing its exact 8-bit two's-complement word. Lane `i`
occupies byte `i`, so every G32 group occupies 32 base bytes. The numeric-format restriction that
excludes code `-128` remains in force.

For Q2, four consecutive lanes occupy one base byte, lane `i` in bits `2 * (i mod 4)`:

```text
base[j] = u[4*j] | (u[4*j + 1] << 2) | (u[4*j + 2] << 4) | (u[4*j + 3] << 6)
```

so every G64 group occupies 16 base bytes.

The complete base plane is the concatenation of these per-group byte sequences in plane traversal
order.

### 3.4 High-bit plane

Only Q5 and Q6 have a high-bit plane. For each lane:

```text
high[i] = (u[i] >> 4) & ((1 << (b - 4)) - 1)
```

The high-bit stream is lane-major. Within one lane, bit 4 is emitted first, followed by bit 5 for
Q6. Stream bit `t` is stored in bit `(t mod 8)` of byte `floor(t / 8)`; bit zero is therefore the
first bit of every byte.

Equivalently:

- Q5 byte 0 contains the high bit of lanes 0 through 7 in byte bits 0 through 7;
- Q6 byte 0 contains lane 0 bits 4 and 5 in byte bits 0 and 1, lane 1 bits 4 and 5 in byte bits 2
  and 3, and so on.

One Q5 G64 group occupies 8 high bytes. One Q6 G64 group occupies 16 high bytes.

### 3.5 Scale plane

Every physical group owns one 16-bit scale word. Scale words follow the same row/group traversal as
the code planes and are stored little-endian:

```text
scale_index(row, group) = row * groups_per_row + group
```

For a logical group, the word is exactly the binary16 multiplier defined by its numeric format. The
padding rule in Section 3.1 defines the scale words for wholly physical groups.

### 3.6 Plane offsets and encoded size

Let:

```text
base_bytes  = N * groups_per_row * B
high_bytes  = N * groups_per_row * H
scale_bytes = N * groups_per_row * 2

base_offset  = 0
high_offset  = align_up(base_bytes, 256)
scale_offset = high_offset + align_up(high_bytes, 256)

payload_bytes = scale_offset + scale_bytes
```

The byte ranges are:

```text
base  = [base_offset,  base_offset  + base_bytes)
high  = [high_offset,  high_offset  + high_bytes)
scale = [scale_offset, scale_offset + scale_bytes)
```

When `H = 0`, `high_bytes = 0` and `scale_offset = high_offset`. Bytes between the end of the base
plane and `high_offset`, and between the end of a nonempty high plane and `scale_offset`, are zero.

The tensor object's JSON `bytes` must equal `payload_bytes`; it does not include the gap needed to
align the following object.

For example, a Q5 tensor with shape `[2,130]` has `K_pad=256`, four groups per row,
`base_bytes=256`, `high_bytes=64`, `scale_bytes=16`, `high_offset=256`, `scale_offset=512`, and
`payload_bytes=528`.

### 3.7 Row views and row slices

Row addressability is an intrinsic property of this layout. Define:

```text
base_row_bytes  = groups_per_row * B
high_row_bytes  = groups_per_row * H
scale_row_bytes = groups_per_row * 2
```

A non-owning logical view of consecutive rows `[row_begin, row_begin + row_count)` uses:

```text
base_view  = base_offset  + row_begin * base_row_bytes
high_view  = high_offset  + row_begin * high_row_bytes   # absent when H = 0
scale_view = scale_offset + row_begin * scale_row_bytes
```

and spans `row_count * base_row_bytes`, `row_count * high_row_bytes`, and
`row_count * scale_row_bytes` in the respective planes. Its logical shape is `[row_count,K]` and it
retains the parent object's `K_pad` and `groups_per_row`. A row view is therefore three plane spans,
not one assumed-contiguous payload range.

If those rows are materialized as a standalone payload, their three row spans are concatenated in
the same plane order, with the plane offsets and zero padding recomputed from Section 3.6 using
`N=row_count`. This produces another valid `row_split_k128_v1` tensor without decoding or repacking
individual codes.

## 4. `block_scale_k16_m128x4_v1`

This layout stores only rank-two `nvfp4` matrices `[N,K]` satisfying:

```text
N > 0
K > 0
N % 128 == 0
K % 64 == 0
```

It adds no logical padding. Let:

```text
code_plane_bytes      = N * K / 2
scale_plane_offset    = align_up(code_plane_bytes, 256)
scale_plane_bytes     = N * K / 16
weight_divisor_offset = scale_plane_offset + scale_plane_bytes
payload_bytes         = weight_divisor_offset + 4
```

The payload is a row-major E2M1 packed-code plane, zero padding to `scale_plane_offset`, a
swizzled E4M3FN scale plane, and the little-endian FP32 weight-divisor word. Within each packed code
byte, the low nibble is the smaller K coordinate and the high nibble is the next coordinate.

For logical row `n`, scale-group coordinate `g=floor(k/16)`, and `K_tiles=K/64`, define:

```text
row_tile   = floor(n / 128)
row_inner  = n % 128
scale_tile = floor(g / 4)
scale_lane = g % 4
```

The scale word's byte offset within the scale plane is:

```text
(row_tile * K_tiles + scale_tile) * 512
+ (row_inner % 32) * 16
+ floor(row_inner / 32) * 4
+ scale_lane
```

Layout decoding must recover the original packed E2M1 words, natural `[N,K/16]` E4M3FN scale-word
matrix, and exact divisor word. It never decodes and re-encodes either floating-point format.

## 5. `row_scale_v1`

`row_scale_v1` stores only rank-two `fp8_e4m3fn_row_bf16` matrices `[N,K]` with positive
dimensions. It adds no logical or physical matrix padding. Let:

```text
code_plane_bytes   = N * K
scale_plane_offset = align_up(code_plane_bytes, 256)
scale_plane_bytes  = N * 2
payload_bytes      = scale_plane_offset + scale_plane_bytes
```

The payload begins with one row-major E4M3FN byte per logical weight. Coordinate `[n,k]` is stored
at code-plane offset `n * K + k`. Zero bytes fill the interval from `code_plane_bytes` to
`scale_plane_offset`.

The scale plane contains one little-endian BF16 word per logical row in increasing `n` order. Scale
word `n` begins at `scale_plane_offset + 2 * n`. The layout neither converts the BF16 multiplier nor
combines it with its E4M3FN row. Code and scale validity and represented-weight reconstruction are
defined by `fp8_e4m3fn_row_bf16` in [`tensor-formats.md`](tensor-formats.md).

A logical row view consists of its `K` consecutive code bytes and its one BF16 scale word; those two
spans are not one assumed-contiguous payload range. A standalone consecutive slice or row gather is
encoded by concatenating the selected code rows, recomputing the scale-plane alignment for the new
row count, and appending the selected scale words in the same row order. It does not decode or
requantize either plane.

## 6. `nvfp4_expert_rg16_v1`

`nvfp4_expert_rg16_v1` stores one MoE layer's routed experts, `[E,H,I]`: E experts with hidden size
H and intermediate size I. Each expert has a gate and an up matrix `[I,H]` and a down matrix `[H,I]`
of `nvfp4_mul` words, plus one FP32 multiplier per matrix. The layout makes every expert one
contiguous, aligned record that CPU and GPU expert kernels read in place and that one DMA or one
`O_DIRECT` read moves (Qwen3.8-Flash-Next design §6.2).

```text
gate_up_row_groups = 2 * I / 16          gate_up_blocks = H / 16
down_row_groups    = H / 16              down_blocks    = I / 16
gate_up_bytes      = gate_up_row_groups * gate_up_blocks * 144
record_bytes       = gate_up_bytes + down_row_groups * down_blocks * 144
record_stride      = align_up(record_bytes, 4096)
multiplier_offset  = E * record_stride
payload_bytes      = multiplier_offset + E * 3 * 4
```

Record e begins at `e * record_stride`. Zero bytes pad each record to its stride.

**Gate/up matrix.** Its rows interleave the two projections: row `2i` is gate row `i` and row
`2i + 1` is up row `i`, for `i < I`. Rows form 16-row groups. Row group `rg` stores
`gate_up_blocks` units in increasing block order, so unit `(rg, b)` begins at record offset
`(rg * gate_up_blocks + b) * 144`. The down matrix follows at `gate_up_bytes`, with
`down_row_groups` row groups of `down_blocks` units in the same arrangement.

**Unit.** One 144-byte unit holds the 16 rows `16rg .. 16rg+15` of one 16-element K block
`16b .. 16b+15`:

- bytes `0..127` are four quads of 32 bytes. Byte `32q + 4j + t` (`q < 4`, `j < 8`, `t < 4`) holds
  the E2M1 code of row `16rg + j` at `k = 16b + 4q + t` in its low nibble, and the code of row
  `16rg + 8 + j` at the same k in its high nibble;
- bytes `128..143` hold the E4M3FN scale words of rows `16rg .. 16rg+15` for block b, in row order.

**Multiplier plane.** `E * 3` little-endian FP32 words follow the records: expert e's gate, up and
down multipliers at `multiplier_offset + 12e`, `+ 4` and `+ 8`. They are ModelOpt's
`weight_scale_2` words, copied unchanged.

The layout has no row views: a binding addresses the complete bank, and consumers obtain the record
and multiplier planes from `expert_bank_planes`. Activation input scales are separate model-role
tensors.

## 7. `ggml_rows_v1`

`ggml_rows_v1` stores a `ggml_*` matrix `[N,K]` as its N rows of `K / block_elems` whole GGUF
blocks, byte for byte as the GGUF stores them, contiguous and unpadded:

```text
row_bytes     = K / block_elems * block_bytes
payload_bytes = N * row_bytes
```

Row n begins at `n * row_bytes`. Row views and whole-row slices are byte ranges of whole rows.

### 7.1 `ggml_tiles16_v1`

`ggml_tiles16_v1` stores the same blocks as `ggml_rows_v1` (formats of 256-element blocks only)
in tiles of 16 rows, so that a tile's 16 rows of one block are one contiguous run that the narrow
kernels copy in 16-byte units (`ops/ggml/ggml_linear.cu`, tiled rows):

```text
tiles         = ceil(N / 16)
payload_bytes = tiles * 16 * row_bytes
block b of row n at   (n / 16) * 16 * row_bytes + (16 * b + n % 16) * block_bytes
```

Rows `N .. 16 * tiles - 1` are zero blocks (scale 0). Tile i begins at `16 * i * row_bytes`, where
row `16 i` begins in `ggml_rows_v1`, so a whole-row view or slice must start at a multiple of 16
rows and end at one or at `N`. Untiling (the inverse permutation) returns the `ggml_rows_v1` bytes
exactly; the token embedding (a row gather) is never tiled.

## 8. GGUF expert banks: `ggml_expert_iq4_nl_v1`, `ggml_expert_q2_0_v1`

These store one MoE layer's routed experts `[E,H,I]` of a GGUF import. The object's format is the
gate/up matrices' ggml type; the layout names the down matrices' (`ggml_iq4_nl` or `ggml_q2_0`).
Each expert is one record of its gate rows, its up rows (`[I,H]` each, rows of `H / block_elems`
blocks of the gate/up type) and its down rows (`[H,I]`, blocks of the down type), every row
verbatim GGUF blocks:

```text
gate_row      = H / block_elems(gate/up) * block_bytes(gate/up)
down_row      = I / block_elems(down) * block_bytes(down)
record_bytes  = 2 * I * gate_row + H * down_row
record_stride = align_up(record_bytes, 4096)
payload_bytes = E * record_stride
```

Record e begins at `e * record_stride`; the up rows at `I * gate_row` within it and the down rows at
`2 * I * gate_row`. Zero bytes pad each record to its stride. There is no tail plane.

## 9. `fp8_expert_b128_v1`

`fp8_expert_b128_v1` stores one MoE layer's routed experts `[E,H,I]` of `fp8_e4m3fn_b128` words
([Qwen3.8-Flash-Next design](qwen3_8-flash-next-design.md) §19.3.25). Each expert is one record: its gate rows, its up rows (`[I,H]`
each) and its down rows (`[H,I]`), one E4M3FN code byte per weight in row-major order, then the
scale plane: the gate's, the up's and the down's BF16 tile scales, each row-major
`[rows / 128][K / 128]` (little-endian words):

```text
tiles         = (I / 128) * (H / 128)        per matrix
scale_offset  = 3 * H * I
record_bytes  = scale_offset + 3 * tiles * 2
record_stride = align_up(record_bytes, 4096)
payload_bytes = E * record_stride
```

Record e begins at `e * record_stride`; up rows at `I * H` and down rows at `2 * I * H` within it.
Zero bytes pad each record to its stride. For the Flash-Next shape `[512, 2560, 640]` a record is
4,915,800 bytes in a 4,919,296-byte stride. Every row is addressed as `base + r * K`, as in the GGUF
banks, so expert routes read rows and their tile row in place.

## 10. `raw_bytes_v1`

`raw_bytes_v1` is a resource encoding, not a tensor layout. Its enclosing object payload is
the resource byte string itself:

```text
payload_bytes = resource_length
alignment     = 1
```

It has no embedded length, header, terminator, filename, character encoding, compression, or
trailing padding. The resource object's JSON `bytes` is its exact nonzero length, and a reader
returns the complete span unchanged. A model contract assigns a resource name and interprets those
bytes; the common encoding does not infer that meaning from the name.

## 11. Decode boundary

Layout decoding yields only persistent logical words:

- `contiguous_le_v1` yields the direct BF16, FP32, or I32 words in logical coordinate order;
- `row_split_k128_v1` yields the grouped signed codes and binary16 scales for logical columns
  `0..K-1`, discarding physical columns `K..K_pad-1`;
- `block_scale_k16_m128x4_v1` yields the packed E2M1 words, natural E4M3FN group-scale words, and
  matrix-level FP32 weight divisor;
- `row_scale_v1` yields the natural row-major E4M3FN code words and one BF16 multiplier per logical
  row;
- `nvfp4_expert_rg16_v1` yields, per expert, the gate, up and down E2M1 code words and E4M3FN
  scale words in natural `[rows, K]` order (undoing the unit packing and row interleave), and the
  three FP32 multipliers;
- `ggml_rows_v1` and the GGUF expert banks yield the verbatim GGUF block rows; `ggml_tiles16_v1`
  the same rows after untiling;
- `fp8_expert_b128_v1` yields, per expert, the gate, up and down E4M3FN code bytes in natural
  `[rows, K]` order and their BF16 tile scale words;
- `raw_bytes_v1` yields the enclosing resource bytes.

Dequantized values follow the reconstruction rule in `tensor-formats.md`. This document does
not select a quantization encoder, output dtype, accumulation dtype, kernel, runtime device layout,
or model consumer.

## 12. Logical views and native operands

Bindings address C-order logical element ranges of a parent object. The parent retains its full
geometry and backing allocation, so a view can locate code and scale planes using the original
matrix dimensions. Materialization uploads each required parent once and binds non-owning views.

[`weight_view.cpp`](../../src/core/weight_view.cpp) provides plane addressing and the bridge to
native operands. Direct tensors can use a contiguous element range. Grouped integer matrices can
use consecutive complete rows with unchanged K, using independent code, high-bit and scale pointers.

The current native `Weight` bridge requires a complete parent for FP8 and NVFP4. Their consumers
use the complete matrix geometry for plane addressing; a row slice cannot be passed as though its
payload were a newly packed smaller matrix. Adjacent logical projections can still share one
parent: when the chosen fused implementation consumes their complete union, it receives that
parent as one native weight.

Offline codecs can produce a standalone slice with its own plane offsets. The loader does not
perform that transformation. An execution implementation that accepts additional view forms must
consume the original parent geometry correctly.
