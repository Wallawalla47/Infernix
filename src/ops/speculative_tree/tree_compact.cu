// ninfer::ops - moving an accepted verification-tree path onto the main-chain columns.
#include "ninfer/ops/speculative_tree.h"

#include "core/device.h"
#include "core/paged_kv_storage.h"
#include "core/pdl.cuh"
#include "ninfer/types.h"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/kv_window.cuh"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr int kCompactThreads   = 256;
constexpr int kMaxCompactLayers = 64;
constexpr int kMaxCompactPlanes = 4;

// One block per (layer, active row). Each accepted path column c_i != i is copied onto column i;
// by the tree layout c_i > main depth >= i, so no copied column is also a destination.
__global__ __launch_bounds__(kCompactThreads) void tree_compact_columns_kernel(
    uint4* data, std::int32_t vectors_per_column, std::int32_t data_columns,
    std::int32_t rows_per_layer, const std::int32_t* row_outer, std::int32_t path_width,
    const std::int32_t* accepted_path, const std::int32_t* accepted_drafts) {
    pdl::enter();
    const int layer = static_cast<int>(blockIdx.x);
    const int row   = static_cast<int>(blockIdx.y);
    const int count = accepted_drafts[row];
    const int outer = layer * rows_per_layer + (row_outer != nullptr ? row_outer[row] : row);
    const std::int64_t base = static_cast<std::int64_t>(outer) * data_columns *
                              static_cast<std::int64_t>(vectors_per_column);
    for (int i = 1; i <= count; ++i) {
        const int column = accepted_path[row * path_width + i];
        if (column == i || column < 0) { continue; }
        const uint4* source = data + base + static_cast<std::int64_t>(column) * vectors_per_column;
        uint4* destination  = data + base + static_cast<std::int64_t>(i) * vectors_per_column;
        for (int v = static_cast<int>(threadIdx.x); v < vectors_per_column; v += kCompactThreads) {
            destination[v] = source[v];
        }
    }
}

// The stored planes of each paged K/V layer. Plane p holds plane_bytes[p] per (token, head) and is
// addressed like every paged plane, as bytes * (page_size * (head + heads * page) + offset).
struct KvCompactArgs {
    unsigned char* planes[kMaxCompactLayers][kMaxCompactPlanes];
    std::int32_t plane_bytes[kMaxCompactPlanes];
    const std::int32_t* tables;
    std::int32_t table_stride;
    std::int32_t kv_heads;
    std::int32_t plane_count;
    std::int32_t width;
    // Exact recent-key window of the vector-quantized formats, whose planes 0/1 are the K/V
    // codes and 2/3 their FP16 row scales: window codes and scales per (layer, role) and tags per
    // layer, of every execution row.
    bool window;
    std::int8_t* window_codes[kMaxCompactLayers][2];
    __half* window_scales[kMaxCompactLayers][2];
    std::int32_t* window_tags[kMaxCompactLayers];
    const std::int32_t* window_slots; // per verification row (null: slot 0)
};

template <class Word>
__device__ __forceinline__ void copy_words(unsigned char* destination, const unsigned char* source,
                                           int words, int thread, int threads) {
    for (int w = thread; w < words; w += threads) {
        reinterpret_cast<Word*>(destination)[w] = reinterpret_cast<const Word*>(source)[w];
    }
}

// Copies one (token, head) element of every plane. Each plane is moved in the widest word its
// per-token byte count admits; plane bases are 256-byte aligned.
__device__ __forceinline__ void copy_position(const KvCompactArgs& args, int layer, int source_page,
                                              int source_offset, int destination_page,
                                              int destination_offset) {
    const int tid   = static_cast<int>(threadIdx.x);
    const int heads = args.kv_heads;
    for (int plane = 0; plane < args.plane_count; ++plane) {
        const std::int32_t bytes = args.plane_bytes[plane];
        unsigned char* data      = args.planes[layer][plane];
        for (int head = 0; head < heads; ++head) {
            const auto at = [&](int page, int offset) {
                return static_cast<std::int64_t>(bytes) *
                       (static_cast<std::int64_t>(kPagedKVPageSize) *
                            (head + static_cast<std::int64_t>(heads) * page) +
                        offset);
            };
            const unsigned char* source = data + at(source_page, source_offset);
            unsigned char* destination  = data + at(destination_page, destination_offset);
            if (bytes % 16 == 0) {
                copy_words<uint4>(destination, source, bytes / 16, tid, kCompactThreads);
            } else if (bytes % 8 == 0) {
                copy_words<uint2>(destination, source, bytes / 8, tid, kCompactThreads);
            } else if (bytes % 4 == 0) {
                copy_words<std::uint32_t>(destination, source, bytes / 4, tid, kCompactThreads);
            } else if (bytes % 2 == 0) {
                copy_words<std::uint16_t>(destination, source, bytes / 2, tid, kCompactThreads);
            } else {
                copy_words<unsigned char>(destination, source, bytes, tid, kCompactThreads);
            }
        }
    }
}

inline constexpr int kMaxCompactWindowHeads = 8;

// Moves the window slot of a compacted position: the source position's slot becomes the
// destination's, re-tagged for the destination position, when it still matched the source codes;
// otherwise the destination slot's tags are cleared and its readers fall back to the codes.
__device__ void move_window_slot(const KvCompactArgs& args, int layer, int row, int source_page,
                                 int source_position, int destination_position,
                                 std::int32_t* tags) {
    const int tid         = static_cast<int>(threadIdx.x);
    const int heads       = args.kv_heads;
    const int source      = kv_window_slot(source_position);
    const int destination = kv_window_slot(destination_position);
    const auto slot_index = [&](int head, int slot) {
        return (static_cast<std::int64_t>(row) * heads + head) * kKVWindowSlots + slot;
    };
    __syncthreads();
    if (tid < 2 * heads) {
        const int head           = tid >> 1;
        const int role           = tid & 1;
        const std::int32_t bytes = args.plane_bytes[role];
        const std::int64_t element =
            static_cast<std::int64_t>(kPagedKVPageSize) *
                (head + static_cast<std::int64_t>(heads) * source_page) +
            (source_position & kPagedKVPageMask);
        const auto* codes =
            reinterpret_cast<const std::uint32_t*>(args.planes[layer][role] + bytes * element);
        const std::uint16_t scale_bits =
            reinterpret_cast<const std::uint16_t*>(args.planes[layer][2 + role])[element];
        const auto tag_of = [&](int position) {
            std::uint32_t hash = kv_window_tag_begin(position);
            for (int w = 0; w < bytes / 4; ++w) hash = kv_window_tag_step(hash, codes[w]);
            return kv_window_tag_finish(hash, scale_bits);
        };
        const bool valid = static_cast<std::uint32_t>(
                               args.window_tags[layer][slot_index(head, source) * 2 + role]) ==
                           tag_of(source_position);
        tags[tid] = valid ? static_cast<std::int32_t>(tag_of(destination_position)) : 0;
    }
    __syncthreads();
    for (int item = tid; item < 2 * heads * 17; item += kCompactThreads) {
        const int unit = item / 17;
        const int part = item - unit * 17;
        const int head = unit >> 1;
        const int role = unit & 1;
        if (tags[unit] == 0) { continue; }
        if (part < 16) {
            const std::int8_t* from =
                args.window_codes[layer][role] + slot_index(head, source) * 256 + part * 16;
            std::int8_t* to =
                args.window_codes[layer][role] + slot_index(head, destination) * 256 + part * 16;
            *reinterpret_cast<uint4*>(to) = *reinterpret_cast<const uint4*>(from);
        } else {
            const __half* from = args.window_scales[layer][role] + slot_index(head, source) * 4;
            __half* to = args.window_scales[layer][role] + slot_index(head, destination) * 4;
            *reinterpret_cast<uint2*>(to) = *reinterpret_cast<const uint2*>(from);
        }
    }
    __syncthreads();
    if (tid < 2 * heads) {
        args.window_tags[layer][slot_index(tid >> 1, destination) * 2 + (tid & 1)] = tags[tid];
    }
}

// One block per (layer, row). Sources (columns after the main chain) are never destinations.
__global__ __launch_bounds__(kCompactThreads) void tree_compact_kv_kernel(
    const __grid_constant__ KvCompactArgs args, const std::int32_t* verify_positions,
    const std::int32_t* table_rows, const std::int32_t* accepted_path,
    const std::int32_t* accepted_drafts) {
    pdl::enter();
    const int layer = static_cast<int>(blockIdx.x);
    const int row   = static_cast<int>(blockIdx.y);
    const int count = accepted_drafts[row];
    const std::int32_t* table =
        args.tables + static_cast<std::int64_t>(table_rows[row]) * args.table_stride;
    // Column 0 of a verification row sits at its base position F.
    const int base = verify_positions[row * args.width];
    __shared__ std::int32_t window_tag[2 * kMaxCompactWindowHeads];
    for (int i = 1; i <= count; ++i) {
        const int column = accepted_path[row * args.width + i];
        if (column == i || column < 0) { continue; }
        const int source_position      = base + column;
        const int destination_position = base + i;
        const int source_page          = table[source_position >> kPagedKVPageShift];
        copy_position(args, layer, source_page, source_position & kPagedKVPageMask,
                      table[destination_position >> kPagedKVPageShift],
                      destination_position & kPagedKVPageMask);
        if (args.window) {
            move_window_slot(args, layer,
                             args.window_slots == nullptr ? 0 : args.window_slots[row],
                             source_page, source_position, destination_position, window_tag);
        }
    }
}

void require_i32(const Tensor& tensor, std::int32_t n0, std::int32_t n1, const char* name) {
    if (tensor.dtype != DType::I32 || tensor.ne[0] != n0 || tensor.ne[1] != n1 ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        tensor.data == nullptr) {
        throw std::invalid_argument(std::string("speculative tree compaction: invalid ") + name);
    }
}

} // namespace

void speculative_tree_compact_columns(Tensor& data, std::int32_t rows_per_layer,
                                      const Tensor& row_outer, const Tensor& accepted_path,
                                      const Tensor& accepted_drafts, cudaStream_t stream) {
    const std::int32_t data_columns = data.ne[1];
    const std::int64_t outer        = static_cast<std::int64_t>(data.ne[2]) * data.ne[3];
    const std::int32_t rows         = accepted_drafts.ne[0];
    const std::int32_t path_width   = accepted_path.ne[0];
    const std::size_t column_bytes  = static_cast<std::size_t>(data.ne[0]) * dtype_size(data.dtype);
    if (data.data == nullptr || !data.is_contiguous() || path_width < 2 ||
        data_columns < path_width || rows < 1 || rows_per_layer < rows ||
        outer % rows_per_layer != 0 || column_bytes % 16 != 0 ||
        (reinterpret_cast<std::uintptr_t>(data.data) & 15U) != 0 ||
        outer / rows_per_layer > 65535) {
        throw std::invalid_argument("speculative_tree_compact_columns: invalid data plane");
    }
    require_i32(accepted_path, path_width, rows, "accepted_path");
    require_i32(accepted_drafts, rows, 1, "accepted_drafts");
    if (row_outer.data != nullptr) { require_i32(row_outer, rows, 1, "row_outer"); }
    const dim3 grid(static_cast<unsigned int>(outer / rows_per_layer),
                    static_cast<unsigned int>(rows));
    CUDA_CHECK(pdl::launch_consumer(
        {grid, dim3(kCompactThreads), 0, stream}, tree_compact_columns_kernel,
        static_cast<uint4*>(data.data), static_cast<std::int32_t>(column_bytes / 16), data_columns,
        rows_per_layer, static_cast<const std::int32_t*>(row_outer.data), path_width,
        static_cast<const std::int32_t*>(accepted_path.data),
        static_cast<const std::int32_t*>(accepted_drafts.data)));
    CUDA_CHECK(cudaGetLastError());
}

void speculative_tree_compact_kv(const PagedKVBatchLayerView* layers, std::int32_t layer_count,
                                 const Tensor& verify_positions, const Tensor& kv_table_rows,
                                 const Tensor& accepted_path, const Tensor& accepted_drafts,
                                 cudaStream_t stream) {
    if (layers == nullptr || layer_count < 1 || layer_count > kMaxCompactLayers) {
        throw std::invalid_argument("speculative_tree_compact_kv: invalid layer list");
    }
    const std::int32_t rows  = accepted_drafts.ne[0];
    const std::int32_t width = accepted_path.ne[0];
    require_i32(accepted_drafts, rows, 1, "accepted_drafts");
    require_i32(accepted_path, width, rows, "accepted_path");
    require_i32(verify_positions, width, rows, "verify_positions");
    require_i32(kv_table_rows, rows, 1, "kv_table_rows");
    const PagedKVBatchLayerView& first = layers[0];
    if (first.num_kv_heads < 1 || first.head_dim < 1 || first.block_tables.data == nullptr) {
        throw std::invalid_argument("speculative_tree_compact_kv: invalid layer geometry");
    }
    const PagedKVStorageLayout layout = paged_kv_storage_layout(first.storage, first.head_dim);
    const std::int32_t key_bytes   = layout.key.data_leading_extent *
                                     static_cast<std::int32_t>(dtype_size(layout.key.data_dtype));
    const std::int32_t value_bytes = layout.value.data_leading_extent *
                                     static_cast<std::int32_t>(dtype_size(layout.value.data_dtype));
    const std::int32_t key_scale_bytes =
        layout.key.scale_leading_extent *
        static_cast<std::int32_t>(dtype_size(layout.key.scale_dtype));
    const std::int32_t value_scale_bytes =
        layout.value.scale_leading_extent *
        static_cast<std::int32_t>(dtype_size(layout.value.scale_dtype));
    KvCompactArgs args{};
    args.tables       = static_cast<const std::int32_t*>(first.block_tables.data);
    args.table_stride = first.block_tables.ne[0];
    args.kv_heads     = first.num_kv_heads;
    args.width        = width;
    args.plane_count  = 0;
    for (int layer = 0; layer < layer_count; ++layer) {
        const PagedKVBatchLayerView& view = layers[layer];
        if (view.storage != first.storage || view.num_kv_heads != first.num_kv_heads ||
            view.head_dim != first.head_dim || view.block_tables.data != first.block_tables.data ||
            view.k_pages.data == nullptr || view.v_pages.data == nullptr ||
            (layout.key.has_scale() && view.k_scale_pages.data == nullptr) ||
            (layout.value.has_scale() && view.v_scale_pages.data == nullptr)) {
            throw std::invalid_argument(
                "speculative_tree_compact_kv: layers must share one table and storage format");
        }
        int planes     = 0;
        const auto add = [&](const Tensor& pages, std::int32_t bytes) {
            if (bytes == 0) { return; }
            args.plane_bytes[planes]     = bytes;
            args.planes[layer][planes++] = static_cast<unsigned char*>(pages.data);
        };
        add(view.k_pages, key_bytes);
        add(view.v_pages, value_bytes);
        add(view.k_scale_pages, key_scale_bytes);
        add(view.v_scale_pages, value_scale_bytes);
        args.plane_count = planes;
        if (view.window.present()) {
            args.window                  = true;
            args.window_codes[layer][0]  = static_cast<std::int8_t*>(view.window.k_codes.data);
            args.window_codes[layer][1]  = static_cast<std::int8_t*>(view.window.v_codes.data);
            args.window_scales[layer][0] = static_cast<__half*>(view.window.k_scales.data);
            args.window_scales[layer][1] = static_cast<__half*>(view.window.v_scales.data);
            args.window_tags[layer]      = static_cast<std::int32_t*>(view.window.tags.data);
            args.window_slots = static_cast<const std::int32_t*>(view.window.slots.data);
        }
    }
    if (args.window && (!kv_storage_has_exact_window(first.storage) ||
                        first.num_kv_heads > kMaxCompactWindowHeads)) {
        throw std::invalid_argument("speculative_tree_compact_kv: unexpected KV window");
    }
    CUDA_CHECK(pdl::launch_consumer(
        {dim3(static_cast<unsigned int>(layer_count), static_cast<unsigned int>(rows)),
         dim3(kCompactThreads), 0, stream},
        tree_compact_kv_kernel, args, static_cast<const std::int32_t*>(verify_positions.data),
        static_cast<const std::int32_t*>(kv_table_rows.data),
        static_cast<const std::int32_t*>(accepted_path.data),
        static_cast<const std::int32_t*>(accepted_drafts.data)));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
