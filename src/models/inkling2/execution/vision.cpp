#include "models/inkling2/execution/vision.h"

#include "core/device.h"
#include "core/tensor.h"
#include "infernix/ops/add_bias.h"
#include "infernix/ops/gelu.h"
#include "infernix/ops/inkling2_vision.h"
#include "infernix/ops/layer_norm.h"
#include "infernix/ops/linear.h"
#include "infernix/ops/softmax_attention.h"
#include "infernix/ops/vision_pos_embed.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace infernix::models::inkling2 {
namespace {

constexpr std::uint64_t kAlign      = 256;
constexpr std::size_t kArenaBytes   = 256; // the single-segment attention's (empty) workspace

std::uint64_t aligned(std::uint64_t bytes) { return (bytes + kAlign - 1) / kAlign * kAlign; }

Tensor matrix(const void* data, std::int32_t rows_fast, std::int32_t columns) {
    return Tensor(const_cast<void*>(data), DType::BF16, {rows_fast, columns});
}

Tensor vector(const std::uint16_t* data, std::int32_t length) {
    return Tensor(const_cast<std::uint16_t*>(data), DType::BF16, {length});
}

// One axis's two bilinear taps (Transformers' get_vision_interpolation_indices_and_weights, align_corners=False,
// border clamping): target index i of n over a side-long table. The weights are FP32 of the FP64 values.
void taps(std::int32_t i, std::int32_t n, std::int32_t side, std::int32_t (&index)[2], float (&weight)[2]) {
    const double src   = (i + 0.5) * side / n - 0.5;
    const double floor = std::floor(src);
    for (int t = 0; t < 2; ++t) {
        const double raw = floor + t;
        index[t]         = static_cast<std::int32_t>(std::clamp(raw, 0.0, static_cast<double>(side - 1)));
        weight[t]        = static_cast<float>(std::max(0.0, 1.0 - std::abs(src - raw)));
    }
}

// The pinned block holding the offloaded tower: its host address (linear operands name it there) and its device-mapped
// address (direct parameters name it there: execution/parameters.cpp reads every direct weight where device code
// reads it, which under WDDM may differ).
struct PinnedBase {
    const std::byte* host   = nullptr;
    const std::byte* device = nullptr;
};

// --vision-offload on: a pointer inside a group's pinned objects (at either address of the block), at its place in the
// group's device region.
const std::byte* staged(const VisionStagingGroup& group, std::byte* region, const PinnedBase& block, const void* p) {
    if (p == nullptr) { return nullptr; }
    const auto* q = static_cast<const std::byte*>(p);
    for (const std::byte* base : {block.host, block.device}) {
        if (base == nullptr || q < base) { continue; }
        const auto at = static_cast<std::size_t>(q - base);
        for (const VisionStagingSegment& s : group.segments) {
            if (at >= s.pinned_offset && at < s.pinned_offset + s.bytes) {
                return region + s.staging_offset + (at - s.pinned_offset);
            }
        }
    }
    throw std::logic_error("Inkling 2 vision: an offloaded weight lies outside its staged group");
}
const std::uint16_t* staged(const VisionStagingGroup& group, std::byte* region, const PinnedBase& block,
                            const std::uint16_t* p) {
    return reinterpret_cast<const std::uint16_t*>(staged(group, region, block, static_cast<const void*>(p)));
}
Weight staged(const VisionStagingGroup& group, std::byte* region, const PinnedBase& block, const Weight& w) {
    Weight out  = w;
    out.payload = staged(group, region, block, w.payload);
    out.qdata   = staged(group, region, block, w.qdata);
    out.qhigh   = staged(group, region, block, w.qhigh);
    out.scales  = staged(group, region, block, w.scales);
    out.escapes = staged(group, region, block, w.escapes);
    return out;
}

// A group's pinned objects copied into its device region on `stream`.
void upload_group(const VisionStagingGroup& group, std::byte* region, const std::byte* block, cudaStream_t stream) {
    for (const VisionStagingSegment& s : group.segments) {
        CUDA_CHECK(cudaMemcpyAsync(region + s.staging_offset, block + s.pinned_offset, s.bytes, cudaMemcpyHostToDevice,
                                   stream));
    }
}

std::uint64_t staging_bytes(const VisionHostWeights& host) {
    return aligned(host.resident.bytes) + 2 * aligned(host.slot_bytes);
}

} // namespace

struct VisionTower::Layout {
    std::uint64_t x = 0, qkv = 0, att = 0, h = 0, o = 0, u = 0, patches = 0, positions = 0, index = 0, weight = 0,
                  arena = 0, total = 0;

    Layout(const VisionConfig& c, std::uint32_t max_cells) {
        const std::uint64_t T = 1 + std::uint64_t{c.merge} * c.merge * max_cells;
        const std::uint64_t C = std::min<std::uint64_t>(kVisionChunk, T);
        const std::uint64_t H = c.hidden_size, wide = std::max(c.intermediate, c.projection_hidden);
        std::uint64_t at      = 0;
        const auto take = [&](std::uint64_t bytes) {
            const std::uint64_t here = at;
            at += aligned(bytes);
            return here;
        };
        x         = take(T * H * 2);
        qkv       = take(std::max(T * 3 * H, std::uint64_t{max_cells} * c.output_size) * 2); // then the output rows
        att       = take(T * H * 2);
        h         = take(C * H * 2);
        o         = take(C * H * 2);
        u         = take(C * wide * 2);
        patches   = take(C * c.patch_width() * 2);
        positions = take(T * 2 * 4);
        index     = take(4 * C * 4);
        weight    = take(4 * C * 4);
        arena     = take(kArenaBytes);
        total     = at;
    }
};

std::uint64_t VisionTower::device_bytes(const VisionConfig& config, std::uint32_t max_cells,
                                        const VisionHostWeights* host) {
    return Layout(config, max_cells).total + (host != nullptr ? staging_bytes(*host) : 0);
}

VisionTower::VisionTower(const VisionConfig& config, const VisionOperands& operands, std::uint32_t max_cells,
                         const VisionHostWeights* host, cudaStream_t transfer)
    : c_(config), w_(&operands), max_cells_(max_cells), host_(host), transfer_(transfer) {
    if (max_cells == 0 || c_.head_dim() != 64 || c_.heads != 16 || operands.layers.size() != c_.layers) {
        throw std::invalid_argument("Inkling 2 vision: a tower of 16 heads of 64 and at least one cell an image");
    }
    if (host_ != nullptr) {
        if (host_->block == nullptr || host_->layers.size() != c_.layers || transfer_ == nullptr) {
            throw std::invalid_argument("Inkling 2 vision: an offloaded tower needs its pinned groups and a transfer stream");
        }
        staging_          = DeviceBuffer(staging_bytes(*host_));
        auto* resident    = static_cast<std::byte*>(staging_.p);
        slot_[0]          = resident + aligned(host_->resident.bytes);
        slot_[1]          = slot_[0] + aligned(host_->slot_bytes);
        const PinnedBase b{host_->block, host_->block_device};
        const VisionStagingGroup& r = host_->resident;
        staged_                 = operands;
        staged_.patch_embedding = staged(r, resident, b, operands.patch_embedding);
        staged_.position_table  = staged(r, resident, b, operands.position_table);
        staged_.class_row       = staged(r, resident, b, operands.class_row);
        staged_.pre_norm        = staged(r, resident, b, operands.pre_norm);
        staged_.pre_norm_bias   = staged(r, resident, b, operands.pre_norm_bias);
        staged_.projector_in    = staged(r, resident, b, operands.projector_in);
        staged_.projector_out   = staged(r, resident, b, operands.projector_out);
        for (std::size_t i = 0; i < operands.layers.size(); ++i) {
            const VisionStagingGroup& g = host_->layers[i];
            std::byte* slot             = slot_[i % 2];
            const VisionLayerOperands& s = operands.layers[i];
            VisionLayerOperands& d       = staged_.layers[i];
            d.attention_norm      = staged(g, slot, b, s.attention_norm);
            d.attention_norm_bias = staged(g, slot, b, s.attention_norm_bias);
            d.qkv                 = staged(g, slot, b, s.qkv);
            d.qkv_bias            = staged(g, slot, b, s.qkv_bias);
            d.output              = staged(g, slot, b, s.output);
            d.output_bias         = staged(g, slot, b, s.output_bias);
            d.attention_scale     = staged(g, slot, b, s.attention_scale);
            d.mlp_norm            = staged(g, slot, b, s.mlp_norm);
            d.mlp_norm_bias       = staged(g, slot, b, s.mlp_norm_bias);
            d.up                  = staged(g, slot, b, s.up);
            d.up_bias             = staged(g, slot, b, s.up_bias);
            d.down                = staged(g, slot, b, s.down);
            d.down_bias           = staged(g, slot, b, s.down_bias);
            d.mlp_scale           = staged(g, slot, b, s.mlp_scale);
        }
        w_ = &staged_;
        for (cudaEvent_t* e : {&uploaded_[0], &uploaded_[1], &fence_}) {
            CUDA_CHECK(cudaEventCreateWithFlags(e, cudaEventDisableTiming));
        }
        // The resident region once: the patch embedding, the position table, the class row, the pre-norm, the projector.
        upload_group(r, resident, b.host, transfer_);
        CUDA_CHECK(cudaStreamSynchronize(transfer_));
    }
    const Layout l(c_, max_cells);
    buffer_     = DeviceBuffer(l.total);
    auto* base  = static_cast<std::uint8_t*>(buffer_.p);
    x_          = reinterpret_cast<std::uint16_t*>(base + l.x);
    qkv_        = reinterpret_cast<std::uint16_t*>(base + l.qkv);
    att_        = reinterpret_cast<std::uint16_t*>(base + l.att);
    h_          = reinterpret_cast<std::uint16_t*>(base + l.h);
    o_          = reinterpret_cast<std::uint16_t*>(base + l.o);
    u_          = reinterpret_cast<std::uint16_t*>(base + l.u);
    patches_    = reinterpret_cast<std::uint16_t*>(base + l.patches);
    positions_  = reinterpret_cast<std::int32_t*>(base + l.positions);
    tap_index_  = reinterpret_cast<std::int32_t*>(base + l.index);
    tap_weight_ = reinterpret_cast<float*>(base + l.weight);
    arena_      = base + l.arena;
}

VisionTower::~VisionTower() {
    if (host_ == nullptr) { return; }
    (void)cudaStreamSynchronize(transfer_); // no upload still lands in the slots
    for (cudaEvent_t e : {uploaded_[0], uploaded_[1], fence_}) {
        if (e != nullptr) { (void)cudaEventDestroy(e); }
    }
}

// Layer `layer`'s pinned objects into its slot on the transfer stream (which already waits for the slot's last reader).
void VisionTower::upload_layer(std::uint32_t layer) {
    const VisionStagingGroup& g = host_->layers[layer];
    upload_group(g, slot_[layer % 2], host_->block, transfer_);
    CUDA_CHECK(cudaEventRecord(uploaded_[layer % 2], transfer_));
    streamed_bytes_ += g.bytes;
    next_upload_ = layer + 1;
}

const std::uint16_t* VisionTower::encode(const VisionImage& image, cudaStream_t stream) {
    const std::int32_t m = static_cast<std::int32_t>(c_.merge), H = static_cast<std::int32_t>(c_.hidden_size);
    const std::int32_t K = static_cast<std::int32_t>(c_.patch_width());
    const std::int32_t I = static_cast<std::int32_t>(c_.intermediate);
    const std::int32_t cells = image.cells();
    if (image.rows <= 0 || image.cols <= 0 || static_cast<std::uint32_t>(cells) > max_cells_ ||
        image.patches.size() != static_cast<std::size_t>(cells) * m * m * K) {
        throw std::invalid_argument("Inkling 2 vision: an image of " + std::to_string(image.rows) + " x " +
                                    std::to_string(image.cols) + " cells is outside the tower's " +
                                    std::to_string(max_cells_) + " cells or its patches disagree");
    }
    const std::int32_t P = cells * m * m, T = 1 + P, gh = image.rows * m, gw = image.cols * m;
    const std::int32_t grid = static_cast<std::int32_t>(c_.position_grid);

    // Rotary positions: the class token (0, 0), patch (gy, gx) at (gy + 1, gx + 1); merge-major patch order.
    host_positions_.assign(static_cast<std::size_t>(T) * 2, 0);
    const auto patch_yx = [&](std::int32_t p, std::int32_t& gy, std::int32_t& gx) {
        const std::int32_t cell = p / (m * m), within = p % (m * m);
        gy = (cell / image.cols) * m + within / m;
        gx = (cell % image.cols) * m + within % m;
    };
    for (std::int32_t p = 0; p < P; ++p) {
        std::int32_t gy = 0, gx = 0;
        patch_yx(p, gy, gx);
        host_positions_[2 * (p + 1)]     = gy + 1;
        host_positions_[2 * (p + 1) + 1] = gx + 1;
    }
    CUDA_CHECK(cudaMemcpyAsync(positions_, host_positions_.data(), host_positions_.size() * 4, cudaMemcpyHostToDevice,
                               stream));

    // Patch embedding plus the resized position table, a chunk at a time, into token columns 1..P.
    const VisionOperands& w = *w_;
    const Tensor table = matrix(w.position_table, H, grid * grid); // [H, grid^2]: row gy * grid + gx
    for (std::int32_t first = 0; first < P; first += kVisionChunk) {
        const std::int32_t n = std::min(kVisionChunk, P - first);
        CUDA_CHECK(cudaMemcpyAsync(patches_, image.patches.data() + static_cast<std::size_t>(first) * K,
                                   static_cast<std::size_t>(n) * K * 2, cudaMemcpyHostToDevice, stream));
        Tensor in  = matrix(patches_, K, n);
        Tensor out = matrix(x_ + static_cast<std::size_t>(1 + first) * H, H, n);
        ops::linear(in, w.patch_embedding, out, stream);
        host_index_.assign(static_cast<std::size_t>(4) * n, 0);
        host_weight_.assign(static_cast<std::size_t>(4) * n, 0.0F);
        for (std::int32_t j = 0; j < n; ++j) {
            std::int32_t gy = 0, gx = 0;
            patch_yx(first + j, gy, gx);
            std::int32_t iy[2], ix[2];
            float wy[2], wx[2];
            taps(gy, gh, grid, iy, wy);
            taps(gx, gw, grid, ix, wx);
            for (int a = 0; a < 2; ++a) {
                for (int b = 0; b < 2; ++b) {
                    host_index_[static_cast<std::size_t>(j) * 4 + 2 * a + b]  = iy[a] * grid + ix[b];
                    host_weight_[static_cast<std::size_t>(j) * 4 + 2 * a + b] = wy[a] * wx[b];
                }
            }
        }
        CUDA_CHECK(cudaMemcpyAsync(tap_index_, host_index_.data(), host_index_.size() * 4, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(tap_weight_, host_weight_.data(), host_weight_.size() * 4, cudaMemcpyHostToDevice,
                                   stream));
        Tensor index(tap_index_, DType::I32, {4, n}); // a patch's four corners consecutive
        Tensor weight(tap_weight_, DType::FP32, {4, n});
        ops::vision_pos_embed_add(table, index, weight, out, stream);
    }
    CUDA_CHECK(cudaMemcpyAsync(x_, w.class_row, static_cast<std::size_t>(H) * 2, cudaMemcpyDeviceToDevice, stream));

    const auto chunks = [&](auto&& body) {
        for (std::int32_t first = 0; first < T; first += kVisionChunk) { body(first, std::min(kVisionChunk, T - first)); }
    };
    const auto norm = [&](const std::uint16_t* weight, const std::uint16_t* bias, std::int32_t first, std::int32_t n) {
        Tensor in  = matrix(x_ + static_cast<std::size_t>(first) * H, H, n);
        Tensor out = matrix(h_, H, n);
        ops::layer_norm(in, vector(weight, H), vector(bias, H), c_.layer_norm_eps, out, stream);
    };
    // The pre-norm: the residual stream is its output.
    chunks([&](std::int32_t first, std::int32_t n) {
        norm(w.pre_norm, w.pre_norm_bias, first, n);
        CUDA_CHECK(cudaMemcpyAsync(x_ + static_cast<std::size_t>(first) * H, h_, static_cast<std::size_t>(n) * H * 2,
                                   cudaMemcpyDeviceToDevice, stream));
    });

    const std::int32_t D = static_cast<std::int32_t>(c_.head_dim()), heads = static_cast<std::int32_t>(c_.heads);
    DeviceArena arena(DeviceSpan{arena_, kArenaBytes});
    // An offloaded tower: layers 0 and 1 into their slots once everything issued so far (the previous image's layers)
    // is past; then each layer waits for its slot, and refills the slot the layer before it read.
    const auto layer_count = static_cast<std::uint32_t>(w.layers.size());
    if (host_ != nullptr) {
        streamed_bytes_ = 0;
        CUDA_CHECK(cudaEventRecord(fence_, stream));
        CUDA_CHECK(cudaStreamWaitEvent(transfer_, fence_, 0));
        upload_layer(0);
        if (layer_count > 1) { upload_layer(1); }
    }
    for (std::uint32_t li = 0; li < layer_count; ++li) {
        const VisionLayerOperands& l = w.layers[li];
        if (host_ != nullptr) {
            CUDA_CHECK(cudaStreamWaitEvent(stream, uploaded_[li % 2], 0));
            if (li >= 1 && next_upload_ == li + 1 && next_upload_ < layer_count) {
                CUDA_CHECK(cudaEventRecord(fence_, stream)); // past layer li - 1, the slot's last reader
                CUDA_CHECK(cudaStreamWaitEvent(transfer_, fence_, 0));
                upload_layer(li + 1);
            }
        }
        chunks([&](std::int32_t first, std::int32_t n) {
            norm(l.attention_norm, l.attention_norm_bias, first, n);
            Tensor out = matrix(qkv_ + static_cast<std::size_t>(first) * 3 * H, 3 * H, n);
            ops::linear(matrix(h_, H, n), l.qkv, out, stream);
        });
        ops::inkling2::vision_qkv_rotary({qkv_, l.qkv_bias, positions_, T, heads, D, c_.rope_theta}, stream);
        Tensor q(qkv_, DType::BF16, {D, heads, T});
        Tensor k(qkv_ + H, DType::BF16, {D, heads, T});
        Tensor v(qkv_ + 2 * H, DType::BF16, {D, heads, T});
        q.nb[2] = k.nb[2] = v.nb[2] = static_cast<std::int64_t>(3) * H * 2;
        Tensor attended(att_, DType::BF16, {D, heads, T});
        ops::softmax_attention(q, k, v, {D, heads, heads}, static_cast<float>(1.0 / std::sqrt(static_cast<double>(D))),
                               arena, attended, stream);
        chunks([&](std::int32_t first, std::int32_t n) {
            Tensor out = matrix(o_, H, n);
            ops::linear(matrix(att_ + static_cast<std::size_t>(first) * H, H, n), l.output, out, stream);
            ops::inkling2::vision_scaled_residual(
                {x_ + static_cast<std::size_t>(first) * H, o_, l.output_bias, l.attention_scale, n, H}, stream);
        });
        chunks([&](std::int32_t first, std::int32_t n) {
            norm(l.mlp_norm, l.mlp_norm_bias, first, n);
            Tensor up = matrix(u_, I, n);
            ops::linear(matrix(h_, H, n), l.up, up, stream);
            ops::add_bias(vector(l.up_bias, I), up, stream);
            ops::gelu(up, ops::GeluMode::Exact, stream);
            Tensor down = matrix(o_, H, n);
            ops::linear(up, l.down, down, stream);
            ops::inkling2::vision_scaled_residual(
                {x_ + static_cast<std::size_t>(first) * H, o_, l.down_bias, l.mlp_scale, n, H}, stream);
        });
    }

    // The projector: each cell's merge^2 consecutive patch rows as one [merge^2 H] row; the output rows into the QKV
    // region (free after the last layer).
    const std::int32_t merged = m * m * H, hidden = static_cast<std::int32_t>(c_.projection_hidden);
    const std::int32_t out_width = static_cast<std::int32_t>(c_.output_size), step = kVisionChunk / (m * m);
    for (std::int32_t first = 0; first < cells; first += step) {
        const std::int32_t n = std::min(step, cells - first);
        Tensor in  = matrix(x_ + static_cast<std::size_t>(1 + first * m * m) * H, merged, n);
        Tensor mid = matrix(u_, hidden, n);
        ops::linear(in, w.projector_in, mid, stream);
        ops::gelu(mid, ops::GeluMode::Exact, stream);
        Tensor out = matrix(qkv_ + static_cast<std::size_t>(first) * out_width, out_width, n);
        ops::linear(mid, w.projector_out, out, stream);
    }
    return qkv_;
}

} // namespace infernix::models::inkling2
