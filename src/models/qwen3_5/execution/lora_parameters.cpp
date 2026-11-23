#include "models/qwen3_5/execution/lora_parameters.h"

#include "infernix/ops/attn_input_proj.h"
#include "infernix/ops/gdn_input_proj.h"
#include "infernix/ops/linear_add.h"
#include "infernix/ops/linear_swiglu.h"
#include "infernix/ops/lossless_lora.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace infernix::models::qwen3_5::execution {
namespace {

// Round to the nearest BF16, ties to even (torch's float32 -> bfloat16 cast); NaN stays NaN.
std::uint16_t bf16_rn(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7F800000U) == 0x7F800000U && (bits & 0x007FFFFFU) != 0) {
        return static_cast<std::uint16_t>((bits >> 16) | 0x0040U);
    }
    bits += 0x7FFFU + ((bits >> 16) & 1U);
    return static_cast<std::uint16_t>(bits >> 16);
}

// One site's factor regions in the device buffer, in elements from its start: A at `a`, B at `b`, the site's span
// [a, end) (its A and B regions, reserved one after the other).
struct SiteLayout {
    std::uint64_t a = 0, b = 0, end = 0; // element offsets
    std::int32_t a_rows = 0;
};

class Packer {
public:
    Packer(const LoraBinding& binding, const TextConfig& text) : binding_(binding), text_(text) {
        r_ = binding.padded_rank;
    }

    std::uint64_t reserve(std::uint64_t elements) {
        const std::uint64_t at = size_;
        size_ += (elements + 127) / 128 * 128; // keep every region 256-byte aligned
        return at;
    }

    std::uint64_t size() const noexcept { return size_; }
    std::uint32_t rank() const noexcept { return r_; }

    // Writes rows [0, rank) of module `index`'s A ([rank, in]) as rows `row0 ..` of the [*, in] region at `region`,
    // the padded rows staying zero.
    void put_a(std::uint16_t* region, std::uint64_t row0, std::size_t index) const {
        const auto& m              = binding_.module(index);
        const std::vector<float> a = binding_.adapter->file().read_f32(m.a_tensor);
        std::uint16_t* out         = region + row0 * m.in;
        for (std::uint64_t i = 0; i < static_cast<std::uint64_t>(m.rank) * m.in; ++i) { out[i] = bf16_rn(a[i]); }
    }

    // Writes module `index`'s B ([out, rank]) rows: destination row dst(src) of the [*, r] region at `region` for
    // each source row.
    template <class Map>
    void put_b(std::uint16_t* region, std::size_t index, Map dst) const {
        const auto& m              = binding_.module(index);
        const std::vector<float> b = binding_.adapter->file().read_f32(m.b_tensor);
        for (std::uint64_t row = 0; row < m.out; ++row) {
            std::uint16_t* out = region + dst(row) * r_;
            for (std::uint32_t j = 0; j < m.rank; ++j) { out[j] = bf16_rn(b[row * m.rank + j]); }
        }
    }

private:
    const LoraBinding& binding_;
    const TextConfig& text_;
    std::uint32_t r_    = 0;
    std::uint64_t size_ = 0;
};

ops::LoraSegment segment(std::int32_t end, std::int32_t u_offset, float scale) {
    return ops::LoraSegment{end, u_offset, u_offset >= 0 ? scale : 0.0F};
}

std::string format_name(QType qtype) {
    switch (qtype) {
    case QType::BF16: return "bf16";
    case QType::NVFP4: return "nvfp4";
    case QType::FP8_E4M3FN_ROW_BF16: return "fp8_e4m3fn_row_bf16";
    case QType::FP8_E4M3FN_ROW_F32: return "fp8_e4m3fn_row_f32";
    case QType::Q4_G64_FP16: return "q4_g64_fp16";
    case QType::Q5_G64_FP16: return "q5_g64_fp16";
    case QType::Q6_G64_FP16: return "q6_g64_fp16";
    case QType::Q8_G32_FP16: return "q8_g32_fp16";
    default: return "format " + std::to_string(static_cast<int>(qtype));
    }
}

// --lossless (docs/maintainer/lossless-route.md §2, LoRA adapter): the factors PEFT computes with, FP32. Each site's
// A as its three exact BF16 planes (ops::lossless_lora_split3: rows [a_rows, 3 K]), each adapted parent's B in FP32
// in the term's row order (as prepare_lora's BF16 layout). A factor the split cannot represent exactly (a nonzero
// value below 2^-110) is refused by name.
std::uint64_t prepare_lossless_lora(const Model& model, TextParameters& text, DeviceBuffer& factors) {
    const LoraBinding& binding = *model.lora();
    const TextConfig& config   = model.config().text;
    const auto h               = static_cast<std::int32_t>(config.hidden_size);
    const auto r               = static_cast<std::int32_t>(binding.padded_rank);
    // A site's byte offsets: A's planes at `a`, B at `b`, its span [a, end) (both regions, one after the other).
    struct Site {
        std::uint64_t a = 0, b = 0, end = 0;
        std::int32_t a_rows = 0, k = 0;
    };
    struct LayerLayout {
        Site input, output, gate_up, down, gdn_input, gdn_output;
        std::int32_t q = -1, k = -1, v = -1, gate = -1, up = -1;
        std::int32_t gqkv = -1, gz = -1, ga = -1, gb = -1;
    };
    // The Gated DeltaNet input site's B rows: the q/k/value channels (C), z (VW), a (Hv), b (Hv).
    const std::int32_t gdn_c  = config.gdn ? static_cast<std::int32_t>(2 * config.gdn->key_width() + config.gdn->value_width()) : 0;
    const std::int32_t gdn_vw = config.gdn ? static_cast<std::int32_t>(config.gdn->value_width()) : 0;
    const std::int32_t gdn_hv = config.gdn ? static_cast<std::int32_t>(config.gdn->linear_num_value_heads) : 0;
    std::uint64_t size   = 0;
    const auto reserve   = [&](std::uint64_t bytes) {
        const std::uint64_t at = size;
        size += (bytes + 255) / 256 * 256;
        return at;
    };
    const auto a_bytes = [](std::int32_t rows, std::int32_t k) { return static_cast<std::uint64_t>(rows) * 3 * k * 2; };
    const auto b_bytes = [&](std::uint64_t rows) { return rows * static_cast<std::uint64_t>(r) * 4; };
    const auto make_site = [&](std::int32_t rows, std::int32_t k, std::uint64_t b_rows) {
        Site site;
        site.a      = reserve(a_bytes(rows, k));
        site.b      = reserve(b_bytes(b_rows));
        site.end    = size;
        site.a_rows = rows;
        site.k      = k;
        return site;
    };
    std::vector<LayerLayout> layout(binding.layers.size());
    for (std::size_t i = 0; i < binding.layers.size(); ++i) {
        const LoraLayer& layer = binding.layers[i];
        LayerLayout& l         = layout[i];
        if (layer.attention_input()) {
            const auto& a     = *config.attention;
            std::int32_t rows = 0;
            if (layer.query) { l.q = rows; rows += r; }
            if (layer.key) { l.k = rows; rows += r; }
            if (layer.value) { l.v = rows; rows += r; }
            l.input = make_site(rows, h, 2 * a.query_width() + 2 * a.key_width());
        }
        if (layer.output) {
            l.output = make_site(r, static_cast<std::int32_t>(config.attention->query_width()), static_cast<std::uint64_t>(h));
        }
        if (layer.mlp_input() || layer.down) {
            const auto inter = static_cast<std::int32_t>(std::get<DenseConfig>(config.ffn).intermediate_size);
            if (layer.mlp_input()) {
                std::int32_t rows = 0;
                if (layer.gate) { l.gate = rows; rows += r; }
                if (layer.up) { l.up = rows; rows += r; }
                l.gate_up = make_site(rows, h, 2ULL * inter);
            }
            if (layer.down) { l.down = make_site(r, inter, static_cast<std::uint64_t>(h)); }
        }
        if (layer.gdn_input()) {
            std::int32_t rows = 0;
            if (layer.gdn_qkv) { l.gqkv = rows; rows += r; }
            if (layer.gdn_z) { l.gz = rows; rows += r; }
            if (layer.gdn_a) { l.ga = rows; rows += r; }
            if (layer.gdn_b) { l.gb = rows; rows += r; }
            l.gdn_input = make_site(rows, h, static_cast<std::uint64_t>(gdn_c + gdn_vw + 2 * gdn_hv));
        }
        if (layer.gdn_output) { l.gdn_output = make_site(r, gdn_vw, static_cast<std::uint64_t>(h)); }
    }
    const bool resident = model.weights_resident();
    // One site at a time (as prepare_lora): each site's span is staged in RAM and uploaded.
    const std::uint8_t* base = nullptr;
    if (resident) {
        factors = DeviceBuffer(size);
        factors.fill(0);
        base = static_cast<const std::uint8_t*>(factors.p);
    }
    std::vector<std::uint8_t> stage;
    const Site* staged = nullptr;
    const auto refuse_split = [&](const artifact::PeftLoraModule& m, const char* factor) {
        throw std::invalid_argument("--lora --lossless: " + m.module + "'s " + factor +
                                    " holds a nonzero value below 2^-110, which the lossless route cannot split exactly");
    };
    // Module `index`'s A rows as rows `row0 ..` of the staged site's planes (rows past its rank stay zero).
    const auto put_a = [&](std::int32_t row0, std::size_t index) {
        const auto& m              = binding.module(index);
        const std::vector<float> a = binding.adapter->file().read_f32(m.a_tensor);
        std::vector<std::uint16_t> planes;
        if (!ops::lossless_lora_split3(a.data(), m.rank, static_cast<std::int64_t>(m.in), planes)) { refuse_split(m, "lora_A"); }
        // The site's span starts at its A planes.
        std::memcpy(stage.data() + static_cast<std::uint64_t>(row0) * 3 * m.in * 2, planes.data(), planes.size() * 2);
    };
    // Module `index`'s B rows: destination row dst(row) of the staged site's FP32 [*, r] region.
    const auto put_b = [&](std::size_t index, auto dst) {
        const auto& m              = binding.module(index);
        const std::vector<float> b = binding.adapter->file().read_f32(m.b_tensor);
        std::vector<std::uint16_t> check;
        if (!ops::lossless_lora_split3(b.data(), static_cast<std::int64_t>(m.out), m.rank, check)) { refuse_split(m, "lora_B"); }
        auto* out = reinterpret_cast<float*>(stage.data() + (staged->b - staged->a));
        for (std::uint64_t row = 0; row < m.out; ++row) {
            std::memcpy(out + dst(row) * r, b.data() + row * m.rank, static_cast<std::size_t>(m.rank) * 4);
        }
    };
    const auto upload = [&](const Site& site, const auto& fill) {
        staged = &site;
        stage.assign(site.end - site.a, 0);
        fill();
        factors.copy_from_host(stage.data(), stage.size(), site.a);
    };
    for (std::size_t i = 0; resident && i < binding.layers.size(); ++i) {
        const LoraLayer& layer = binding.layers[i];
        const LayerLayout& l   = layout[i];
        if (layer.attention_input()) {
            const auto& a        = *config.attention;
            const std::uint64_t d = a.head_dim, qw = a.query_width(), kw = a.key_width();
            upload(l.input, [&] {
                if (layer.query) {
                    put_a(l.q, *layer.query);
                    put_b(*layer.query, [&](std::uint64_t row) {
                        const std::uint64_t head = row / (2 * d), within = row % (2 * d);
                        return within < d ? head * d + within : qw + kw + head * d + (within - d);
                    });
                }
                if (layer.key) {
                    put_a(l.k, *layer.key);
                    put_b(*layer.key, [&](std::uint64_t row) { return qw + row; });
                }
                if (layer.value) {
                    put_a(l.v, *layer.value);
                    put_b(*layer.value, [&](std::uint64_t row) { return 2 * qw + kw + row; });
                }
            });
        }
        if (layer.output) {
            upload(l.output, [&] {
                put_a(0, *layer.output);
                put_b(*layer.output, [](std::uint64_t row) { return row; });
            });
        }
        if (layer.mlp_input()) {
            const std::uint64_t inter = std::get<DenseConfig>(config.ffn).intermediate_size;
            upload(l.gate_up, [&] {
                if (layer.gate) {
                    put_a(l.gate, *layer.gate);
                    put_b(*layer.gate, [](std::uint64_t row) { return row; });
                }
                if (layer.up) {
                    put_a(l.up, *layer.up);
                    put_b(*layer.up, [&](std::uint64_t row) { return inter + row; });
                }
            });
        }
        if (layer.down) {
            upload(l.down, [&] {
                put_a(0, *layer.down);
                put_b(*layer.down, [](std::uint64_t row) { return row; });
            });
        }
        if (layer.gdn_input()) {
            const auto c = static_cast<std::uint64_t>(gdn_c), vw = static_cast<std::uint64_t>(gdn_vw),
                       hv = static_cast<std::uint64_t>(gdn_hv);
            upload(l.gdn_input, [&] {
                if (layer.gdn_qkv) {
                    put_a(l.gqkv, *layer.gdn_qkv);
                    put_b(*layer.gdn_qkv, [](std::uint64_t row) { return row; });
                }
                if (layer.gdn_z) {
                    put_a(l.gz, *layer.gdn_z);
                    put_b(*layer.gdn_z, [&](std::uint64_t row) { return c + row; });
                }
                if (layer.gdn_a) {
                    put_a(l.ga, *layer.gdn_a);
                    put_b(*layer.gdn_a, [&](std::uint64_t row) { return c + vw + row; });
                }
                if (layer.gdn_b) {
                    put_a(l.gb, *layer.gdn_b);
                    put_b(*layer.gdn_b, [&](std::uint64_t row) { return c + vw + hv + row; });
                }
            });
        }
        if (layer.gdn_output) {
            upload(l.gdn_output, [&] {
                put_a(0, *layer.gdn_output);
                put_b(*layer.gdn_output, [](std::uint64_t row) { return row; });
            });
        }
    }
    stage = {};
    const auto scale = [&](const std::optional<std::size_t>& m) { return m ? binding.module(*m).scale : 0.0F; };
    const auto site_of = [&](const Site& s, std::initializer_list<ops::LoraSegment> segments) {
        LosslessLoraSite out;
        out.a3     = base == nullptr ? nullptr : base + s.a;
        out.a_rows = s.a_rows;
        out.k      = s.k;
        out.b      = base == nullptr ? nullptr : reinterpret_cast<const float*>(base + s.b);
        out.rank   = r;
        for (const ops::LoraSegment& seg : segments) { out.segment[out.segments++] = seg; }
        return out;
    };
    for (std::size_t i = 0; i < binding.layers.size(); ++i) {
        const LoraLayer& layer = binding.layers[i];
        const LayerLayout& l   = layout[i];
        BlockParameters& block = text.layers.at(i);
        if (layer.attention_input() || layer.output) {
            auto* attention = std::get_if<AttentionParameters>(&block.mixer);
            if (attention == nullptr) { throw std::logic_error("--lora: attention modules on a non-attention layer"); }
            const auto qw = static_cast<std::int32_t>(config.attention->query_width());
            const auto kw = static_cast<std::int32_t>(config.attention->key_width());
            if (layer.attention_input()) {
                attention->lossless_lora_input = site_of(
                    l.input, {segment(qw, l.q, scale(layer.query)), segment(qw + kw, l.k, scale(layer.key)),
                              segment(2 * qw + kw, l.q, scale(layer.query)), segment(2 * qw + 2 * kw, l.v, scale(layer.value))});
            }
            if (layer.output) { attention->lossless_lora_output = site_of(l.output, {segment(h, 0, scale(layer.output))}); }
        }
        if (layer.mlp_input() || layer.down) {
            auto* dense = std::get_if<DenseParameters>(&block.ffn);
            if (dense == nullptr) { throw std::logic_error("--lora: MLP modules on a non-dense layer"); }
            const auto inter = static_cast<std::int32_t>(std::get<DenseConfig>(config.ffn).intermediate_size);
            if (layer.mlp_input()) {
                dense->lossless_lora_gate_up =
                    site_of(l.gate_up, {segment(inter, l.gate, scale(layer.gate)), segment(2 * inter, l.up, scale(layer.up))});
            }
            if (layer.down) { dense->lossless_lora_down = site_of(l.down, {segment(h, 0, scale(layer.down))}); }
        }
        if (layer.gdn_input() || layer.gdn_output) {
            auto* gdn = std::get_if<GdnParameters>(&block.mixer);
            if (gdn == nullptr) { throw std::logic_error("--lora: linear-attention modules on a non-linear-attention layer"); }
            if (layer.gdn_input()) {
                gdn->lossless_lora_input =
                    site_of(l.gdn_input, {segment(gdn_c, l.gqkv, scale(layer.gdn_qkv)),
                                          segment(gdn_c + gdn_vw, l.gz, scale(layer.gdn_z)),
                                          segment(gdn_c + gdn_vw + gdn_hv, l.ga, scale(layer.gdn_a)),
                                          segment(gdn_c + gdn_vw + 2 * gdn_hv, l.gb, scale(layer.gdn_b))});
            }
            if (layer.gdn_output) {
                gdn->lossless_lora_output = site_of(l.gdn_output, {segment(h, 0, scale(layer.gdn_output))});
            }
        }
    }
    return resident ? 0 : size;
}

} // namespace

std::uint64_t prepare_lora(const Model& model, TextParameters& text, DeviceBuffer& factors) {
    const LoraBinding* binding = model.lora();
    if (binding == nullptr) { return 0; }
    if (model.options().lossless) { return prepare_lossless_lora(model, text, factors); }
    const TextConfig& config = model.config().text;
    const auto h             = static_cast<std::int32_t>(config.hidden_size);
    Packer packer(*binding, config);
    const auto r = static_cast<std::int32_t>(packer.rank());

    // Pass 1: the layout (offsets only), so the device buffer is one allocation.
    struct LayerLayout {
        SiteLayout input, output, gate_up, down, gdn_input, gdn_output;
        std::int32_t q = -1, k = -1, v = -1, gate = -1, up = -1; // U offsets of the stacked blocks
        std::int32_t gqkv = -1, gz = -1, ga = -1, gb = -1;
    };
    // The Gated DeltaNet input site's B: the projection parent's rows (q/k/value channels C, then z VW), then the
    // control's (a's Hv, then b's Hv).
    const std::int64_t gdn_c  = config.gdn ? static_cast<std::int64_t>(2 * config.gdn->key_width() + config.gdn->value_width()) : 0;
    const std::int64_t gdn_vw = config.gdn ? static_cast<std::int64_t>(config.gdn->value_width()) : 0;
    const std::int64_t gdn_hv = config.gdn ? static_cast<std::int64_t>(config.gdn->linear_num_value_heads) : 0;
    std::vector<LayerLayout> layout(binding->layers.size());
    for (std::size_t i = 0; i < binding->layers.size(); ++i) {
        const LoraLayer& layer = binding->layers[i];
        LayerLayout& l         = layout[i];
        if (layer.gdn_input()) {
            std::int32_t rows = 0;
            if (layer.gdn_qkv) { l.gqkv = rows; rows += r; }
            if (layer.gdn_z) { l.gz = rows; rows += r; }
            if (layer.gdn_a) { l.ga = rows; rows += r; }
            if (layer.gdn_b) { l.gb = rows; rows += r; }
            l.gdn_input.a_rows = rows;
            l.gdn_input.a      = packer.reserve(static_cast<std::uint64_t>(rows) * h);
            l.gdn_input.b      = packer.reserve(static_cast<std::uint64_t>(gdn_c + gdn_vw + 2 * gdn_hv) * r);
            l.gdn_input.end    = packer.size();
        }
        if (layer.gdn_output) {
            l.gdn_output.a_rows = r;
            l.gdn_output.a      = packer.reserve(static_cast<std::uint64_t>(r) * gdn_vw);
            l.gdn_output.b      = packer.reserve(static_cast<std::uint64_t>(h) * r);
            l.gdn_output.end    = packer.size();
        }
        if (layer.attention_input()) {
            const auto& a = *config.attention;
            std::int32_t rows = 0;
            if (layer.query) { l.q = rows; rows += r; }
            if (layer.key) { l.k = rows; rows += r; }
            if (layer.value) { l.v = rows; rows += r; }
            l.input.a_rows = rows;
            l.input.a      = packer.reserve(static_cast<std::uint64_t>(rows) * h);
            l.input.b      = packer.reserve((2 * a.query_width() + 2 * a.key_width()) * r);
            l.input.end    = packer.size();
        }
        if (layer.output) {
            l.output.a_rows = r;
            l.output.a      = packer.reserve(static_cast<std::uint64_t>(r) * config.attention->query_width());
            l.output.b      = packer.reserve(static_cast<std::uint64_t>(h) * r);
            l.output.end    = packer.size();
        }
        if (layer.mlp_input() || layer.down) {
            const auto inter = std::get<DenseConfig>(config.ffn).intermediate_size;
            if (layer.mlp_input()) {
                std::int32_t rows = 0;
                if (layer.gate) { l.gate = rows; rows += r; }
                if (layer.up) { l.up = rows; rows += r; }
                l.gate_up.a_rows = rows;
                l.gate_up.a      = packer.reserve(static_cast<std::uint64_t>(rows) * h);
                l.gate_up.b      = packer.reserve(2ULL * inter * r);
                l.gate_up.end    = packer.size();
            }
            if (layer.down) {
                l.down.a_rows = r;
                l.down.a      = packer.reserve(static_cast<std::uint64_t>(r) * inter);
                l.down.b      = packer.reserve(static_cast<std::uint64_t>(h) * r);
                l.down.end    = packer.size();
            }
        }
    }
    const std::uint64_t bytes = packer.size() * 2;
    // Engine::plan's model reads and allocates nothing; its operands keep their shapes (workspace sizing reads them)
    // with null addresses.
    const bool resident = model.weights_resident();

    // Pass 2: the factors, rounded to BF16, one site at a time: each site's span (its A and B regions) is staged in
    // RAM and uploaded, so the load's RAM peak is one site's span (2.8 MB for the 27B's gate/up at rank 32) and one
    // module's FP32 factor, not the whole buffer. The alignment gaps between spans are zeroed once on the device.
    const std::uint16_t* base = nullptr;
    if (resident) {
        factors = DeviceBuffer(bytes);
        factors.fill(0);
        base = static_cast<const std::uint16_t*>(factors.p);
    }
    std::vector<std::uint16_t> stage;
    const auto upload = [&](const SiteLayout& site, const auto& fill) {
        stage.assign(site.end - site.a, 0);
        // region(offset): the staged element at the buffer's element `offset` (within the site's span).
        fill([&](std::uint64_t offset) { return stage.data() + (offset - site.a); });
        factors.copy_from_host(stage.data(), stage.size() * 2, site.a * 2);
    };
    for (std::size_t i = 0; resident && i < binding->layers.size(); ++i) {
        const LoraLayer& layer = binding->layers[i];
        const LayerLayout& l   = layout[i];
        if (layer.attention_input()) {
            const auto& a        = *config.attention;
            const std::uint64_t d = a.head_dim, qw = a.query_width(), kw = a.key_width();
            // The single parent's rows: query [0, qw), key [qw, qw + kw), gate [qw + kw, 2 qw + kw), value after.
            upload(l.input, [&](const auto& region) {
                if (layer.query) {
                    packer.put_a(region(l.input.a), l.q, *layer.query);
                    // q_proj's rows are each head's query (d rows) then its output gate (d rows).
                    packer.put_b(region(l.input.b), *layer.query, [&](std::uint64_t row) {
                        const std::uint64_t head = row / (2 * d), within = row % (2 * d);
                        return within < d ? head * d + within : qw + kw + head * d + (within - d);
                    });
                }
                if (layer.key) {
                    packer.put_a(region(l.input.a), l.k, *layer.key);
                    packer.put_b(region(l.input.b), *layer.key, [&](std::uint64_t row) { return qw + row; });
                }
                if (layer.value) {
                    packer.put_a(region(l.input.a), l.v, *layer.value);
                    packer.put_b(region(l.input.b), *layer.value, [&](std::uint64_t row) { return 2 * qw + kw + row; });
                }
            });
        }
        if (layer.output) {
            upload(l.output, [&](const auto& region) {
                packer.put_a(region(l.output.a), 0, *layer.output);
                packer.put_b(region(l.output.b), *layer.output, [](std::uint64_t row) { return row; });
            });
        }
        if (layer.mlp_input()) {
            const std::uint64_t inter = std::get<DenseConfig>(config.ffn).intermediate_size;
            upload(l.gate_up, [&](const auto& region) {
                if (layer.gate) {
                    packer.put_a(region(l.gate_up.a), l.gate, *layer.gate);
                    packer.put_b(region(l.gate_up.b), *layer.gate, [](std::uint64_t row) { return row; });
                }
                if (layer.up) {
                    packer.put_a(region(l.gate_up.a), l.up, *layer.up);
                    packer.put_b(region(l.gate_up.b), *layer.up, [&](std::uint64_t row) { return inter + row; });
                }
            });
        }
        if (layer.down) {
            upload(l.down, [&](const auto& region) {
                packer.put_a(region(l.down.a), 0, *layer.down);
                packer.put_b(region(l.down.b), *layer.down, [](std::uint64_t row) { return row; });
            });
        }
        if (layer.gdn_input()) {
            // in_proj_qkv's rows are the projection parent's q/k/value channels in order, in_proj_z's its z rows;
            // in_proj_a's and in_proj_b's the control's a and b rows.
            const auto c = static_cast<std::uint64_t>(gdn_c), vw = static_cast<std::uint64_t>(gdn_vw),
                       hv = static_cast<std::uint64_t>(gdn_hv);
            upload(l.gdn_input, [&](const auto& region) {
                if (layer.gdn_qkv) {
                    packer.put_a(region(l.gdn_input.a), l.gqkv, *layer.gdn_qkv);
                    packer.put_b(region(l.gdn_input.b), *layer.gdn_qkv, [](std::uint64_t row) { return row; });
                }
                if (layer.gdn_z) {
                    packer.put_a(region(l.gdn_input.a), l.gz, *layer.gdn_z);
                    packer.put_b(region(l.gdn_input.b), *layer.gdn_z, [&](std::uint64_t row) { return c + row; });
                }
                if (layer.gdn_a) {
                    packer.put_a(region(l.gdn_input.a), l.ga, *layer.gdn_a);
                    packer.put_b(region(l.gdn_input.b), *layer.gdn_a, [&](std::uint64_t row) { return c + vw + row; });
                }
                if (layer.gdn_b) {
                    packer.put_a(region(l.gdn_input.a), l.gb, *layer.gdn_b);
                    packer.put_b(region(l.gdn_input.b), *layer.gdn_b,
                                 [&](std::uint64_t row) { return c + vw + hv + row; });
                }
            });
        }
        if (layer.gdn_output) {
            upload(l.gdn_output, [&](const auto& region) {
                packer.put_a(region(l.gdn_output.a), 0, *layer.gdn_output);
                packer.put_b(region(l.gdn_output.b), *layer.gdn_output, [](std::uint64_t row) { return row; });
            });
        }
    }
    stage = {};
    const auto at = [&](std::uint64_t offset) -> const void* { return base == nullptr ? nullptr : base + offset; };

    // Pass 3: the operands. Every adapted parent must have the term in its Op's kernels for its format.
    const auto scale = [&](const std::optional<std::size_t>& m) {
        return m ? binding->module(*m).scale : 0.0F;
    };
    const auto refuse = [&](std::size_t layer, const char* role, const Weight& weight) {
        throw std::invalid_argument("--lora: text layer " + std::to_string(layer) + "'s " + role + " is stored as " +
                                    format_name(weight.qtype) + " [" + std::to_string(weight.n) + "," +
                                    std::to_string(weight.k) + "], whose projection takes no LoRA term yet");
    };
    for (std::size_t i = 0; i < binding->layers.size(); ++i) {
        const LoraLayer& layer = binding->layers[i];
        const LayerLayout& l   = layout[i];
        BlockParameters& block = text.layers.at(i);
        if (layer.attention_input() || layer.output) {
            auto* attention = std::get_if<AttentionParameters>(&block.mixer);
            if (attention == nullptr) { throw std::logic_error("--lora: attention modules on a non-attention layer"); }
            const auto& a          = *config.attention;
            const auto qw          = static_cast<std::int32_t>(a.query_width());
            const auto kw          = static_cast<std::int32_t>(a.key_width());
            if (layer.attention_input()) {
                // The paired Q4/Q5 projection takes the same term: its parents' rows are the single parent's query/key
                // and gate/value halves (attn_input_proj's paired form).
                if (const auto* single = std::get_if<LinearParameters>(&attention->projection)) {
                    if (!ops::attn_input_proj_lora_supported(single->weight.qtype, single->weight.n, single->weight.k)) {
                        refuse(i, "attention input projection", single->weight);
                    }
                }
                LoraSiteParameters site;
                site.a               = {at(l.input.a), l.input.a_rows, h};
                site.addend.b        = at(l.input.b);
                site.addend.rank     = r;
                site.addend.u_rows   = l.input.a_rows;
                site.addend.segments = 4;
                site.addend.segment  = {segment(qw, l.q, scale(layer.query)), segment(qw + kw, l.k, scale(layer.key)),
                                        segment(2 * qw + kw, l.q, scale(layer.query)),
                                        segment(2 * qw + 2 * kw, l.v, scale(layer.value))};
                attention->lora_input = site;
            }
            if (layer.output) {
                if (!ops::linear_add_lora_supported(attention->output.weight.qtype, attention->output.weight.n,
                                                    attention->output.weight.k)) {
                    refuse(i, "attention output projection", attention->output.weight);
                }
                LoraSiteParameters site;
                site.a               = {at(l.output.a), r, qw};
                site.addend.b        = at(l.output.b);
                site.addend.rank     = r;
                site.addend.u_rows   = r;
                site.addend.segments = 1;
                site.addend.segment  = {segment(h, 0, scale(layer.output)), {}, {}, {}};
                attention->lora_output = site;
            }
        }
        if (layer.mlp_input() || layer.down) {
            auto* dense = std::get_if<DenseParameters>(&block.ffn);
            if (dense == nullptr) { throw std::logic_error("--lora: MLP modules on a non-dense layer"); }
            const auto inter = static_cast<std::int32_t>(std::get<DenseConfig>(config.ffn).intermediate_size);
            if (layer.mlp_input()) {
                if (!ops::linear_swiglu_lora_supported(dense->gate_up.weight.qtype, dense->gate_up.weight.n,
                                                       dense->gate_up.weight.k, dense->gate_up.policy)) {
                    refuse(i, "MLP gate/up projection", dense->gate_up.weight);
                }
                LoraSiteParameters site;
                site.a               = {at(l.gate_up.a), l.gate_up.a_rows, h};
                site.addend.b        = at(l.gate_up.b);
                site.addend.rank     = r;
                site.addend.u_rows   = l.gate_up.a_rows;
                site.addend.segments = 2;
                site.addend.segment  = {segment(inter, l.gate, scale(layer.gate)),
                                        segment(2 * inter, l.up, scale(layer.up)), {}, {}};
                dense->lora_gate_up = site;
            }
            if (layer.down) {
                if (!ops::linear_add_lora_supported(dense->down.weight.qtype, dense->down.weight.n, dense->down.weight.k)) {
                    refuse(i, "MLP down projection", dense->down.weight);
                }
                LoraSiteParameters site;
                site.a               = {at(l.down.a), r, inter};
                site.addend.b        = at(l.down.b);
                site.addend.rank     = r;
                site.addend.u_rows   = r;
                site.addend.segments = 1;
                site.addend.segment  = {segment(h, 0, scale(layer.down)), {}, {}, {}};
                dense->lora_down = site;
            }
        }
        if (layer.gdn_input() || layer.gdn_output) {
            auto* gdn = std::get_if<GdnParameters>(&block.mixer);
            if (gdn == nullptr) { throw std::logic_error("--lora: linear-attention modules on a non-linear-attention layer"); }
            const auto c = static_cast<std::int32_t>(gdn_c), vw = static_cast<std::int32_t>(gdn_vw),
                       hv = static_cast<std::int32_t>(gdn_hv);
            if (layer.gdn_input()) {
                if (layer.gdn_projection()) {
                    if (const auto* single = std::get_if<LinearParameters>(&gdn->projection)) {
                        if (!ops::gdn_input_proj_lora_supported(single->weight.qtype, single->weight.n,
                                                                single->weight.k)) {
                            refuse(i, "linear-attention input projection", single->weight);
                        }
                    }
                }
                if (layer.gdn_control()) {
                    // gdn_gating_proj's LoRA form: the BF16 control of 48 heads over 5120 inputs.
                    const auto* single = std::get_if<LinearParameters>(&gdn->control);
                    const Weight& w    = single != nullptr ? single->weight
                                                           : std::get<ops::PairedProjectionWeights>(gdn->control).first;
                    if (w.qtype != QType::BF16 || w.k != 5120 || hv != 48 || w.n != (single != nullptr ? 2 * hv : hv)) {
                        refuse(i, "linear-attention control projection", w);
                    }
                }
                LoraSiteParameters site;
                site.a = {at(l.gdn_input.a), l.gdn_input.a_rows, h};
                if (layer.gdn_projection()) {
                    site.addend.b        = at(l.gdn_input.b);
                    site.addend.rank     = r;
                    site.addend.u_rows   = l.gdn_input.a_rows;
                    site.addend.segments = 2;
                    site.addend.segment  = {segment(c, l.gqkv, scale(layer.gdn_qkv)),
                                            segment(c + vw, l.gz, scale(layer.gdn_z)), {}, {}};
                }
                gdn->lora_input = site;
                if (layer.gdn_control()) {
                    ops::LoraAddend control;
                    control.b        = at(l.gdn_input.b + static_cast<std::uint64_t>(c + vw) * r);
                    control.rank     = r;
                    control.u_rows   = l.gdn_input.a_rows;
                    control.segments = 2;
                    control.segment  = {segment(hv, l.ga, scale(layer.gdn_a)), segment(2 * hv, l.gb, scale(layer.gdn_b)),
                                        {}, {}};
                    gdn->lora_control = control;
                }
            }
            if (layer.gdn_output) {
                if (!ops::linear_add_lora_supported(gdn->output.weight.qtype, gdn->output.weight.n,
                                                    gdn->output.weight.k)) {
                    refuse(i, "linear-attention output projection", gdn->output.weight);
                }
                LoraSiteParameters site;
                site.a               = {at(l.gdn_output.a), r, vw};
                site.addend.b        = at(l.gdn_output.b);
                site.addend.rank     = r;
                site.addend.u_rows   = r;
                site.addend.segments = 1;
                site.addend.segment  = {segment(h, 0, scale(layer.gdn_output)), {}, {}, {}};
                gdn->lora_output = site;
            }
        }
    }
    return resident ? 0 : bytes;
}

} // namespace infernix::models::qwen3_5::execution
