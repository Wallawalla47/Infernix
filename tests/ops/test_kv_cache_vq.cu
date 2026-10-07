// Append contract of the vector-quantized KV formats (Vq2, Q4KeyVq2Value): every persistent code
// byte and row scale, every window slot and tag, against the exact host encoders of
// kv_cache_vq_reference.h; untouched cache and window bytes; and, independently of the FP32
// encoder, that every stored VQ2 word is a nearest codeword in FP64.

#include "core/paged_kv_cache.h"
#include "kv_cache_vq_reference.h"
#include "infernix/ops/kv_cache_append.h"
#include "infernix/ops/speculative_tree.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace infernix;
using namespace infernix::test;

namespace {

constexpr int kDim   = 256;
constexpr int kPage  = 64;
constexpr int kSlots = kKVWindowSlots;

std::size_t input_index(int d, int head, int token, int kv_heads) {
    return static_cast<std::size_t>(d) +
           kDim * (static_cast<std::size_t>(head) + static_cast<std::size_t>(kv_heads) * token);
}

// Fills every byte with a recognizable pattern so untouched bytes are detectable.
void fill_pattern(GuardedDeviceBuffer& buffer, std::size_t bytes, int value) {
    buffer.fill(value);
    (void)bytes;
}

template <class T>
std::vector<T> read(const GuardedDeviceBuffer& buffer, std::size_t count) {
    std::vector<T> out(count);
    buffer.copy_to_host(out.data(), count * sizeof(T));
    return out;
}

// FP64 distance of a VQ2 word under the codec's sign rule (parity-flipped coordinate negated).
double word_distance(const std::array<double, 8>& u, const int (&levels)[8]) {
    double d = 0.0;
    for (int j = 0; j < 8; ++j) {
        const double e = u[j] - levels[j] / 16.0;
        d += e * e;
    }
    return d;
}

double best_word_distance(const std::array<double, 8>& u) {
    double best = 1e300;
    int negatives = 0;
    for (double value : u) negatives += value < 0.0 ? 1 : 0;
    for (int p = 0; p < 512; ++p) {
        // Signs follow u; an odd count flips the coordinate whose flip costs least.
        int levels[8];
        double flip_cost = 1e300;
        int flip         = -1;
        for (int j = 0; j < 8; ++j) {
            const int v = infernix::ops::kKVCacheVq2Codebook[8 * p + j];
            levels[j]   = u[j] < 0.0 ? -v : v;
            const double cost = std::abs(u[j]) * v;
            if (cost < flip_cost) {
                flip_cost = cost;
                flip      = j;
            }
        }
        if (negatives & 1) levels[flip] = -levels[flip];
        best = std::min(best, word_distance(u, levels));
    }
    return best;
}

int append_case(KvCacheStorage storage, int kv_heads, int tokens, int first_position,
                std::uint32_t seed) {
    const bool q4             = storage == KvCacheStorage::Q4KeyVq2Value;
    const int key_bytes       = q4 ? 128 : 64;
    const int last            = first_position + tokens - 1;
    const int logical_pages   = last / kPage + 1;
    const int physical_pages  = 2 * logical_pages + 1;
    std::vector<std::int32_t> mapping(static_cast<std::size_t>(logical_pages));
    for (int page = 0; page < logical_pages; ++page) mapping[page] = 2 * page + 1;
    std::vector<std::int32_t> positions(static_cast<std::size_t>(tokens));
    for (int t = 0; t < tokens; ++t) positions[t] = first_position + t;

    const std::size_t inputs = static_cast<std::size_t>(kDim) * kv_heads * tokens;
    std::vector<float> host_k(inputs), host_v(inputs);
    fill_uniform(host_k, seed, -3.0f, 3.0f);
    fill_uniform(host_v, seed + 1, -1.5f, 1.5f);
    // A zero row, a row with one dominant coordinate and a large-magnitude row.
    for (int d = 0; d < kDim; ++d) host_k[input_index(d, 0, 0, kv_heads)] = 0.0f;
    if (tokens > 1) {
        for (int d = 0; d < kDim; ++d) host_v[input_index(d, kv_heads - 1, 1, kv_heads)] = 0.0f;
        host_v[input_index(7, kv_heads - 1, 1, kv_heads)] = 40.0f;
    }
    if (tokens > 2) {
        for (int d = 0; d < kDim; ++d) host_k[input_index(d, 0, 2, kv_heads)] *= 900.0f;
    }
    round_to_bf16(host_k);
    round_to_bf16(host_v);

    DeviceBuffer d_k   = to_device_bf16(host_k);
    DeviceBuffer d_v   = to_device_bf16(host_v);
    DeviceBuffer d_pos = to_device(positions);
    DeviceBuffer d_map = to_device(mapping);

    const auto plane = [&](int extent) {
        return static_cast<std::size_t>(extent) * kPage * kv_heads * physical_pages;
    };
    GuardedDeviceBuffer cache_k(plane(key_bytes));
    GuardedDeviceBuffer cache_v(plane(64));
    GuardedDeviceBuffer scale_k(plane(1) * 2);
    GuardedDeviceBuffer scale_v(plane(1) * 2);
    const std::size_t window_codes = static_cast<std::size_t>(kDim) * kSlots * kv_heads;
    const std::size_t window_scale = static_cast<std::size_t>(4) * kSlots * kv_heads;
    const std::size_t window_tags  = static_cast<std::size_t>(2) * kSlots * kv_heads;
    GuardedDeviceBuffer win_k(window_codes);
    GuardedDeviceBuffer win_v(window_codes);
    GuardedDeviceBuffer win_ks(window_scale * 2);
    GuardedDeviceBuffer win_vs(window_scale * 2);
    GuardedDeviceBuffer win_tags(window_tags * 4);
    fill_pattern(cache_k, 0, 0x5a);
    fill_pattern(cache_v, 0, 0x5a);
    fill_pattern(scale_k, 0, 0x5a);
    fill_pattern(scale_v, 0, 0x5a);
    fill_pattern(win_k, 0, 0xa5);
    fill_pattern(win_v, 0, 0xa5);
    fill_pattern(win_ks, 0, 0xa5);
    fill_pattern(win_vs, 0, 0xa5);
    fill_pattern(win_tags, 0, 0xa5);

    PagedKVLayerView cache{
        .k_pages       = Tensor(cache_k.data(), DType::U8, {key_bytes, kPage, kv_heads, physical_pages}),
        .v_pages       = Tensor(cache_v.data(), DType::U8, {64, kPage, kv_heads, physical_pages}),
        .k_scale_pages = Tensor(scale_k.data(), DType::FP16, {1, kPage, kv_heads, physical_pages}),
        .v_scale_pages = Tensor(scale_v.data(), DType::FP16, {1, kPage, kv_heads, physical_pages}),
        .block_table   = Tensor(d_map.p, DType::I32, {logical_pages}),
        .head_dim      = kDim,
        .num_kv_heads  = kv_heads,
        .storage       = storage,
        .window =
            PagedKVWindowView{
                .k_codes  = Tensor(win_k.data(), DType::I8, {kDim, kSlots, kv_heads, 1}),
                .v_codes  = Tensor(win_v.data(), DType::I8, {kDim, kSlots, kv_heads, 1}),
                .k_scales = Tensor(win_ks.data(), DType::FP16, {4, kSlots, kv_heads, 1}),
                .v_scales = Tensor(win_vs.data(), DType::FP16, {4, kSlots, kv_heads, 1}),
                .tags     = Tensor(win_tags.data(), DType::I32, {2, kSlots, kv_heads, 1}),
            },
    };
    Tensor k(d_k.p, DType::BF16, {kDim, kv_heads, tokens});
    Tensor v(d_v.p, DType::BF16, {kDim, kv_heads, tokens});
    Tensor position_tensor(d_pos.p, DType::I32, {tokens});
    ops::kv_cache_append(k, v, position_tensor, cache, nullptr);
    cuda_synchronize();

    const auto got_k      = read<std::uint8_t>(cache_k, plane(key_bytes));
    const auto got_v      = read<std::uint8_t>(cache_v, plane(64));
    const auto got_ks     = read<std::uint16_t>(scale_k, plane(1));
    const auto got_vs     = read<std::uint16_t>(scale_v, plane(1));
    const auto got_win_k  = read<std::int8_t>(win_k, window_codes);
    const auto got_win_v  = read<std::int8_t>(win_v, window_codes);
    const auto got_win_ks = read<std::uint16_t>(win_ks, window_scale);
    const auto got_win_vs = read<std::uint16_t>(win_vs, window_scale);
    const auto got_tags   = read<std::uint32_t>(win_tags, window_tags);

    std::vector<std::uint8_t> touched_k(got_k.size(), 0), touched_v(got_v.size(), 0);
    std::vector<std::uint8_t> touched_slot(static_cast<std::size_t>(kSlots) * kv_heads, 0);
    const std::string label = std::string(q4 ? "k4v2" : "vq2") + " kv_heads=" +
                              std::to_string(kv_heads) + " tokens=" + std::to_string(tokens) +
                              " first=" + std::to_string(first_position);
    int failures       = 0;
    int reported       = 0;
    double worst_ratio = 0.0;
    const auto fail = [&](const std::string& what) {
        ++failures;
        if (reported++ < 8) std::cerr << "FAIL " << label << ": " << what << "\n";
    };

    for (int t = 0; t < tokens; ++t) {
        const int position = positions[t];
        const int page     = mapping[position / kPage];
        for (int head = 0; head < kv_heads; ++head) {
            const std::size_t row = (static_cast<std::size_t>(page) * kv_heads + head) * kPage +
                                    position % kPage;
            for (int role = 0; role < 2; ++role) {
                const auto& source = role ? host_v : host_k;
                std::array<float, kDim> y{};
                for (int d = 0; d < kDim; ++d) y[d] = source[input_index(d, head, t, kv_heads)];
                vq::hadamard(y);
                const auto encoded = (role == 0 && q4) ? vq::encode_q4(y) : vq::encode_vq2(y);
                const int bytes    = role == 0 ? key_bytes : 64;
                const auto& got    = role ? got_v : got_k;
                auto& touched      = role ? touched_v : touched_k;
                for (int b = 0; b < bytes; ++b) {
                    touched[row * bytes + b] = 1;
                    if (got[row * bytes + b] != encoded.codes[b]) {
                        fail("code byte " + std::to_string(b) + " role " + std::to_string(role) +
                             " token " + std::to_string(t) + " head " + std::to_string(head));
                        break;
                    }
                }
                if ((role ? got_vs : got_ks)[row] != encoded.scale) {
                    fail("scale role " + std::to_string(role) + " token " + std::to_string(t) +
                         " head " + std::to_string(head));
                }
                // FP64 optimality of every stored VQ2 word.
                if (!(role == 0 && q4)) {
                    double sq = 0.0;
                    for (float value : y) sq += static_cast<double>(value) * value;
                    const double sigma = std::sqrt(sq / 256.0);
                    if (sigma > 0.0) {
                        for (int w = 0; w < vq::kWords; ++w) {
                            std::array<double, 8> u{};
                            int levels[8];
                            for (int j = 0; j < 8; ++j) {
                                u[j]      = y[8 * w + j] / sigma;
                                levels[j] = encoded.levels[8 * w + j];
                            }
                            const double chosen = word_distance(u, levels);
                            const double best   = best_word_distance(u);
                            worst_ratio = std::max(worst_ratio, (chosen - best) / (best + 1e-12));
                        }
                    }
                }
                // Window slot of sinks and the call's final ring span.
                if (position < kKVWindowSinkTokens || position > last - kKVWindowRingTokens) {
                    const int slot           = vq::window_slot(position);
                    const std::size_t s      = static_cast<std::size_t>(head) * kSlots + slot;
                    touched_slot[s]          = 1;
                    const vq::Int8Row int8   = vq::encode_int8(y);
                    const auto& win_codes    = role ? got_win_v : got_win_k;
                    const auto& win_scales   = role ? got_win_vs : got_win_ks;
                    for (int d = 0; d < kDim; ++d) {
                        if (win_codes[s * kDim + d] != int8.codes[d]) {
                            fail("window code role " + std::to_string(role) + " position " +
                                 std::to_string(position));
                            break;
                        }
                    }
                    for (int g = 0; g < 4; ++g) {
                        if (win_scales[s * 4 + g] != int8.scales[g])
                            fail("window scale position " + std::to_string(position));
                    }
                    const std::uint32_t tag = vq::window_tag(position, encoded.codes, encoded.scale);
                    if (got_tags[s * 2 + role] != tag)
                        fail("window tag role " + std::to_string(role) + " position " +
                             std::to_string(position));
                }
            }
        }
    }
    // Untouched cache bytes and window slots keep their pattern.
    for (std::size_t i = 0; i < got_k.size(); ++i)
        if (!touched_k[i] && got_k[i] != 0x5a) { fail("untouched K byte changed"); break; }
    for (std::size_t i = 0; i < got_v.size(); ++i)
        if (!touched_v[i] && got_v[i] != 0x5a) { fail("untouched V byte changed"); break; }
    for (std::size_t s = 0; s < touched_slot.size(); ++s) {
        if (touched_slot[s]) continue;
        if (got_tags[2 * s] != 0xa5a5a5a5u || got_tags[2 * s + 1] != 0xa5a5a5a5u ||
            got_win_k[s * kDim] != static_cast<std::int8_t>(0xa5)) {
            fail("untouched window slot changed");
            break;
        }
    }
    if (worst_ratio > 1e-5) {
        fail("VQ2 word is not a nearest codeword: relative excess " + std::to_string(worst_ratio));
    }
    if (failures == 0) {
        std::cout << "OK   append " << label << " (VQ2 nearest-codeword excess "
                  << worst_ratio << ")\n";
    }
    return failures;
}

// Tree compaction moves an accepted column's codes and its window slot, re-tagged for the
// destination position; a destination whose source slot no longer matched its codes ends
// without a valid slot. The window is the second of two state slots, selected through the
// batch view's slot indices; the other slot stays untouched.
int compaction_case(KvCacheStorage storage) {
    constexpr int heads = 4, width = 8, first = 1200;
    const bool q4       = storage == KvCacheStorage::Q4KeyVq2Value;
    const int key_bytes = q4 ? 128 : 64;
    const int logical   = (first + width) / kPage + 1;
    const int physical  = logical + 1;
    std::vector<std::int32_t> mapping(logical);
    for (int page = 0; page < logical; ++page) mapping[page] = physical - 1 - page;
    const auto plane = [&](int extent) {
        return static_cast<std::size_t>(extent) * kPage * heads * physical;
    };
    GuardedDeviceBuffer cache_k(plane(key_bytes)), cache_v(plane(64)), scale_k(plane(2)),
        scale_v(plane(2));
    const std::size_t window_codes = static_cast<std::size_t>(kDim) * kSlots * heads;
    GuardedDeviceBuffer win_k(2 * window_codes), win_v(2 * window_codes),
        win_ks(2 * window_codes / 32), win_vs(2 * window_codes / 32), win_tags(2 * window_codes / 32);
    const auto window_planes = [&](std::int32_t first_slot, std::int32_t count) {
        const auto at = [&](GuardedDeviceBuffer& buffer, std::size_t slot_bytes) {
            return static_cast<std::uint8_t*>(buffer.data()) + slot_bytes * first_slot;
        };
        return PagedKVWindowView{
            .k_codes  = Tensor(at(win_k, window_codes), DType::I8, {kDim, kSlots, heads, count}),
            .v_codes  = Tensor(at(win_v, window_codes), DType::I8, {kDim, kSlots, heads, count}),
            .k_scales = Tensor(at(win_ks, window_codes / 32), DType::FP16, {4, kSlots, heads, count}),
            .v_scales = Tensor(at(win_vs, window_codes / 32), DType::FP16, {4, kSlots, heads, count}),
            .tags     = Tensor(at(win_tags, window_codes / 32), DType::I32, {2, kSlots, heads, count}),
        };
    };
    for (auto* b : {&cache_k, &cache_v, &scale_k, &scale_v, &win_k, &win_v, &win_ks, &win_vs,
                    &win_tags})
        b->fill(0);
    DeviceBuffer d_map = to_device(mapping);
    PagedKVLayerView cache{
        .k_pages       = Tensor(cache_k.data(), DType::U8, {key_bytes, kPage, heads, physical}),
        .v_pages       = Tensor(cache_v.data(), DType::U8, {64, kPage, heads, physical}),
        .k_scale_pages = Tensor(scale_k.data(), DType::FP16, {1, kPage, heads, physical}),
        .v_scale_pages = Tensor(scale_v.data(), DType::FP16, {1, kPage, heads, physical}),
        .block_table   = Tensor(d_map.p, DType::I32, {logical}),
        .head_dim      = kDim,
        .num_kv_heads  = heads,
        .storage       = storage,
        .window        = window_planes(1, 1),
    };
    const std::size_t inputs = static_cast<std::size_t>(kDim) * heads * width;
    std::vector<float> host_k(inputs), host_v(inputs);
    fill_uniform(host_k, 501, -2.0f, 2.0f);
    fill_uniform(host_v, 502, -1.0f, 1.0f);
    round_to_bf16(host_k);
    round_to_bf16(host_v);
    std::vector<std::int32_t> positions(width);
    for (int t = 0; t < width; ++t) positions[t] = first + t;
    DeviceBuffer d_k = to_device_bf16(host_k), d_v = to_device_bf16(host_v);
    DeviceBuffer d_pos = to_device(positions);
    ops::kv_cache_append(Tensor(d_k.p, DType::BF16, {kDim, heads, width}),
                         Tensor(d_v.p, DType::BF16, {kDim, heads, width}),
                         Tensor(d_pos.p, DType::I32, {width}), cache, nullptr);
    cuda_synchronize();
    // Corrupt the tag of column 6's K slot of head 1 so that its move must not carry a slot.
    const std::size_t slot_base = static_cast<std::size_t>(heads) * kSlots; // state slot 1
    const std::size_t stale =
        (slot_base + static_cast<std::size_t>(1) * kSlots + vq::window_slot(first + 6)) * 2;
    const std::uint32_t zero = 0x12345679u;
    cudaMemcpy(static_cast<std::uint32_t*>(win_tags.data()) + stale, &zero, 4,
               cudaMemcpyHostToDevice);
    // Accepted path [0, 3, 6]: columns 3 and 6 move onto columns 1 and 2.
    const std::vector<std::int32_t> path{0, 3, 6, -1, -1, -1, -1, -1};
    const std::vector<std::int32_t> accepted{2}, rows{0}, state_slots{1};
    DeviceBuffer d_path = to_device(path), d_acc = to_device(accepted), d_rows = to_device(rows);
    DeviceBuffer d_state_slots = to_device(state_slots);
    PagedKVBatchLayerView view = single_row_paged_kv_batch_view(cache);
    view.window                = window_planes(0, 2);
    view.window.slots          = Tensor(d_state_slots.p, DType::I32, {1});
    ops::speculative_tree_compact_kv(&view, 1, Tensor(d_pos.p, DType::I32, {width, 1}),
                                     Tensor(d_rows.p, DType::I32, {1}),
                                     Tensor(d_path.p, DType::I32, {width, 1}),
                                     Tensor(d_acc.p, DType::I32, {1}), nullptr);
    cuda_synchronize();

    const auto codes_k = read<std::uint8_t>(cache_k, plane(key_bytes));
    const auto codes_v = read<std::uint8_t>(cache_v, plane(64));
    const auto ks      = read<std::uint16_t>(scale_k, plane(1));
    const auto vs      = read<std::uint16_t>(scale_v, plane(1));
    const auto tags    = read<std::uint32_t>(win_tags, 2 * window_codes / 128);
    const auto wk      = read<std::int8_t>(win_k, 2 * window_codes);
    int failures       = 0;
    const auto row_of  = [&](int position, int head) {
        return (static_cast<std::size_t>(mapping[position / kPage]) * heads + head) * kPage +
               position % kPage;
    };
    for (const auto& [destination, source] : {std::pair{1, 3}, std::pair{2, 6}}) {
        for (int head = 0; head < heads; ++head) {
            const std::size_t d_row = row_of(first + destination, head);
            const std::size_t s_row = row_of(first + source, head);
            for (int role = 0; role < 2; ++role) {
                const int bytes      = role ? 64 : key_bytes;
                const auto& codes    = role ? codes_v : codes_k;
                std::vector<std::uint8_t> moved(codes.begin() + d_row * bytes,
                                                codes.begin() + (d_row + 1) * bytes);
                const std::vector<std::uint8_t> original(codes.begin() + s_row * bytes,
                                                         codes.begin() + (s_row + 1) * bytes);
                if (moved != original) ++failures;
                const std::uint16_t scale = (role ? vs : ks)[d_row];
                const std::size_t slot = slot_base + static_cast<std::size_t>(head) * kSlots +
                                         vq::window_slot(first + destination);
                const bool expect_slot = !(source == 6 && head == 1 && role == 0);
                const std::uint32_t want =
                    expect_slot ? vq::window_tag(first + destination, moved, scale) : 0u;
                if (tags[slot * 2 + role] != want) ++failures;
                if (expect_slot && role == 0) {
                    const std::size_t source_slot = slot_base +
                                                    static_cast<std::size_t>(head) * kSlots +
                                                    vq::window_slot(first + source);
                    if (!std::equal(wk.begin() + slot * kDim, wk.begin() + (slot + 1) * kDim,
                                    wk.begin() + source_slot * kDim))
                        ++failures;
                }
            }
        }
    }
    // State slot 0 was never named: its tags and codes stay zero.
    if (std::any_of(tags.begin(), tags.begin() + slot_base * 2, [](std::uint32_t t) { return t != 0; }) ||
        std::any_of(wk.begin(), wk.begin() + window_codes, [](std::int8_t c) { return c != 0; }))
        ++failures;
    std::cout << (failures == 0 ? "OK   " : "FAIL ") << "tree compaction window "
              << (q4 ? "k4v2" : "vq2") << " (" << failures << " mismatches)\n";
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP no CUDA device\n";
        return 77;
    }
    int failures = 0;
    for (const auto storage : {KvCacheStorage::Vq2, KvCacheStorage::Q4KeyVq2Value}) {
        for (const int heads : {4, 2}) {
            failures += append_case(storage, heads, 3, 61, 0x1234u + heads);  // sink + page edge
            failures += append_case(storage, heads, 9, 1015, 0x4321u + heads); // ring wrap
        }
        // A wide append writes only its final ring span (and sinks) into the window.
        failures += append_case(storage, 4, 1100, 0, 0x7777u);
        failures += compaction_case(storage);
    }
    std::cout << (failures == 0 ? "PASS" : "FAIL") << " kv_cache_vq (" << failures
              << " failures)\n";
    return failures == 0 ? 0 : 1;
}
