// The Qwen3.5-122B-A10B's calibrated expert order (models/qwen3_5/program/calibrated_order.h; candidates §23 NU-43,
// offload design §9.2.1) and the warm start's seed choice the offloaded Programs share
// (runtime/expert_offload/expert_state.h choose_warm_start), host only:
// - the shape guard: a table applies only to a model of its exact shape (each field changed alone is refused), an
//   empty table to none; a table of the wrong size or with a row that is not a permutation is refused;
// - the conversion: counts are the ranks reversed, the ranked list takes every layer's next expert in turn;
// - the built-in table: absent (0 x 0), no shape has an order; generated, the 122B's shape has one (48 x 256, every
//   row a permutation) and no other shape;
// - precedence: a usable saved state wins over the calibrated order; a missing file, another artifact's state, a
//   damaged file or no file given fall back to the order; --expert-state off (no seed) and a model without an order
//   start empty.

#include "test_temp_path.h"
#include "models/qwen3_5/program/calibrated_order.h"
#include "runtime/expert_offload/expert_state.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

using namespace infernix::runtime::expert_offload::expert_cache;
using infernix::models::qwen3_5::calibrated_expert_order;
using infernix::models::qwen3_5::calibrated_order_for;
using infernix::models::qwen3_5::ExpertOrderShape;
using infernix::models::qwen3_5::has_calibrated_order_table;
using infernix::models::qwen3_5::kCalibratedOrderShape;

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

template <class F>
bool throws_invalid_argument(F&& f) {
    try {
        f();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

// Two layers of four experts: layer 0 ranks 2, 0, 3, 1; layer 1 ranks 1, 3, 0, 2.
constexpr ExpertOrderShape kSmall{64, 2, 4, 2, 32, 32};
const std::vector<std::uint16_t> kSmallOrder{2, 0, 3, 1, 1, 3, 0, 2};

void test_shape_guard() {
    const auto exact = calibrated_order_for(kSmall, kSmall, kSmallOrder);
    check(exact.has_value(), "a table applies to a model of its shape");
    const auto refused = [&](ExpertOrderShape model) { return !calibrated_order_for(model, kSmall, kSmallOrder); };
    ExpertOrderShape other = kSmall;
    other.hidden           = 128;
    check(refused(other), "another hidden size is refused");
    other        = kSmall;
    other.layers = 3;
    check(refused(other), "another layer count is refused");
    other         = kSmall;
    other.experts = 8;
    check(refused(other), "another expert count is refused");
    other       = kSmall;
    other.top_k = 1;
    check(refused(other), "another top-k is refused");
    other              = kSmall;
    other.intermediate = 64;
    check(refused(other), "another expert width is refused");
    other                     = kSmall;
    other.shared_intermediate = 64;
    check(refused(other), "another shared-expert width is refused");
    check(!calibrated_order_for(kSmall, ExpertOrderShape{}, {}), "an empty table gives no order");
    check(throws_invalid_argument([] {
              (void)calibrated_order_for(kSmall, kSmall, std::span<const std::uint16_t>(kSmallOrder).first(7));
          }),
          "a table of the wrong size is refused");
    std::vector<std::uint16_t> duplicate = kSmallOrder;
    duplicate[1]                         = 2; // layer 0: 2, 2, 3, 1
    check(throws_invalid_argument([&] { (void)calibrated_order_for(kSmall, kSmall, duplicate); }),
          "a row that repeats an expert is refused");
    std::vector<std::uint16_t> outside = kSmallOrder;
    outside[4]                         = 4; // layer 1 names expert 4 of 4
    check(throws_invalid_argument([&] { (void)calibrated_order_for(kSmall, kSmall, outside); }),
          "a row naming an expert past the count is refused");
}

void test_conversion() {
    const auto order = calibrated_order_for(kSmall, kSmall, kSmallOrder);
    if (!order) {
        check(false, "the small order converts");
        return;
    }
    // Key = layer * 4 + expert; counts are 4 - rank.
    check(order->counts == std::vector<std::uint32_t>{3, 1, 4, 2, 2, 4, 1, 3}, "counts are the ranks reversed");
    check(order->ranked == std::vector<std::uint32_t>{2, 5, 0, 7, 3, 4, 1, 6},
          "the ranked list takes every layer's next expert in turn");
}

void test_built_in_table() {
    const SavedState* order = calibrated_expert_order(kCalibratedOrderShape);
    check((order != nullptr) == has_calibrated_order_table(), "the 122B has an order exactly when one is generated");
    ExpertOrderShape flash = {2560, 48, 512, 10, 640, 640}; // Qwen3.8-Flash-Next
    check(calibrated_expert_order(flash) == nullptr, "another model's shape has no order");
    ExpertOrderShape near = kCalibratedOrderShape;
    near.top_k            = 10;
    check(calibrated_expert_order(near) == nullptr, "the 122B's layers and experts at another top-k have no order");
    if (order == nullptr) {
        std::printf("no calibrated order is generated yet (calibrated_order.inc is empty): every model starts empty\n");
        return;
    }
    constexpr std::uint32_t kLayers = 48, kExperts = 256, kKeys = kLayers * kExperts;
    check(order->counts.size() == kKeys && order->ranked.size() == kKeys, "the generated order covers every key");
    bool rows = true;
    for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
        std::vector<std::uint32_t> counts(order->counts.begin() + layer * kExperts,
                                          order->counts.begin() + (layer + 1) * kExperts);
        std::sort(counts.begin(), counts.end());
        for (std::uint32_t i = 0; i < kExperts; ++i) { rows = rows && counts[i] == i + 1; }
    }
    check(rows, "every generated layer ranks each of its 256 experts once");
}

void test_precedence() {
    const auto path = infernix::test::unique_temp_path("infernix_qwen3_5_calibrated_order_state.bin");
    const auto calibrated = calibrated_order_for(kSmall, kSmall, kSmallOrder);
    if (!calibrated) {
        check(false, "the small order converts");
        return;
    }
    constexpr std::uint32_t kKeys = 8;
    SavedState saved;
    saved.counts = {0, 7, 0, 0, 0, 0, 9, 0};
    saved.ranked = {6, 1};
    check(save_expert_state(path, "artifact A", saved).empty(), "the state saves");

    const WarmStartSeed with_file = choose_warm_start(path, "artifact A", kKeys, true, &*calibrated);
    check(with_file.source == WarmStartSource::Saved && with_file.state && with_file.state->ranked == saved.ranked,
          "a usable saved state wins over the calibrated order");
    const WarmStartSeed off_with_file = choose_warm_start(path, "artifact A", kKeys, false, &*calibrated);
    check(off_with_file.source == WarmStartSource::Saved, "a usable saved state is used even without the seed");

    const WarmStartSeed other = choose_warm_start(path, "artifact B", kKeys, true, &*calibrated);
    check(other.source == WarmStartSource::Calibrated && other.state && other.state->ranked == calibrated->ranked &&
              !other.message.empty(),
          "another artifact's state falls back to the calibrated order, with the reason");
    const auto missing = infernix::test::unique_temp_path("infernix_qwen3_5_calibrated_order_missing.bin");
    check(choose_warm_start(missing, "artifact A", kKeys, true, &*calibrated).source == WarmStartSource::Calibrated,
          "a first start (no file yet) seeds from the calibrated order");
    check(choose_warm_start({}, "artifact A", kKeys, true, &*calibrated).source == WarmStartSource::Calibrated,
          "no file given seeds from the calibrated order");
    check(choose_warm_start({}, "artifact A", kKeys, false, &*calibrated).source == WarmStartSource::None,
          "--expert-state off starts empty");
    check(choose_warm_start(missing, "artifact A", kKeys, true, nullptr).source == WarmStartSource::None,
          "a model without an order starts empty");
    check(choose_warm_start({}, "artifact A", kKeys + 1, true, &*calibrated).source == WarmStartSource::None,
          "an order of another key count is not used");
    {
        const auto size = std::filesystem::file_size(path);
        std::filesystem::resize_file(path, size - 3);
    }
    check(choose_warm_start(path, "artifact A", kKeys, true, &*calibrated).source == WarmStartSource::Calibrated,
          "a damaged file falls back to the calibrated order");
    std::filesystem::remove(path);
}

} // namespace

int main() {
    test_shape_guard();
    test_conversion();
    test_built_in_table();
    test_precedence();
    if (g_failures) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("calibrated expert order checks passed\n");
    return 0;
}
