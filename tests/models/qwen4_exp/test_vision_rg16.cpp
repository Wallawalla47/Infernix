// The NVIDIA FP4-Local vision tower's nvfp4_rg16_v1 projections at binding and planning (Qwen3.8-
// Flash-Next design §6.1.2), host only:
// - a complete nvfp4_rg16_v1 parent of each vision MLP and merger shape becomes, through
//   ops::prepare_linear_weight (as Qwen4Exp's Parameters prepare the tower), a native NVFP4_MUL /
//   Rg16 Weight with the parent's units and FP32 multiplier and the A16 policy; a row range of it is
//   refused (the layout addresses whole matrices);
// - Linear's workspace query plans each of them at every width the tower runs (4 .. 131,072
//   patches) with one pass's workspace, and refuses an unregistered [N, K];
// - the BF16 tower of the other Flash-Next artifacts still plans (its registered BF16 problems);
// - the vision offload window stages every object 16-byte aligned after an Rg16 payload, whose size
//   ends 4 bytes past a 256-byte boundary, while a BF16 layer's objects keep their packed offsets.

#include "artifact/materializer.h"
#include "core/weight.h"
#include "core/weight_view.h"
#include "infernix/ops/linear.h"
#include "infernix/ops/weight_input.h"
#include "models/qwen3_5/load/vision_overlay.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace infernix;
namespace qwen = infernix::models::qwen3_5;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

template <class F>
bool refuses(F&& f) {
    try {
        f();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

struct Shape {
    std::uint64_t n, k;
};

// fc1, fc2, merger fc1, merger fc2.
constexpr std::array<Shape, 4> kVision = {{{4304, 1152}, {1152, 4304}, {4608, 4608}, {2560, 4608}}};

std::uint64_t rg16_bytes(Shape s) {
    const std::vector<std::uint64_t> shape{s.n, s.k};
    return weight_geometry(QType::NVFP4_MUL, QuantLayout::Rg16, shape).bytes;
}

void test_native_weights() {
    for (const Shape s : kVision) {
        const std::string label = "[" + std::to_string(s.n) + ", " + std::to_string(s.k) + "]";
        const std::vector<std::uint64_t> shape{s.n, s.k};
        const WeightGeometry g = weight_geometry(QType::NVFP4_MUL, QuantLayout::Rg16, shape);
        check(g.code_bytes == s.n / 16 * (s.k / 16) * 144 && g.divisor_offset % 256 == 0 &&
                  g.bytes == g.divisor_offset + 4,
              label + ": 16-row groups of 144-byte units, then the multiplier word");
        std::vector<std::byte> payload(g.bytes); // host bytes stand in for the materialized backing
        WeightParent parent{g, payload.data()};
        parent.weight_multiplier = 3.0517578e-05F;
        const WeightView view{shape, {WeightRegion{&parent, 0, g.elements}}};
        const ops::SingleProjectionWeight prepared = ops::prepare_linear_weight(ops::WeightInput{view});
        const Weight& w = prepared.weight;
        check(w.qtype == QType::NVFP4_MUL && w.layout == QuantLayout::Rg16 && w.n == static_cast<std::int32_t>(s.n) &&
                  w.k == static_cast<std::int32_t>(s.k) && w.qdata == payload.data() && w.payload == payload.data() &&
                  w.weight_multiplier == parent.weight_multiplier && prepared.policy == ops::LinearPolicy::A16Only,
              label + ": a native NVFP4_MUL / Rg16 Weight with the parent's units and multiplier");
        const WeightView rows{{16, s.k}, {WeightRegion{&parent, 0, 16 * s.k}}};
        check(refuses([&] { (void)ops::prepare_linear_weight(ops::WeightInput{rows}); }),
              label + ": a row range of an nvfp4_rg16_v1 matrix is refused");
        const auto n = static_cast<std::int32_t>(s.n), k = static_cast<std::int32_t>(s.k);
        const std::size_t one_pass =
            ops::linear_workspace_capacity_bytes(QType::NVFP4_MUL, n, k, ops::LinearPolicy::A16Only, 4, 4096);
        const std::size_t widest =
            ops::linear_workspace_capacity_bytes(QType::NVFP4_MUL, n, k, ops::LinearPolicy::A16Only, 4, 131072);
        check(one_pass > 0 && widest == one_pass, label + ": every width plans within one pass's workspace");
        check(ops::linear_workspace_capacity_bytes(QType::NVFP4_MUL, n, k, ops::LinearPolicy::A16Only, 1, 1) > 0,
              label + ": a one-column call plans the GEMV");
        // The other Flash-Next artifacts keep this projection in BF16 on its registered routes.
        (void)ops::linear_workspace_capacity_bytes(QType::BF16, n, k, ops::LinearPolicy::A16Only, 4, 131072);
    }
    check(refuses([] {
              (void)ops::linear_workspace_capacity_bytes(QType::NVFP4_MUL, 4304, 1168, ops::LinearPolicy::A16Only, 4,
                                                         64);
          }),
          "an unregistered nvfp4_rg16_v1 [N, K] is refused");
}

// One layer group pinned as the binder places it (256-byte aligned objects), staged by the overlay.
qwen::VisionOverlayGroup stage(const std::vector<std::uint64_t>& bytes) {
    artifact::MaterializationPlan plan;
    std::vector<artifact::ObjectHandle> layer;
    std::uint64_t at = 0;
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        plan.pinned_objects.push_back({artifact::ObjectHandle{i}, at, bytes[i]});
        layer.push_back(artifact::ObjectHandle{i});
        at = (at + bytes[i] + 255) / 256 * 256;
    }
    return qwen::make_vision_overlay_layout(plan, {}, {layer}, {}).layers.front();
}

void test_overlay_staging() {
    // FP4-Local: fc1 (Rg16), its BF16 bias, fc2 (Rg16), its BF16 bias.
    const std::vector<std::uint64_t> fp4_local{rg16_bytes(kVision[0]), 4304 * 2, rg16_bytes(kVision[1]), 1152 * 2};
    check(fp4_local[0] % 16 == 4, "an nvfp4_rg16_v1 payload ends 4 bytes past a 16-byte boundary");
    const qwen::VisionOverlayGroup group = stage(fp4_local);
    std::uint64_t end = 0, pinned = 0;
    bool aligned = true, ordered = true;
    for (std::size_t i = 0; i < group.segments.size(); ++i) {
        const auto& s = group.segments[i];
        aligned       = aligned && s.staging_offset % 16 == 0 && s.bytes == fp4_local[i];
        ordered       = ordered && s.staging_offset >= end && s.staging_offset < end + 16 && s.pinned_offset == pinned;
        end           = s.staging_offset + s.bytes;
        pinned        = (pinned + s.bytes + 255) / 256 * 256;
    }
    check(group.segments.size() == fp4_local.size() && aligned, "every staged object starts 16-byte aligned");
    check(ordered && group.bytes == end, "staged objects are packed in order with at most 15 bytes of padding");
    // A BF16 layer (the other Flash-Next artifacts): whole multiples of 16 bytes, staged unpadded.
    const std::vector<std::uint64_t> bf16{4304ULL * 1152 * 2, 4304 * 2, 1152ULL * 4304 * 2, 1152 * 2};
    const qwen::VisionOverlayGroup packed = stage(bf16);
    std::uint64_t offset = 0;
    bool unpadded = true;
    for (std::size_t i = 0; i < packed.segments.size(); ++i) {
        unpadded = unpadded && packed.segments[i].staging_offset == offset;
        offset += bf16[i];
    }
    check(unpadded && packed.bytes == offset, "a BF16 layer's staged offsets are unchanged");
}

} // namespace

int main() {
    try {
        test_native_weights();
        test_overlay_staging();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    if (failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("vision rg16 binding: all checks passed\n");
    return 0;
}
