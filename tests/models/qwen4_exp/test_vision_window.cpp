// The Vision encode window's planning and visual columns (program/vision_window.h; design §19.3.2,
// test VT3's visual cases), host only. The oracle is direct enumeration over the token positions.
// Checks: a chunk's handoff slice and call-local columns, an image starting exactly at a chunk
// boundary, a chunk with no image token, the MTP cell mapping (cell c takes token c + 1, so a
// chunk's last cell can take the next chunk's first image token); placement W (one step, the
// handoff only lent) against L (handoff and window lent, several layers per step); items inside the
// reused prefix are not encoded, and a reused prefix ending inside an item is refused.

#include "models/qwen4_exp/program/vision_window.h"

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace q4 = infernix::models::qwen4_exp;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

// The oracle: handoff columns of tokens in [a, b) and their positions relative to `base`.
std::pair<std::uint32_t, std::vector<std::int32_t>> enumerate(const std::vector<std::uint32_t>& visual, std::uint32_t a,
                                                               std::uint32_t b, std::uint32_t base) {
    std::uint32_t first = static_cast<std::uint32_t>(visual.size());
    std::vector<std::int32_t> columns;
    for (std::uint32_t i = 0; i < visual.size(); ++i) {
        if (visual[i] >= a && visual[i] < b) {
            if (columns.empty()) { first = i; }
            columns.push_back(static_cast<std::int32_t>(visual[i] - base));
        }
    }
    return {columns.empty() ? 0 : first, columns};
}

void check_slice(const std::vector<std::uint32_t>& visual, std::uint32_t a, std::uint32_t b, std::uint32_t base,
                 const std::string& what) {
    std::vector<std::int32_t> words(b - a + 1, -1);
    const q4::VisualSlice got = q4::visual_slice(visual, a, b, base, words);
    const auto [first, columns] = enumerate(visual, a, b, base);
    bool ok = got.count == columns.size() && (columns.empty() || got.first == first);
    for (std::size_t i = 0; ok && i < columns.size(); ++i) { ok = words[i] == columns[i]; }
    check(ok, what);
}

} // namespace

int main() {
    try {
        // Two images: tokens [10, 64) and [100, 130) of a 160-token prompt.
        std::vector<std::uint32_t> visual;
        for (std::uint32_t t = 10; t < 64; ++t) { visual.push_back(t); }
        for (std::uint32_t t = 100; t < 130; ++t) { visual.push_back(t); }
        // Prefill chunks of 32 and of 10 (the second image starts at a chunk boundary).
        for (const std::uint32_t chunk : {32U, 10U, 7U}) {
            for (std::uint32_t begin = 0; begin < 160; begin += chunk) {
                check_slice(visual, begin, std::min(begin + chunk, 160U), begin,
                            "chunk [" + std::to_string(begin) + ", +" + std::to_string(chunk) + ")");
            }
        }
        std::vector<std::int32_t> words(16, -1);
        check(q4::visual_slice(visual, 70, 86, 70, words).count == 0, "a chunk without image tokens has no columns");
        const q4::VisualSlice at_boundary = q4::visual_slice(visual, 100, 110, 100, words);
        check(at_boundary.first == 54 && at_boundary.count == 10 && words[0] == 0,
              "an image starting at the chunk boundary takes handoff column 54 at call column 0");
        // MTP cells of chunk [0, 10): cell c takes token c + 1; cell 9 takes token 10, the first image
        // token, which the next chunk holds.
        const q4::VisualSlice cells = q4::visual_slice(visual, 1, 11, 1, words);
        check(cells.first == 0 && cells.count == 1 && words[0] == 9,
              "the last MTP cell of a chunk takes the next chunk's first image token");

        // Placement. Frames of 2,764,800 bytes; the handoff is 2 * 2560 bytes per token.
        constexpr std::uint64_t kFrame = 2'764'800;
        const std::vector<q4::VisionWindowItem> small{{10, 74, 256, 64}};
        const q4::VisionWindowPlan w = q4::plan_vision_window(small, 0, 3'500'000, 136'000'000, 512ULL << 20, kFrame,
                                                              2560, 1152, 27);
        check(!w.lease && w.steps == 1 && w.frames == 1 && w.handoff_bytes == 2ULL * 2560 * 64,
              "a small image runs from the workspace in one step, lending one frame for its handoff");
        const std::vector<q4::VisionWindowItem> large{{10, 16394, 65536, 16384}};
        const q4::VisionWindowPlan l = q4::plan_vision_window(large, 0, 900'000'000, 136'000'000, 512ULL << 20, kFrame,
                                                              2560, 1152, 27);
        check(l.lease && l.steps > 1 && l.layers_per_step >= 1 && l.layers_per_step < 27, "the largest item leases a run in steps");
        check(l.frames == (((2ULL * 2560 * 16384 + 255) / 256 * 256) + l.window_bytes + kFrame - 1) / kFrame,
              "the lease covers the handoff then the window");

        // Reuse: items inside the reused prefix are skipped; a prefix ending inside an item is refused.
        const std::vector<q4::VisionWindowItem> two{{10, 74, 256, 64}, {100, 164, 256, 64}};
        const q4::VisionWindowPlan reused = q4::plan_vision_window(two, 80, 3'500'000, 0, 512ULL << 20, kFrame, 2560, 1152, 27);
        check(reused.items.size() == 1 && reused.items[0] == 1 && reused.encoded_tokens == 64,
              "an item inside the reused prefix is not encoded");
        check(q4::plan_vision_window(two, 200, 1, 0, 512ULL << 20, kFrame, 2560, 1152, 27).items.empty(),
              "every item reused: nothing to encode");
        bool threw = false;
        try {
            (void)q4::plan_vision_window(two, 40, 1, 0, 512ULL << 20, kFrame, 2560, 1152, 27);
        } catch (const std::logic_error&) { threw = true; }
        check(threw, "a reused prefix ending inside an item is refused");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        ++failures;
    }
    std::printf(failures == 0 ? "qwen4_exp vision window checks passed\n" : "FAIL: %d checks failed\n", failures);
    return failures == 0 ? 0 : 1;
}
