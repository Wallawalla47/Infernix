// idle-wide-prefill planning rules (models/qwen3_5/program/planning/idle_prefill.h): the idle width
// ladder stays on the chunk's grid, and a wide step ends where a chunk-wide prefill realizes the
// next flexible prefix-cache tap. The memory fit (runtime resolve_kv_capacity_beside) is covered by
// infernix_kv_capacity_test; the Program's use of both needs a model (see docs/maintainer/model-
// quality.md section 6.7 for the real-model checks).

#include "models/qwen3_5/program/planning/idle_prefill.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

namespace q = infernix::models::qwen3_5::detail;

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

// Wave rounding as prefill_step_width applies it: clamp to the context, then whole waves.
std::uint32_t step_width(std::uint32_t requested, std::uint32_t context, std::uint32_t wave) {
    requested = std::min(requested, context);
    if (wave == 0 || requested < wave) { return requested; }
    return requested / wave * wave;
}

std::vector<std::uint32_t> ladder(std::uint32_t chunk_request, std::uint32_t context,
                                  std::uint32_t wave) {
    std::array<std::uint32_t, q::kIdlePrefillRequestedWidths.size()> steps{};
    for (std::size_t i = 0; i < steps.size(); ++i) {
        steps[i] = step_width(q::kIdlePrefillRequestedWidths[i], context, wave);
    }
    return q::idle_prefill_width_ladder(step_width(chunk_request, context, wave), steps);
}

// Reference: the boundary where hybrid_after_prefill_chunk realizes a flexible tap when prefill
// advances one chunk at a time from `cursor`, or 0 when no boundary below the prompt end does.
std::uint32_t chunk_wide_realization(std::uint32_t cursor, std::uint32_t prompt,
                                     std::uint32_t chunk, std::uint32_t tap) {
    for (std::uint32_t boundary = cursor + chunk; boundary < prompt; boundary += chunk) {
        if (tap <= boundary || prompt - boundary <= chunk) { return boundary; }
    }
    return 0;
}

} // namespace

int main() {
    int failures = 0;

    // RTX 5090, 24 query heads, a wave-tiled prompt kernel: 896-token waves, chunk 1024 -> 896.
    failures += check(ladder(1024, 262144, 896) == std::vector<std::uint32_t>{3584, 1792},
                      "default chunk on 896-token waves plans 3584 then 1792 (1024 is the chunk)");
    // Without wave tiling (BF16 KV): 4096 then 2048.
    failures += check(ladder(1024, 262144, 0) == std::vector<std::uint32_t>{4096, 2048},
                      "default chunk without waves plans 4096 then 2048");
    // A 2048 chunk on waves (1792): only 3584 is wider and on its grid.
    failures += check(ladder(2048, 262144, 896) == std::vector<std::uint32_t>{3584},
                      "a 1792 chunk plans 3584 only");
    // A chunk as wide as the widest request leaves nothing to widen.
    failures += check(ladder(4096, 262144, 896).empty(), "a 3584 chunk plans no idle width");
    failures += check(ladder(4096, 262144, 0).empty(), "a 4096 chunk plans no idle width");
    // A chunk off the requested grid: widths round down to its multiples.
    failures += check(ladder(1536, 262144, 0) == std::vector<std::uint32_t>{3072},
                      "a 1536 chunk plans 3072 (4096 rounded down; 2048 rounds to the chunk)");
    // The context clamps every width, as it clamps the chunk.
    failures += check(ladder(1024, 2000, 896) == std::vector<std::uint32_t>{1792},
                      "a 2000-token context plans 1792 only");
    failures += check(ladder(1024, 1000, 896).empty(), "a 1000-token context plans nothing");
    for (const std::uint32_t chunk_request : {512U, 1024U, 1536U, 2048U}) {
        for (const std::uint32_t wave : {0U, 640U, 896U}) {
            const std::uint32_t chunk = step_width(chunk_request, 262144, wave);
            const std::vector<std::uint32_t> widths = ladder(chunk_request, 262144, wave);
            for (std::size_t i = 0; i < widths.size(); ++i) {
                // A chunk below one wave is not whole waves itself; its multiples need not be.
                failures += check(widths[i] > chunk && widths[i] % chunk == 0 &&
                                      (wave == 0 || chunk % wave != 0 || widths[i] % wave == 0) &&
                                      (i == 0 || widths[i] < widths[i - 1]),
                                  "each idle width is a wider multiple of the chunk (and of the "
                                  "wave when the chunk is), widest first");
            }
        }
    }

    // A wide step ends at or before the boundary a chunk-wide prefill realizes the tap at, and on
    // that chunk grid.
    for (const std::uint32_t chunk : {896U, 1024U}) {
        for (const std::uint32_t cursor : {0U, 64U, 4000U}) {
            for (const std::uint32_t prompt :
                 {cursor + 1U, cursor + chunk, cursor + chunk + 1U, cursor + 3U * chunk,
                  cursor + 5U * chunk + 17U, cursor + 24576U}) {
                for (const std::uint32_t tap :
                     {0U, cursor, cursor + 1U, cursor + chunk, cursor + chunk + 1U,
                      cursor + 2U * chunk + 5U, prompt - 1U, prompt}) {
                    const std::uint32_t width = 4U * chunk;
                    const std::uint32_t limit =
                        q::flexible_tap_step_limit(cursor, prompt, chunk, tap, width);
                    const std::uint32_t boundary = chunk_wide_realization(cursor, prompt, chunk, tap);
                    const std::uint32_t expected =
                        boundary == 0 ? width : std::min(width, boundary - cursor);
                    failures += check(limit == expected,
                                      "wide step limit matches the chunk-wide realization (chunk " +
                                          std::to_string(chunk) + ", cursor " +
                                          std::to_string(cursor) + ", prompt " +
                                          std::to_string(prompt) + ", tap " +
                                          std::to_string(tap) + ")");
                    failures += check(limit >= chunk || limit == width,
                                      "a limited step still covers at least one chunk");
                    failures += check(limit % chunk == 0, "a limited step stays on the chunk grid");
                }
            }
        }
    }
    // Worked cases: a ladder tap 4096 tokens before the end of a 24,576-token prompt at chunk 896.
    failures += check(q::flexible_tap_step_limit(0, 24576, 896, 20480, 3584) == 3584,
                      "a distant tap does not limit the step");
    failures += check(q::flexible_tap_step_limit(17920, 24576, 896, 20480, 3584) == 2688,
                      "the step ends at the first chunk boundary past the tap (20608)");
    // The prompt tail tap is realized at the start of the final chunk: from 21504 the chunks
    // start at 22400, 23296 and 24192, the last of which leaves 384 tokens.
    failures += check(q::flexible_tap_step_limit(21504, 24576, 896, 24575, 3584) == 2688,
                      "the step ends where the final chunk starts");
    failures += check(q::flexible_tap_step_limit(23296, 24576, 896, 24575, 3584) == 896,
                      "one chunk before the final one the step is one chunk");
    failures += check(q::flexible_tap_step_limit(24192, 24576, 896, 24575, 3584) == 3584,
                      "inside the final chunk nothing limits the step");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
