// RoPE staging of a Qwen4Exp lane (program/rope_positions.h; design §19.3.2, test VT3), host only.
// The oracle is an independent port of the frontend's position rule (text tokens: their running
// position on every axis; an image run of h x w merged tokens: (cur, cur + y, cur + x), then cur +=
// max(h, w); rope_delta = max + 1 - length), never the helpers under test. Checks: prompt tokens of a
// media prompt, tokens past the prompt (forced tokens, decode, verification and draft cells) at
// index + delta, a text-only lane at its index, pooled-block starts (token R floor(start / R)) for
// every start, call staging layout ([columns, 3] axis-major, only the call's columns written), and
// MTP sub-chunks whose first cell is not a multiple of 4 or 512, with a prepended cell.

#include "models/qwen4_exp/program/rope_positions.h"

#include <algorithm>
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

struct ImageRun {
    int begin = 0, rows = 0, cols = 0;
};

// The frontend's rule: positions [3, n] axis-major and the rope delta.
struct Positions {
    std::vector<std::int32_t> axes;
    std::int32_t delta = 0;
};

Positions oracle(int n, const std::vector<ImageRun>& runs) {
    Positions out;
    out.axes.assign(3ULL * n, 0);
    std::int32_t current = 0, maximum = 0;
    int cursor           = 0;
    const auto text = [&](int begin, int end) {
        for (int i = begin; i < end; ++i) {
            const std::int32_t p = current + (i - begin);
            for (int a = 0; a < 3; ++a) { out.axes[static_cast<std::size_t>(a) * n + i] = p; }
            maximum = std::max(maximum, p);
        }
        current += end - begin;
    };
    for (const ImageRun& run : runs) {
        text(cursor, run.begin);
        int i = run.begin;
        for (int y = 0; y < run.rows; ++y) {
            for (int x = 0; x < run.cols; ++x, ++i) {
                out.axes[i]                             = current;
                out.axes[static_cast<std::size_t>(n) + i]     = current + y;
                out.axes[2 * static_cast<std::size_t>(n) + i] = current + x;
                maximum = std::max({maximum, current + y, current + x});
            }
        }
        current += std::max(run.rows, run.cols);
        cursor = run.begin + run.rows * run.cols;
    }
    text(cursor, n);
    out.delta = maximum + 1 - n;
    return out;
}

q4::RopePosition expected(const Positions& p, int n, std::uint32_t index) {
    if (index < static_cast<std::uint32_t>(n)) {
        return {p.axes[index], p.axes[static_cast<std::size_t>(n) + index], p.axes[2 * static_cast<std::size_t>(n) + index]};
    }
    const auto v = static_cast<std::int32_t>(index) + p.delta;
    return {v, v, v};
}

std::string str(const q4::RopePosition& r) {
    return "(" + std::to_string(r[0]) + "," + std::to_string(r[1]) + "," + std::to_string(r[2]) + ")";
}

} // namespace

int main() {
    try {
        constexpr std::uint32_t kRatio = 4;

        // A text-only lane: the index on every axis, also past the prompt.
        const q4::LaneRope text{.prompt = {}, .prompt_tokens = 40, .delta = 0};
        bool ok = true;
        for (std::uint32_t i = 0; i < 200; ++i) { ok &= q4::rope_of(text, i) == q4::RopePosition{static_cast<std::int32_t>(i), static_cast<std::int32_t>(i), static_cast<std::int32_t>(i)}; }
        check(ok, "a text-only lane takes its index on every axis");

        // A media prompt: text, a 6 x 9 image from token 10, text, a 12 x 5 image, text.
        constexpr int kN = 10 + 54 + 37 + 60 + 23;
        const std::vector<ImageRun> runs{{10, 6, 9}, {101, 12, 5}};
        const Positions p = oracle(kN, runs);
        const q4::LaneRope media{.prompt = p.axes, .prompt_tokens = kN, .delta = p.delta};
        check(p.delta < 0, "the image prompt's rope delta is negative (an image advances by max(h, w) < h w)");
        ok = true;
        std::uint32_t first_bad = 0;
        // Prompt tokens, then forced tokens, decode, verification and draft cells past the prompt.
        for (std::uint32_t i = 0; i < kN + 300; ++i) {
            if (q4::rope_of(media, i) != expected(p, kN, i)) {
                if (ok) { first_bad = i; }
                ok = false;
            }
        }
        check(ok, "rope_of equals the frontend rule for prompt and later tokens (first difference at " +
                      std::to_string(first_bad) + ")");
        check(q4::rope_of(media, kN) == q4::RopePosition{kN + p.delta, kN + p.delta, kN + p.delta},
              "the first token past the prompt continues at length + delta");
        check(q4::rope_of(media, 10 + 2 * 9 + 4)[1] - q4::rope_of(media, 10)[1] == 2 &&
                  q4::rope_of(media, 10 + 2 * 9 + 4)[2] - q4::rope_of(media, 10)[2] == 4,
              "image token (y 2, x 4) is offset by its row and column " + str(q4::rope_of(media, 32)));

        // Block starts: token R floor(start / R), for every start.
        ok = true;
        for (std::uint32_t start = 0; start < kN + 20; ++start) {
            ok &= q4::block_start_rope(media, start, kRatio) == expected(p, kN, start / kRatio * kRatio);
        }
        check(ok, "block_start_rope is the RoPE position of token R floor(start / R)");

        // Call staging: [columns, 3] axis-major; only the call's columns are written.
        constexpr std::int32_t kColumns = 7;
        std::vector<std::int32_t> words(3 * kColumns, -1);
        q4::stage_rope(media, 60, 3, words, kColumns, 2);
        ok = true;
        for (std::int32_t c = 0; c < kColumns; ++c) {
            for (int a = 0; a < 3; ++a) {
                const std::int32_t got = words[static_cast<std::size_t>(a) * kColumns + c];
                ok &= c >= 2 && c < 5 ? got == expected(p, kN, 60 + static_cast<std::uint32_t>(c - 2))[static_cast<std::size_t>(a)]
                                      : got == -1;
            }
        }
        check(ok, "stage_rope writes axis a of column c at a * columns + c, nothing else");
        bool threw = false;
        try {
            q4::stage_rope(media, 0, 3, words, kColumns, 5);
        } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "stage_rope refuses columns past the call");

        // MTP sub-chunks: a prepended cell makes the first cell 1022 (not a multiple of 4 or 512);
        // 1,100 cells in sub-chunks of 512 cover the second image and the text after the prompt.
        for (const std::uint32_t first : {1022U, 13U, 0U}) {
            constexpr std::int32_t kCells = 1100, kSub = 512;
            const std::int32_t chunks = (kCells + kSub - 1) / kSub;
            const q4::LaneRope& lane  = first == 0 ? text : media;
            const Positions& oracle_positions = p;
            std::vector<std::int32_t> cells(3 * kCells + 9, -7), blocks(3 * chunks + 3, -7);
            q4::stage_mtp_chunk_rope(lane, first, kCells, kSub, kRatio, cells, blocks);
            ok = true;
            for (std::int32_t k = 0; k < chunks; ++k) {
                const std::int32_t n     = std::min(kSub, kCells - k * kSub);
                const std::uint32_t base = first + static_cast<std::uint32_t>(k * kSub);
                for (std::int32_t i = 0; i < n; ++i) {
                    const q4::RopePosition want =
                        first == 0 ? q4::RopePosition{static_cast<std::int32_t>(base + i), static_cast<std::int32_t>(base + i),
                                                      static_cast<std::int32_t>(base + i)}
                                   : expected(oracle_positions, kN, base + static_cast<std::uint32_t>(i));
                    for (int a = 0; a < 3; ++a) {
                        ok &= cells[3ULL * kSub * k + static_cast<std::size_t>(a) * n + i] == want[static_cast<std::size_t>(a)];
                    }
                }
                const q4::RopePosition block = q4::block_start_rope(lane, base, kRatio);
                for (int a = 0; a < 3; ++a) { ok &= blocks[3ULL * k + a] == block[static_cast<std::size_t>(a)]; }
            }
            ok &= std::all_of(cells.begin() + 3 * kCells, cells.end(), [](std::int32_t v) { return v == -7; });
            ok &= std::all_of(blocks.begin() + 3 * chunks, blocks.end(), [](std::int32_t v) { return v == -7; });
            check(ok, "MTP sub-chunks from cell " + std::to_string(first) +
                          ": each [n_k, 3] block at 3 * 512 * k and its block start at 3 * k");
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        ++failures;
    }
    std::printf(failures == 0 ? "qwen4_exp rope position checks passed\n" : "FAIL: %d checks failed\n", failures);
    return failures == 0 ? 0 : 1;
}
