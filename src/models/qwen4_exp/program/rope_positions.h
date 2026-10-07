#pragma once

// RoPE positions of a Qwen4Exp lane (docs/maintainer/qwen3_8-flash-next-design.md §19.3.2, "M-RoPE
// through the text path"). Every call stages the three-axis RoPE position of each column beside its
// KV position: prompt tokens of a prompt with media take the frontend's M-RoPE positions, every
// other token its index plus the prompt's rope delta on all three axes (text-only prompts: the
// index). Equal axes rotate exactly as the 1-D index, so text lanes are unchanged.

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace infernix::models::qwen4_exp {

// A token's RoPE position: axes (temporal, height, width).
using RopePosition = std::array<std::int32_t, 3>;

// The prompt's RoPE positions, kept only for a prompt with media.
struct LaneRope {
    std::vector<std::int32_t> prompt; // I32 [3, prompt_tokens] axis-major; empty for text-only
    std::uint32_t prompt_tokens = 0;
    std::int32_t delta          = 0;   // rope_delta: the RoPE position of token i >= prompt_tokens is i + delta
};

// The RoPE position of token `index` of the lane's sequence.
[[nodiscard]] RopePosition rope_of(const LaneRope& rope, std::uint32_t index) noexcept;

// The RoPE position of the first token of the pooled block holding token `start` (QSA ratio R):
// token R * floor(start / R).
[[nodiscard]] RopePosition block_start_rope(const LaneRope& rope, std::uint32_t start, std::uint32_t ratio) noexcept;

// Writes the RoPE positions of tokens first .. first + count - 1 as columns column .. column +
// count - 1 of a call of `columns` columns: axis a of column c at words[a * columns + c].
void stage_rope(const LaneRope& rope, std::uint32_t first, std::int32_t count, std::span<std::int32_t> words,
                std::int32_t columns, std::int32_t column);

// Writes axis a of `value` at words[a * columns + column].
void stage_rope_value(const RopePosition& value, std::span<std::int32_t> words, std::int32_t columns,
                      std::int32_t column);

// The MTP cells of a prefill or forced-token chunk run in sub-chunks of `sub` cells; sub-chunk k
// holds cells first + sub*k ... Its RoPE positions are written as their own [n_k, 3] block at word
// 3 * sub * k of `cells_words`, its block start (the RoPE position of token R * floor((first +
// sub*k) / R)) as three words at 3 * k of `block_words`.
void stage_mtp_chunk_rope(const LaneRope& rope, std::uint32_t first, std::int32_t cells, std::int32_t sub,
                          std::uint32_t ratio, std::span<std::int32_t> cells_words,
                          std::span<std::int32_t> block_words);

} // namespace infernix::models::qwen4_exp
