#pragma once

// Internal route trace of the Qwen4Exp Program (ProgramOptions::route_trace; design §19.3.7, the
// memory track's step R0): every round's routed experts per layer, with its kind, columns, rows,
// live mask and the promotion budget the expert cache applied, appended to a binary file for the
// two-level replay tools/expert_cache_replay/host_tier.py; and every CUDA graph executable the
// Program instantiates, with device free memory around it (measurement RM0d). It only reads the
// route log that the expert cache already downloads with each round and queries free device
// memory outside graph capture, so it never changes execution.
//
// File, little-endian. A 64-byte header:
//   char magic[8] = "NRTRACE1"; u32 version = 1, layers, experts, top_k, frames, max_columns,
//   lanes, max_width, mtp_draft_tokens, ngram_draft_tokens, prefill_chunk; u32 reserved[3] = 0
// then one record per event, in execution order, each starting with eight u32 words whose first
// is the kind:
//   Round (kind 0-3, RouteTraceKind):
//     u32 kind, columns, rows, width, tokens, budget, position, reserved = 0
//     u32 lanes[rows]
//     u8 live[columns], zero-padded to a multiple of 4 bytes (1: the column's experts count as uses)
//     i32 routes[layers][top_k * columns] (column c's top_k experts at c * top_k)
//   Graph (kind 4): u32 kind, family (RouteTraceGraph), batch, width, then the device free bytes
//     before the capture and after the instantiation and first launch as two u64 (low word first).
//     No payload.
// For rounds, rows * width == columns, and row r's columns are r * width .. r * width + width - 1.
// tokens is what the round advances: the chunk's tokens for a prefill chunk or forced tokens, 1 for
// decode, the longest row's accepted tokens plus one for verification. position is row 0's first
// position for prefill chunks and forced tokens (0 starts a request) and kUnknownPosition
// otherwise. A graph's width is its verification or catch-up width, or the draft steps of an MTP
// draft graph (1 for a decode graph).

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <span>

namespace infernix::models::qwen4_exp {

enum class RouteTraceKind : std::uint32_t { PrefillChunk = 0, ForcedTokens = 1, Decode = 2, Verify = 3, Graph = 4 };
enum class RouteTraceGraph : std::uint32_t { Decode = 0, Verify = 1, MtpDraft = 2, MtpCatchUp = 3 };

struct RouteTraceSetup {
    std::uint32_t layers = 0, experts = 0, top_k = 0, frames = 0, max_columns = 0;
    std::uint32_t lanes = 0, max_width = 0, mtp_draft_tokens = 0, ngram_draft_tokens = 0, prefill_chunk = 0;
};

struct RouteTraceRound {
    RouteTraceKind kind    = RouteTraceKind::Decode;
    std::uint32_t rows     = 0;
    std::uint32_t width    = 0;
    std::uint32_t tokens   = 0;
    std::uint32_t budget   = 0;
    std::uint32_t position = 0;
};

class RouteTrace {
public:
    static constexpr std::uint32_t kUnknownPosition = 0xFFFFFFFFU;

    // Creates (truncates) the file and writes the header. Throws if it cannot be written.
    RouteTrace(const std::filesystem::path& path, const RouteTraceSetup& setup);
    ~RouteTrace();
    RouteTrace(const RouteTrace&)            = delete;
    RouteTrace& operator=(const RouteTrace&) = delete;

    // Appends one round. `lanes` holds round.rows lanes; an empty `live` marks every column live;
    // routes[layer * route_stride + i] for i < top_k * columns. Throws on a write error.
    void append(const RouteTraceRound& round, std::span<const std::int32_t> lanes,
                std::span<const std::uint8_t> live, const std::int32_t* routes, std::size_t route_stride);
    // Appends one graph instantiation. Throws on a write error.
    void graph(RouteTraceGraph family, std::uint32_t batch, std::uint32_t width, std::uint64_t free_before,
               std::uint64_t free_after);

private:
    void write(const void* data, std::size_t bytes);

    std::FILE* file_ = nullptr;
    std::uint32_t layers_ = 0, top_k_ = 0, max_columns_ = 0;
};

// The index of `item` within the contiguous range `items`, or -1 when it lies outside.
template <class Range, class T>
[[nodiscard]] std::ptrdiff_t route_trace_index(const Range& items, const T* item) noexcept {
    if (items.size() == 0) { return -1; }
    const T* first = items.data();
    const std::less<const T*> before;
    if (before(item, first) || !before(item, first + items.size())) { return -1; }
    return item - first;
}

namespace testing {

// The process-wide route-trace path that construct_qwen4_exp passes to ProgramOptions::route_trace
// (empty: no trace). For engine-level capture tools and tests only.
void set_route_trace(std::filesystem::path path);
[[nodiscard]] std::filesystem::path route_trace();

} // namespace testing

} // namespace infernix::models::qwen4_exp
