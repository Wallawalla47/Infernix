#include "models/qwen4_exp/program/route_trace.h"

#include <array>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::models::qwen4_exp {
namespace {

constexpr std::uint32_t kVersion = 1;
constexpr std::size_t kBufferBytes = std::size_t{4} << 20;

std::mutex& seam_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::filesystem::path& seam_path() {
    static std::filesystem::path path;
    return path;
}

} // namespace

RouteTrace::RouteTrace(const std::filesystem::path& path, const RouteTraceSetup& setup)
    : layers_(setup.layers), top_k_(setup.top_k), max_columns_(setup.max_columns) {
#ifdef _WIN32
    file_ = _wfopen(path.c_str(), L"wb");
#else
    file_ = std::fopen(path.c_str(), "wb");
#endif
    if (file_ == nullptr) { throw std::runtime_error("route trace: cannot create " + path.string()); }
    std::setvbuf(file_, nullptr, _IOFBF, kBufferBytes);
    std::array<std::uint32_t, 16> header{};
    std::memcpy(header.data(), "NRTRACE1", 8);
    header[2]  = kVersion;
    header[3]  = setup.layers;
    header[4]  = setup.experts;
    header[5]  = setup.top_k;
    header[6]  = setup.frames;
    header[7]  = setup.max_columns;
    header[8]  = setup.lanes;
    header[9]  = setup.max_width;
    header[10] = setup.mtp_draft_tokens;
    header[11] = setup.ngram_draft_tokens;
    header[12] = setup.prefill_chunk;
    write(header.data(), sizeof(header));
}

RouteTrace::~RouteTrace() {
    if (file_ != nullptr) { (void)std::fclose(file_); }
}

void RouteTrace::write(const void* data, std::size_t bytes) {
    if (bytes != 0 && std::fwrite(data, 1, bytes, file_) != bytes) {
        throw std::runtime_error("route trace: write failed");
    }
}

void RouteTrace::append(const RouteTraceRound& round, std::span<const std::int32_t> lanes,
                        std::span<const std::uint8_t> live, const std::int32_t* routes, std::size_t route_stride) {
    const std::uint64_t columns = static_cast<std::uint64_t>(round.rows) * round.width;
    if (round.kind == RouteTraceKind::Graph || columns == 0 || columns > max_columns_ || lanes.size() != round.rows ||
        (!live.empty() && live.size() < columns) || static_cast<std::uint64_t>(top_k_) * columns > route_stride) {
        throw std::logic_error("route trace: round does not match its routes");
    }
    const std::array<std::uint32_t, 8> head{static_cast<std::uint32_t>(round.kind),
                                            static_cast<std::uint32_t>(columns),
                                            round.rows,
                                            round.width,
                                            round.tokens,
                                            round.budget,
                                            round.position,
                                            0U};
    write(head.data(), sizeof(head));
    write(lanes.data(), lanes.size_bytes());
    std::vector<std::uint8_t> mask((columns + 3) / 4 * 4, 0);
    for (std::size_t c = 0; c < columns; ++c) { mask[c] = live.empty() ? 1 : (live[c] != 0 ? 1 : 0); }
    write(mask.data(), mask.size());
    const std::size_t used = static_cast<std::size_t>(top_k_) * columns;
    for (std::uint32_t layer = 0; layer < layers_; ++layer) {
        write(routes + static_cast<std::size_t>(layer) * route_stride, used * sizeof(std::int32_t));
    }
}

void RouteTrace::graph(RouteTraceGraph family, std::uint32_t batch, std::uint32_t width, std::uint64_t free_before,
                       std::uint64_t free_after) {
    const std::array<std::uint32_t, 8> head{static_cast<std::uint32_t>(RouteTraceKind::Graph),
                                            static_cast<std::uint32_t>(family),
                                            batch,
                                            width,
                                            static_cast<std::uint32_t>(free_before),
                                            static_cast<std::uint32_t>(free_before >> 32),
                                            static_cast<std::uint32_t>(free_after),
                                            static_cast<std::uint32_t>(free_after >> 32)};
    write(head.data(), sizeof(head));
}

namespace testing {

void set_route_trace(std::filesystem::path path) {
    const std::lock_guard lock(seam_mutex());
    seam_path() = std::move(path);
}

std::filesystem::path route_trace() {
    const std::lock_guard lock(seam_mutex());
    return seam_path();
}

} // namespace testing

} // namespace ninfer::models::qwen4_exp
