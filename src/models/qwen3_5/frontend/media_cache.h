#pragma once

#include "models/qwen3_5/frontend/processor.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string_view>

namespace infernix::models::qwen3_5::frontend {

struct MediaCacheKey {
    std::array<std::uint8_t, 32> digest{};
    Modality modality = Modality::Image;

    [[nodiscard]] bool operator==(const MediaCacheKey&) const noexcept = default;
};

struct PreparedMedia {
    VisionItem item;
    std::shared_ptr<const qwen3_5::PreparedMediaPayload> payload;
};

struct MediaCacheFlight;
struct MediaPreparationGate;

class MediaPreparationPermit {
public:
    MediaPreparationPermit() noexcept = default;
    ~MediaPreparationPermit();
    MediaPreparationPermit(MediaPreparationPermit&&) noexcept;
    MediaPreparationPermit& operator=(MediaPreparationPermit&&) noexcept;

    MediaPreparationPermit(const MediaPreparationPermit&)            = delete;
    MediaPreparationPermit& operator=(const MediaPreparationPermit&) = delete;

    void reset() noexcept { release(); }

private:
    explicit MediaPreparationPermit(std::shared_ptr<MediaPreparationGate> gate) noexcept;
    void release() noexcept;
    std::shared_ptr<MediaPreparationGate> gate_;

    friend class MediaPreprocessCache;
};

enum class MediaCacheDisposition : std::uint8_t {
    Hit,
    Producer,
    SingleflightWaiter,
};

struct PendingMedia {
    MediaCacheDisposition disposition = MediaCacheDisposition::Hit;
    PreparedMedia ready;
    std::shared_ptr<MediaCacheFlight> flight;
};

struct MediaCacheRequestStats {
    std::size_t hits               = 0;
    std::size_t misses             = 0;
    std::size_t singleflight_waits = 0;
    std::size_t built_patch_bytes  = 0;
    std::size_t reused_patch_bytes = 0;
    double build_seconds           = 0.0;
};

struct MediaCacheStats {
    std::size_t capacity_bytes       = 0;
    std::size_t live_capacity_bytes  = 0;
    std::size_t retained_bytes       = 0;
    std::size_t live_bytes           = 0;
    std::size_t entries              = 0;
    std::size_t inflight             = 0;
    std::size_t queued_tasks         = 0;
    std::size_t active_tasks         = 0;
    std::uint32_t preprocess_threads = 0;
    std::uint64_t hits               = 0;
    std::uint64_t misses             = 0;
    std::uint64_t singleflight_waits = 0;
    std::uint64_t evictions          = 0;
    std::uint64_t oversize_bypasses  = 0;
};

// A single-flight waiter's flight failed for its producer's own reasons, not the media's: the producer's control
// tripped (cancelled or expired) or its per-request vision budget was exceeded, or the build never started. The waiter
// prepares the item again under its own control (begin_prepare with its own builder). A content error (undecodable
// media, one item over Vision's capacity) is shared with every waiter instead.
class MediaFlightAbandoned final : public std::runtime_error {
public:
    MediaFlightAbandoned() : std::runtime_error("the shared media preparation failed for its producer's own reasons") {}
};

class MediaPreprocessCache {
public:
    using Builder = std::function<PreparedMedia()>;

    MediaPreprocessCache(std::size_t capacity_bytes, std::size_t live_capacity_bytes,
                         std::uint32_t preprocess_threads  = 0,
                         std::size_t maximum_request_bytes = 0);
    ~MediaPreprocessCache();

    MediaPreprocessCache(const MediaPreprocessCache&)            = delete;
    MediaPreprocessCache& operator=(const MediaPreprocessCache&) = delete;

    [[nodiscard]] std::shared_ptr<qwen3_5::PreparedMediaPayload>
    allocate_payload(std::size_t elements, const PreparationControl& control);

    [[nodiscard]] MediaPreparationPermit acquire_request(const PreparationControl& control) const;

    // The producer's builder runs under `control` (its trips stop the build early); a build that finished is a result
    // whatever the producer's control does afterwards.
    [[nodiscard]] PendingMedia begin_prepare(const MediaCacheKey& key,
                                             const PreparationControl& control, Builder builder);
    // begin_prepare and await, preparing again (a copy of `builder`) when a flight it waited on was abandoned.
    [[nodiscard]] PreparedMedia get_or_prepare(const MediaCacheKey& key,
                                               const PreparationControl& control, Builder builder,
                                               MediaCacheRequestStats& request_stats);
    // The flight's result, or its error: the producer gets its own; a waiter gets a content error, or
    // MediaFlightAbandoned for one that was the producer's own.
    [[nodiscard]] PreparedMedia await(const PendingMedia& pending,
                                      const PreparationControl& control,
                                      MediaCacheRequestStats& request_stats) const;

    [[nodiscard]] MediaCacheStats stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

void check_preparation_control(const PreparationControl& control,
                               std::string_view stage = "preparation");

} // namespace infernix::models::qwen3_5::frontend
