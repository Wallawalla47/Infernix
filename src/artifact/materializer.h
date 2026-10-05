#pragma once

#include "artifact/schema.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/evictable_weight_pool.h"
#include "core/weight_view.h"
#include "ninfer/types.h"

#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::artifact {

class Reader;

struct DevicePlacement {
    ObjectHandle object;
    std::uint64_t offset    = 0;
    std::uint64_t bytes     = 0;
    std::uint64_t alignment = 256;
};

struct HostPlacement {
    ObjectHandle object;
    // Already-read resources move into final storage without invalidating their byte views.
    std::vector<std::byte> data;
};

struct PinnedPlacement {
    ObjectHandle object;
    std::uint64_t offset = 0;
    std::uint64_t bytes  = 0;
};

struct MaterializationPlan {
    const Reader* source                = nullptr;
    std::size_t object_count            = 0;
    std::uint64_t device_capacity_bytes = 0;
    std::uint64_t prior_read_bytes      = 0;
    std::uint64_t owned_value_bytes     = 0;
    std::vector<DevicePlacement> device_objects;
    std::vector<HostPlacement> host_objects;
    std::uint64_t pinned_capacity_bytes = 0;
    std::vector<PinnedPlacement> pinned_objects;
    // Objects read in place at run time; located, never read here.
    std::vector<ObjectHandle> streamed_objects;
    // Physical Host memory the pinned block must leave free, after the caller's later pins
    // (`later_pinned_bytes`, allocated once the model is loaded) are counted against it too.
    std::uint64_t host_reserve_bytes = kDefaultRamHeadroomBytes;
    std::uint64_t later_pinned_bytes = 0;
    // Arena suffix holding the ranked (low-traffic) device weights that an
    // EvictableWeightPool may temporarily evict; chunk-aligned by finish().
    std::uint64_t evictable_tail_bytes = 0;
};

struct MaterializationStats {
    std::uint64_t file_bytes = 0; // Declared container file set, including framing.
    std::uint64_t read_bytes = 0; // Actual payload reads, including direct-I/O alignment.
    std::uint64_t h2d_bytes  = 0;
    std::uint64_t device_capacity_bytes = 0;
    std::uint64_t retained_host_bytes   = 0;
    std::uint64_t owned_value_bytes     = 0;
    std::uint64_t peak_staging_bytes    = 0;
    std::uint64_t pinned_bytes          = 0;
    std::size_t device_object_count     = 0;
    std::size_t host_object_count       = 0;
    std::size_t pinned_object_count     = 0;
    double upload_seconds               = 0;
};

class MaterializedArtifact;

// One contiguous piece of a byte range in one artifact file.
struct FileSegment {
    std::uint32_t file   = 0; // index into StreamSource::files()
    std::uint64_t offset = 0; // byte offset in that file
    std::uint64_t at     = 0; // byte offset inside the requested range
    std::uint64_t bytes  = 0;
};

// Where Streamed objects live in the artifact's files, for reading them in place (design §19.3.7):
// the file paths (the entry and its parts) and each object's segments, kept after the Reader is
// gone. It opens no file; the reader of the bytes opens its own handles.
class StreamSource {
public:
    [[nodiscard]] std::span<const std::filesystem::path> files() const noexcept { return files_; }
    [[nodiscard]] bool contains(ObjectHandle object) const noexcept;
    [[nodiscard]] std::uint64_t object_bytes(ObjectHandle object) const;
    // The file segments of [offset, offset + bytes) of a streamed object.
    [[nodiscard]] std::vector<FileSegment> segments(ObjectHandle object, std::uint64_t offset,
                                                    std::uint64_t bytes) const;

private:
    friend MaterializedArtifact materialize(const Reader&, MaterializationPlan&&, DeviceContext&,
                                            const StartupObserver*, std::unique_ptr<EvictableWeightPool>);
    struct Object {
        std::uint64_t bytes = 0;
        std::vector<FileSegment> segments; // the whole object, `at` from its start
    };
    std::vector<std::filesystem::path> files_;
    std::vector<std::optional<Object>> objects_;
};

class MaterializedArtifact {
public:
    MaterializedArtifact()                                           = default;
    ~MaterializedArtifact()                                          = default;
    MaterializedArtifact(MaterializedArtifact&&) noexcept            = default;
    MaterializedArtifact& operator=(MaterializedArtifact&&) noexcept = default;
    MaterializedArtifact(const MaterializedArtifact&)                = delete;
    MaterializedArtifact& operator=(const MaterializedArtifact&)     = delete;

    [[nodiscard]] const WeightParent& device_parent(ObjectHandle handle) const;
    [[nodiscard]] const WeightParent& host_parent(ObjectHandle handle) const;
    [[nodiscard]] const WeightParent& pinned_parent(ObjectHandle handle) const;
    // A Streamed object's geometry with no resident data (data == nullptr).
    [[nodiscard]] const WeightParent& streamed_parent(ObjectHandle handle) const;
    [[nodiscard]] const StreamSource& stream_source() const noexcept { return stream_source_; }
    [[nodiscard]] std::span<const std::byte> host_bytes(ObjectHandle handle) const;
    [[nodiscard]] bool has_device(ObjectHandle handle) const noexcept;

    // Pinned host block backing HostPinned objects (e.g. the vision tower in overlay
    // mode), exposed so the runtime can stream the weights to device staging.
    [[nodiscard]] const PinnedHostBuffer& pinned_block() const;
    [[nodiscard]] std::span<const std::byte> pinned_bytes_range() const;

    // VMM arena pool present only when the plan reserved an evictable tail.
    [[nodiscard]] EvictableWeightPool* eviction_pool() noexcept {
        return eviction_pool_.get();
    }
    [[nodiscard]] const EvictableWeightPool* eviction_pool() const noexcept {
        return eviction_pool_.get();
    }

    [[nodiscard]] const MaterializationStats& stats() const noexcept { return stats_; }

private:
    friend MaterializedArtifact materialize(const Reader&, MaterializationPlan&&, DeviceContext&,
                                            const StartupObserver*, std::unique_ptr<EvictableWeightPool>);

    struct ObjectStorage {
        std::optional<WeightParent> device;
        std::optional<WeightParent> host;
        std::optional<WeightParent> pinned;
        std::optional<WeightParent> streamed;
        std::vector<std::byte> host_data;
    };

    std::unique_ptr<DeviceArena> arena_;
    std::unique_ptr<PinnedHostBuffer> pinned_;
    std::unique_ptr<EvictableWeightPool> eviction_pool_;
    std::vector<ObjectStorage> objects_;
    StreamSource stream_source_;
    MaterializationStats stats_;
};

[[nodiscard]] MaterializedArtifact materialize(const Reader& reader, MaterializationPlan&& plan,
                                               DeviceContext& device,
                                               const StartupObserver* startup_observer = nullptr,
                                               std::unique_ptr<EvictableWeightPool> eviction_pool =
                                                   nullptr);

} // namespace ninfer::artifact
