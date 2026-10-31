#pragma once

#include "artifact/schema.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/evictable_weight_pool.h"
#include "core/host_memory.h"
#include "core/weight_view.h"
#include "infernix/types.h"

#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace infernix::artifact {

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
    std::uint64_t host_reserve_bytes = kDefaultRamFreeBytes;
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
    friend class MaterializedArtifact; // locate_streamed fills it
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

    // Pinned host block backing HostPinned objects (e.g. the vision tower in overlay mode), at its
    // host address, so the runtime can copy the weights to device staging. Device code reads a
    // pinned object at its parent's WeightParent::device instead.
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
    friend MaterializedArtifact materialize_geometry(const Reader&, MaterializationPlan&&);

    struct ObjectStorage {
        std::optional<WeightParent> device;
        std::optional<WeightParent> host;
        std::optional<WeightParent> pinned;
        std::optional<WeightParent> streamed;
        std::vector<std::byte> host_data;
    };

    // The steps materialize() and materialize_geometry() share: the plan's counts in the stats, the
    // streamed objects' files and segments (nothing read), and the Host objects' bytes (read here
    // only when the binder did not).
    void begin(const Reader& reader, const MaterializationPlan& plan);
    void locate_streamed(const Reader& reader, const MaterializationPlan& plan);
    void bind_host_objects(const Reader& reader, std::vector<HostPlacement>& placements);

    std::unique_ptr<DeviceArena> arena_;
    std::unique_ptr<RegisteredHostBuffer> pinned_;
    std::unique_ptr<EvictableWeightPool> eviction_pool_;
    // materialize_geometry: the address ranges the device and pinned objects are bound in (no
    // memory behind them).
    std::unique_ptr<AddressReservation> geometry_device_;
    std::unique_ptr<AddressReservation> geometry_pinned_;
    std::vector<ObjectStorage> objects_;
    StreamSource stream_source_;
    MaterializationStats stats_;
};

[[nodiscard]] MaterializedArtifact materialize(const Reader& reader, MaterializationPlan&& plan,
                                               DeviceContext& device,
                                               const StartupObserver* startup_observer = nullptr,
                                               std::unique_ptr<EvictableWeightPool> eviction_pool =
                                                   nullptr);

// materialize()'s plan-only twin (Engine::plan): every object gets the parent a load gives it, with
// device and pinned objects bound at their planned offsets inside address reservations
// (AddressReservation) instead of device and pinned memory. Host objects keep the bytes the binder
// already read, NVFP4 divisors and multipliers are read from the artifact as materialize() reads
// them, and streamed objects are located; nothing else is read, uploaded, pinned or allocated on the
// device, and no CUDA call is made. A device or pinned view's bytes must never be accessed (they
// fault): its shape, format and offsets are what planning reads. The stats report no upload.
[[nodiscard]] MaterializedArtifact materialize_geometry(const Reader& reader, MaterializationPlan&& plan);

// The refusal materialize() gives a plan whose pinned block, later pins and reserve exceed
// `available_bytes` of physical Host memory; empty when they fit (or nothing is pinned).
[[nodiscard]] std::string pinned_shortfall(const MaterializationPlan& plan, std::uint64_t available_bytes);

} // namespace infernix::artifact
