#pragma once

#include "artifact/materializer.h"
#include "artifact/reader.h"

#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ninfer::artifact {

enum class Residency { Device, Host, Values, HostPinned };

struct ParameterReference {
    std::string name;
    Shape shape;
    Binding binding;
    Residency residency = Residency::Device;
};

struct HostValues {
    QType format           = QType::FP32;
    std::uint64_t elements = 0;
    std::vector<std::byte> data;

    [[nodiscard]] float scalar_f32() const;
    [[nodiscard]] std::vector<std::int32_t> integers() const;
};

// Collects selected logical demands; neither physical object IDs nor whole-artifact profiles
// determine parameter shapes or native operation support.
class Binder {
public:
    explicit Binder(const Reader& reader);

    [[nodiscard]] ParameterReference parameter(std::string_view name, Shape shape,
                                               Residency residency = Residency::Device,
                                               std::optional<QType> exact_format = {});
    [[nodiscard]] ParameterReference binding(std::string name, const Binding& binding, Shape shape,
                                             Residency residency,
                                             std::optional<QType> exact_format = {});
    [[nodiscard]] const Use& use(std::string_view parameter, std::string_view input) const;
    [[nodiscard]] bool contains(std::string_view parameter) const;

    [[nodiscard]] const Reader& reader() const noexcept { return reader_; }

    void require_device(ObjectHandle object, std::uint64_t alignment = 256);
    // Rank a device demand for the evictable arena suffix: higher ranks pack closer to the
    // arena end and are evicted first. Requires the object to be a device demand.
    void mark_device_evictable(ObjectHandle object, std::uint32_t evict_rank);
    void require_pinned(ObjectHandle object, std::uint64_t alignment = 256);
    [[nodiscard]] std::span<const std::byte> host_object(ObjectHandle object);
    [[nodiscard]] ObjectHandle resource(std::string_view component, std::string_view role);
    [[nodiscard]] HostValues values(const Binding& binding, std::optional<QType> format = {});
    // evictable_alignment (a power of two, typically EvictableWeightPool::kChunkBytes)
    // chunk-aligns the first ranked object so the evictable suffix is chunk-aligned.
    [[nodiscard]] MaterializationPlan finish(std::uint64_t evictable_alignment = 1) &&;

private:
    struct Demand {
        bool device             = false;
        bool host               = false;
        bool pinned             = false;
        std::uint64_t alignment = 256;
        std::uint32_t evict_rank = 0;
        std::vector<std::byte> host_data;
    };

    const Reader& reader_;
    std::vector<Demand> demands_;
    std::uint64_t read_bytes_        = 0;
    std::uint64_t owned_value_bytes_ = 0;
};

} // namespace ninfer::artifact
