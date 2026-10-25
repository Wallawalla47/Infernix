#include "core/dxcore_adapter.h"

#include <dlfcn.h>
#include <unistd.h>

#include <cstddef>
#include <cstring>

namespace infernix {
namespace {

// The DXCore interfaces this file calls, declared as WSL's DirectX headers lay them out
// (dxcore_interface.h, include/wsl/winadapter.h: IUnknown without a destructor, x86-64 System V
// calls). Only the virtual functions' order and signatures matter; later slots are not called.
using HResult = std::int32_t;

struct Guid {
    std::uint32_t data1;
    std::uint16_t data2;
    std::uint16_t data3;
    std::uint8_t data4[8];
};

constexpr Guid kAdapterFactoryId{0x78ee5945, 0xc36e, 0x4b13, {0xa6, 0x69, 0x00, 0x5d, 0xd1, 0x1c, 0x0f, 0x06}};
constexpr Guid kAdapterListId{0x526c7776, 0x40e9, 0x459b, {0xb7, 0x11, 0xf3, 0x2a, 0xd7, 0x6d, 0xfc, 0x28}};
constexpr Guid kAdapterId{0xf0db4c7f, 0xfe5a, 0x42a2, {0xbd, 0x62, 0xf2, 0xa6, 0xcf, 0x6f, 0xc8, 0x3e}};
// DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE: every adapter CUDA can run on has it.
constexpr Guid kCoreComputeAttribute{0x248e2800, 0xa793, 0x4724, {0xab, 0xaa, 0x23, 0xa6, 0xde, 0x1b, 0xe0, 0x90}};

enum class Property : std::uint32_t {
    InstanceLuid       = 0,
    HardwareId         = 3,
    SharedSystemMemory = 9,
    IsHardware         = 11,
};
enum class State : std::uint32_t { AdapterMemoryBudget = 1 };
enum class SegmentGroup : std::uint32_t { Local = 0, NonLocal = 1 };

struct Luid {
    std::uint32_t low;
    std::int32_t high;
};
struct HardwareId {
    std::uint32_t vendor;
    std::uint32_t device;
    std::uint32_t subsystem;
    std::uint32_t revision;
};
struct BudgetQuery {
    std::uint32_t node;
    SegmentGroup group;
};
struct Budget {
    std::uint64_t budget;
    std::uint64_t usage;
    std::uint64_t available_for_reservation;
    std::uint64_t reservation;
};

struct Unknown {
    virtual HResult query_interface(const Guid& id, void** object) = 0;
    virtual std::uint32_t add_ref()                                 = 0;
    virtual std::uint32_t release()                                 = 0;
};

struct Adapter : Unknown {
    virtual bool is_valid()                                                   = 0;
    virtual bool is_attribute_supported(const Guid& attribute)                = 0;
    virtual bool is_property_supported(Property property)                     = 0;
    virtual HResult get_property(Property property, std::size_t size, void* data) = 0;
    virtual HResult get_property_size(Property property, std::size_t* size)   = 0;
    virtual bool is_query_state_supported(State state)                        = 0;
    virtual HResult query_state(State state, std::size_t input_size, const void* input, std::size_t output_size,
                                void* output)                                 = 0;
};

struct AdapterList : Unknown {
    virtual HResult get_adapter(std::uint32_t index, const Guid& id, void** adapter) = 0;
    virtual std::uint32_t get_adapter_count()                                       = 0;
};

struct AdapterFactory : Unknown {
    virtual HResult create_adapter_list(std::uint32_t attributes, const Guid* filter, const Guid& id,
                                        void** list)                                          = 0;
    virtual HResult get_adapter_by_luid(const Luid& luid, const Guid& id, void** adapter) = 0;
};

constexpr std::uint32_t kNvidiaVendor = 0x10de;

bool succeeded(HResult status) { return status >= 0; }

template <class T>
void release(T*& object) noexcept {
    if (object != nullptr) { object->release(); }
    object = nullptr;
}

bool nvidia_hardware(Adapter& adapter) {
    bool hardware = false;
    HardwareId id{};
    return adapter.is_valid() && succeeded(adapter.get_property(Property::IsHardware, sizeof(hardware), &hardware)) &&
           hardware && succeeded(adapter.get_property(Property::HardwareId, sizeof(id), &id)) &&
           id.vendor == kNvidiaVendor;
}

// The adapter with `luid`, else the only hardware NVIDIA compute adapter; null otherwise.
Adapter* find_adapter(AdapterFactory& factory, const unsigned char* luid) {
    if (luid != nullptr) {
        Luid wanted{};
        std::memcpy(&wanted, luid, sizeof(wanted));
        void* found = nullptr;
        if (succeeded(factory.get_adapter_by_luid(wanted, kAdapterId, &found)) && found != nullptr) {
            auto* adapter = static_cast<Adapter*>(found);
            if (nvidia_hardware(*adapter)) { return adapter; }
            release(adapter);
        }
    }
    void* listed = nullptr;
    if (!succeeded(factory.create_adapter_list(1, &kCoreComputeAttribute, kAdapterListId, &listed)) ||
        listed == nullptr) {
        return nullptr;
    }
    auto* list       = static_cast<AdapterList*>(listed);
    Adapter* match   = nullptr;
    bool ambiguous   = false;
    const std::uint32_t count = list->get_adapter_count();
    for (std::uint32_t i = 0; i < count && !ambiguous; ++i) {
        void* found = nullptr;
        if (!succeeded(list->get_adapter(i, kAdapterId, &found)) || found == nullptr) { continue; }
        auto* adapter = static_cast<Adapter*>(found);
        if (!nvidia_hardware(*adapter)) {
            release(adapter);
        } else if (match == nullptr) {
            match = adapter;
        } else {
            release(adapter);
            ambiguous = true;
        }
    }
    release(list);
    if (ambiguous) { release(match); }
    return match;
}

} // namespace

std::unique_ptr<DxcoreAdapter> DxcoreAdapter::open(const unsigned char* luid) {
    if (::access("/dev/dxg", F_OK) != 0) { return nullptr; }
    void* library = ::dlopen("libdxcore.so", RTLD_LAZY | RTLD_LOCAL);
    if (library == nullptr) { return nullptr; }
    using CreateFactory = HResult (*)(const Guid&, void**);
    const auto create   = reinterpret_cast<CreateFactory>(::dlsym(library, "DXCoreCreateAdapterFactory"));
    void* created       = nullptr;
    if (create == nullptr || !succeeded(create(kAdapterFactoryId, &created)) || created == nullptr) {
        ::dlclose(library);
        return nullptr;
    }
    auto* factory    = static_cast<AdapterFactory*>(created);
    Adapter* adapter = find_adapter(*factory, luid);
    release(factory);
    std::uint64_t shared = 0;
    if (adapter == nullptr || !succeeded(adapter->get_property(Property::SharedSystemMemory, sizeof(shared), &shared)) ||
        shared == 0) {
        release(adapter);
        ::dlclose(library);
        return nullptr;
    }
    std::unique_ptr<DxcoreAdapter> out(new DxcoreAdapter());
    out->library_ = library;
    out->adapter_ = adapter;
    out->shared_  = shared;
    return out;
}

DxcoreAdapter::~DxcoreAdapter() {
    auto* adapter = static_cast<Adapter*>(adapter_);
    release(adapter);
    if (library_ != nullptr) { ::dlclose(library_); }
}

bool DxcoreAdapter::nonlocal_usage(std::uint64_t& usage) const {
    const BudgetQuery query{0, SegmentGroup::NonLocal};
    Budget budget{};
    if (!succeeded(static_cast<Adapter*>(adapter_)->query_state(State::AdapterMemoryBudget, sizeof(query), &query,
                                                                  sizeof(budget), &budget))) {
        return false;
    }
    usage = budget.usage;
    return true;
}

} // namespace infernix
