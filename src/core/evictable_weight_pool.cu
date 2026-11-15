#include "core/evictable_weight_pool.h"

#include "core/cuda_vmm.h"
#include "core/device.h"

#include <cuda.h>

#include <algorithm>
#include <chrono>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace infernix {
namespace {

using Clock = std::chrono::steady_clock;

#define INFERNIX_CU_CHECK(expr) ::infernix::vmm::check((expr), #expr)

std::size_t align_up(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

} // namespace

struct EvictableWeightPool::Impl {
    Config config{};
    std::size_t granularity     = 0;
    std::size_t arena_reserved  = 0;
    std::size_t overlay_reserved = 0;
    std::size_t tail_begin      = 0;   // arena offset of the first evictable chunk
    CUdeviceptr home            = 0;
    CUdeviceptr overlay         = 0;
    // One handle per region piece: [0, tail_begin) in large pieces, the tail in
    // kChunkBytes chunks. Offsets are arena offsets; the vectors are parallel.
    std::vector<CUmemGenericAllocationHandle> handles;
    std::vector<std::size_t> offsets;
    std::vector<std::size_t> sizes;
    std::size_t first_tail_piece = 0;
    std::size_t evicted_pieces   = 0;   // count of evicted pieces at the vector end
    // Where each piece's handle is mapped, so that a constructor or transaction that fails part-way
    // releases exactly what exists (the device memory behind a handle is held until it is released).
    enum class Mapping : std::uint8_t { None, Home, Overlay };
    std::vector<Mapping> mapping;
    std::vector<std::uint8_t> created;      // the piece's handle exists
    std::size_t overlay_first = 0;          // the piece mapped at the overlay's start (an open or failed eviction)
    std::unique_ptr<PinnedHostBuffer> mirror;
    double last_evict_seconds   = 0.0;
    double last_restore_seconds = 0.0;

    Impl() = default;
    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;

    ~Impl() {
        for (std::size_t piece = 0; piece < created.size(); ++piece) {
            if (mapping[piece] == Mapping::Home) {
                (void)cuMemUnmap(home + offsets[piece], sizes[piece]);
            } else if (mapping[piece] == Mapping::Overlay) {
                (void)cuMemUnmap(overlay + (piece - overlay_first) * kChunkBytes, sizes[piece]);
            }
            if (created[piece] != 0) { (void)cuMemRelease(handles[piece]); }
        }
        if (home != 0) { (void)cuMemAddressFree(home, arena_reserved); }
        if (overlay != 0) { (void)cuMemAddressFree(overlay, overlay_reserved); }
    }

    void map_home(std::size_t piece) {
        INFERNIX_CU_CHECK(cuMemMap(home + offsets[piece], sizes[piece], 0, handles[piece], 0));
        mapping[piece] = Mapping::Home;
        set_access(home + offsets[piece], sizes[piece]);
    }

    void unmap(std::size_t piece) {
        const CUdeviceptr at = mapping[piece] == Mapping::Home ? home + offsets[piece]
                                                               : overlay + (piece - overlay_first) * kChunkBytes;
        INFERNIX_CU_CHECK(cuMemUnmap(at, sizes[piece]));
        mapping[piece] = Mapping::None;
    }

    void map_overlay(std::size_t piece) {
        const CUdeviceptr at = overlay + (piece - overlay_first) * kChunkBytes;
        INFERNIX_CU_CHECK(cuMemMap(at, sizes[piece], 0, handles[piece], 0));
        mapping[piece] = Mapping::Overlay;
        set_access(at, sizes[piece]);
    }

    void set_access(CUdeviceptr va, std::size_t bytes) { vmm::set_access(va, bytes, config.device); }
};

bool EvictableWeightPool::supported(int device) { return vmm::supported(device); }

std::size_t EvictableWeightPool::mapped_bytes(std::size_t arena_bytes) noexcept {
    return align_up(arena_bytes, kChunkBytes);
}

std::size_t EvictableWeightPool::mirror_bytes(std::size_t arena_bytes, std::size_t evictable_tail_bytes) noexcept {
    if (evictable_tail_bytes == 0 || evictable_tail_bytes > arena_bytes) { return 0; }
    // The first evictable chunk (the constructor's tail_begin) to the arena's end.
    const std::size_t tail_begin = align_up(arena_bytes - evictable_tail_bytes, kChunkBytes);
    return tail_begin < arena_bytes ? arena_bytes - tail_begin : 0;
}

EvictableWeightPool::EvictableWeightPool(const Config& config) : impl_(std::make_unique<Impl>()) {
    if (config.arena_bytes == 0) {
        throw std::invalid_argument("evictable pool arena must not be empty");
    }
    if (config.evictable_tail_bytes == 0 || config.evictable_tail_bytes > config.arena_bytes) {
        throw std::invalid_argument("evictable pool tail must be a nonempty arena suffix");
    }
    Impl& impl  = *impl_;
    impl.config = config;

    const CUmemAllocationProp prop = vmm::device_prop(config.device);
    impl.granularity               = vmm::granularity(config.device);
    if (kChunkBytes % impl.granularity != 0) {
        throw std::runtime_error("evictable pool chunk is not a multiple of the VMM granularity");
    }

    impl.arena_reserved = mapped_bytes(config.arena_bytes);
    // Chunk-align INTO the evictable suffix so no chunk ever covers non-evictable bytes.
    impl.tail_begin = align_up(config.arena_bytes - config.evictable_tail_bytes, kChunkBytes);
    // Address space is free: reserve overlay VA for the whole evictable tail so
    // a window may borrow any extent the ladder can supply, regardless of the
    // planning hint in config.overlay_bytes.
    impl.overlay_reserved = impl.arena_reserved - impl.tail_begin;

    INFERNIX_CU_CHECK(cuMemAddressReserve(&impl.home, impl.arena_reserved, kChunkBytes, 0, 0));
    INFERNIX_CU_CHECK(cuMemAddressReserve(&impl.overlay, impl.overlay_reserved, kChunkBytes, 0, 0));

    // Non-evictable prefix in large pieces, evictable tail in chunks.
    constexpr std::size_t kPrefixPiece = 1024ULL * 1024ULL * 1024ULL;
    std::size_t offset                 = 0;
    while (offset < impl.tail_begin) {
        const std::size_t piece = std::min(kPrefixPiece, impl.tail_begin - offset);
        impl.offsets.push_back(offset);
        impl.sizes.push_back(piece);
        offset += piece;
    }
    impl.first_tail_piece = impl.offsets.size();
    while (offset < impl.arena_reserved) {
        impl.offsets.push_back(offset);
        impl.sizes.push_back(kChunkBytes);
        offset += kChunkBytes;
    }

    // A step that fails (the device has no memory for a piece) throws out of this constructor; the Impl's
    // destructor then releases the handles, mappings and address ranges made so far.
    impl.handles.resize(impl.offsets.size());
    impl.mapping.assign(impl.offsets.size(), Impl::Mapping::None);
    impl.created.assign(impl.offsets.size(), 0);
    for (std::size_t piece = 0; piece < impl.offsets.size(); ++piece) {
        INFERNIX_CU_CHECK(cuMemCreate(&impl.handles[piece], impl.sizes[piece], &prop, 0));
        impl.created[piece] = 1;
        impl.map_home(piece);
    }
}

// The Impl releases its handles, mappings and address ranges.
EvictableWeightPool::~EvictableWeightPool() = default;

EvictableWeightPool::EvictableWeightPool(EvictableWeightPool&&) noexcept            = default;
EvictableWeightPool& EvictableWeightPool::operator=(EvictableWeightPool&&) noexcept = default;

DeviceSpan EvictableWeightPool::arena() const noexcept {
    return DeviceSpan{reinterpret_cast<void*>(impl_->home), impl_->config.arena_bytes};
}

std::size_t EvictableWeightPool::evictable_tail_bytes() const noexcept {
    return impl_->arena_reserved - impl_->tail_begin;
}

std::size_t EvictableWeightPool::overlay_capacity() const noexcept {
    return impl_->overlay_reserved;
}

void EvictableWeightPool::capture_tail_mirror(cudaStream_t stream) {
    Impl& impl = *impl_;
    if (impl.mirror != nullptr) {
        throw std::logic_error("evictable pool tail mirror was already captured");
    }
    const std::size_t mirrored = mirror_bytes(impl.config.arena_bytes, impl.config.evictable_tail_bytes);
    impl.mirror                = std::make_unique<PinnedHostBuffer>(mirrored);
    CUDA_CHECK(cudaMemcpyAsync(impl.mirror->data(),
                               reinterpret_cast<void*>(impl.home + impl.tail_begin), mirrored,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

std::byte* EvictableWeightPool::evict(std::size_t bytes, std::size_t* mapped_bytes) {
    Impl& impl = *impl_;
    if (impl.mirror == nullptr) {
        throw std::logic_error("evictable pool has no tail mirror; capture it after load");
    }
    if (impl.evicted_pieces != 0) {
        throw std::logic_error("evictable pool transaction is already open");
    }
    if (bytes == 0) { throw std::invalid_argument("evictable pool cannot evict zero bytes"); }
    const std::size_t chunks = (bytes + kChunkBytes - 1) / kChunkBytes;
    const std::size_t extent = chunks * kChunkBytes;
    if (extent > impl.overlay_reserved || extent > impl.arena_reserved - impl.tail_begin) {
        throw std::invalid_argument("evictable pool request exceeds tail or overlay capacity: " +
                                    std::to_string(bytes) + " bytes");
    }
    const auto start        = Clock::now();
    const std::size_t first = impl.handles.size() - chunks;
    impl.overlay_first      = first;
    try {
        for (std::size_t piece = first; piece < impl.handles.size(); ++piece) {
            impl.unmap(piece);
            impl.map_overlay(piece);
        }
    } catch (...) {
        // No transaction opened: every piece goes home again, its memory (and so the weights in it) untouched.
        // A piece that cannot be mapped home stays unmapped for the destructor; the caller sees the first error.
        for (std::size_t piece = first; piece < impl.handles.size(); ++piece) {
            try {
                if (impl.mapping[piece] == Impl::Mapping::Overlay) { impl.unmap(piece); }
                if (impl.mapping[piece] == Impl::Mapping::None) { impl.map_home(piece); }
            } catch (...) {}
        }
        throw;
    }
    impl.evicted_pieces     = chunks;
    impl.last_evict_seconds = std::chrono::duration<double>(Clock::now() - start).count();
    if (mapped_bytes != nullptr) { *mapped_bytes = extent; }
    return reinterpret_cast<std::byte*>(impl.overlay);
}

void EvictableWeightPool::restore(cudaStream_t stream) {
    Impl& impl = *impl_;
    if (impl.evicted_pieces == 0) { return; }
    const auto start         = Clock::now();
    const std::size_t chunks = impl.evicted_pieces;
    const std::size_t first  = impl.handles.size() - chunks;
    for (std::size_t piece = first; piece < impl.handles.size(); ++piece) {
        // A restore that failed part-way and is called again finds some pieces home already.
        if (impl.mapping[piece] == Impl::Mapping::Overlay) { impl.unmap(piece); }
        if (impl.mapping[piece] == Impl::Mapping::None) { impl.map_home(piece); }
    }
    // Only bytes below arena_bytes ever hold weights; the aligned slack stays untouched.
    const std::size_t evicted_begin = impl.offsets[first];
    const std::size_t evicted_end   = std::min(impl.config.arena_bytes, impl.arena_reserved);
    if (evicted_end > evicted_begin) {
        const std::size_t upload = evicted_end - evicted_begin;
        CUDA_CHECK(cudaMemcpyAsync(
            reinterpret_cast<void*>(impl.home + evicted_begin),
            static_cast<const std::byte*>(impl.mirror->data()) +
                (evicted_begin - impl.tail_begin),
            upload, cudaMemcpyHostToDevice, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    impl.evicted_pieces       = 0;
    impl.last_restore_seconds = std::chrono::duration<double>(Clock::now() - start).count();
}

bool EvictableWeightPool::evicted() const noexcept { return impl_->evicted_pieces != 0; }

double EvictableWeightPool::last_evict_seconds() const noexcept {
    return impl_->last_evict_seconds;
}

double EvictableWeightPool::last_restore_seconds() const noexcept {
    return impl_->last_restore_seconds;
}

} // namespace infernix