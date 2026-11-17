#pragma once

#include "runtime/prefix_cache/prefix_index.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace infernix::runtime::prefix_cache {

// Move-only holds on what PrefixCacheIndex lends: node pins, node retentions, snapshot pins, path pins, staging
// Device slots, Host image reservations and the backend's Device blocks. A hold gives its resource back when it
// dies still holding it, so a step that throws between an acquisition and the hand-off to the resource's owner (a
// transfer batch, a binding transaction, a lane, a publication) leaks nothing; release() hands the resource over.
// Holds cover only that window: the owners release in their own order (device waits first), by hand.
//
// The give-back calls are host bookkeeping. One that throws is an index invariant failure, which terminates from a
// destructor as it does in abort_hybrid_binding.

// A batch that records what it acquires (a restore's Filling nodes and pins, a write's nodes) records it before the
// acquisition, so its abort never misses one: this makes the next push_back onto `list` unable to throw. It grows
// geometrically as push_back does; an exact reserve(size() + 1) would copy the whole list for every element.
template <class T>
void reserve_for_push(std::vector<T>& list) {
    if (list.size() == list.capacity()) { list.reserve(std::max<std::size_t>(8, 2 * list.capacity())); }
}

template <class Ref, void (PrefixCacheIndex::*Take)(Ref), void (PrefixCacheIndex::*GiveBack)(Ref)>
class IndexHold {
public:
    IndexHold() noexcept = default;
    // Takes `ref` from the index (Take), or adopts one the caller took when the hold has no Take.
    IndexHold(PrefixCacheIndex& index, Ref ref) : ref_(ref) {
        if constexpr (Take != nullptr) { (index.*Take)(ref); }
        index_ = &index;
    }
    IndexHold(IndexHold&& other) noexcept : index_(std::exchange(other.index_, nullptr)), ref_(other.ref_) {}
    IndexHold& operator=(IndexHold&& other) noexcept {
        if (this != &other) {
            reset();
            index_ = std::exchange(other.index_, nullptr);
            ref_   = other.ref_;
        }
        return *this;
    }
    IndexHold(const IndexHold&)            = delete;
    IndexHold& operator=(const IndexHold&) = delete;
    ~IndexHold() { reset(); }

    [[nodiscard]] bool held() const noexcept { return index_ != nullptr; }
    [[nodiscard]] Ref get() const noexcept { return ref_; }
    // Hands the resource to its owner: the hold gives nothing back.
    Ref release() noexcept {
        index_ = nullptr;
        return ref_;
    }
    void reset() noexcept {
        if (PrefixCacheIndex* index = std::exchange(index_, nullptr)) { (index->*GiveBack)(ref_); }
    }

private:
    PrefixCacheIndex* index_ = nullptr;
    Ref ref_{};
};

// A transfer's hold on one node's Device copy (pin_node).
using NodePin = IndexHold<NodeRef, &PrefixCacheIndex::pin_node, &PrefixCacheIndex::unpin_node>;
// A holder's retention of a node and its Host copy (retain_node).
using NodeRetention = IndexHold<NodeRef, &PrefixCacheIndex::retain_node, &PrefixCacheIndex::unretain_node>;
// A transfer's pin on a snapshot and its Device tail (pin_snapshot).
using SnapshotPin = IndexHold<SnapshotRef, &PrefixCacheIndex::pin_snapshot, &PrefixCacheIndex::unpin_snapshot>;
// A staging Device slot from acquire_device_slot (adopted), until publish_snapshot consumes it.
using DeviceSlotHold = IndexHold<std::uint32_t, nullptr, &PrefixCacheIndex::release_device_slot>;

// Pins on a root path (acquire_path). The hold views the caller's path, which outlives it.
class PathPin {
public:
    PathPin() noexcept = default;
    PathPin(PrefixCacheIndex& index, std::span<const NodeRef> path) : path_(path) {
        if (!path_.empty()) {
            index.acquire_path(path_);
            index_ = &index;
        }
    }
    PathPin(PathPin&& other) noexcept : index_(std::exchange(other.index_, nullptr)), path_(other.path_) {}
    PathPin& operator=(PathPin&& other) noexcept {
        if (this != &other) {
            reset();
            index_ = std::exchange(other.index_, nullptr);
            path_  = other.path_;
        }
        return *this;
    }
    PathPin(const PathPin&)            = delete;
    PathPin& operator=(const PathPin&) = delete;
    ~PathPin() { reset(); }

    // Hands the pins to their owner, which releases them (release_path).
    void release() noexcept { index_ = nullptr; }
    void reset() noexcept {
        if (PrefixCacheIndex* index = std::exchange(index_, nullptr)) { index->release_path(path_); }
    }

private:
    PrefixCacheIndex* index_ = nullptr;
    std::span<const NodeRef> path_;
};

// Host slabs reserved for an image being copied before its snapshot exists (reserve_host_image), until
// publish_host_snapshot consumes them.
class HostImageHold {
public:
    HostImageHold() noexcept = default;
    HostImageHold(PrefixCacheIndex& index, HostImageReservation&& image) noexcept
        : index_(&index), image_(std::move(image)) {}
    HostImageHold(HostImageHold&& other) noexcept
        : index_(std::exchange(other.index_, nullptr)), image_(std::move(other.image_)) {}
    HostImageHold& operator=(HostImageHold&& other) noexcept {
        if (this != &other) {
            reset();
            index_ = std::exchange(other.index_, nullptr);
            image_ = std::move(other.image_);
        }
        return *this;
    }
    HostImageHold(const HostImageHold&)            = delete;
    HostImageHold& operator=(const HostImageHold&) = delete;
    ~HostImageHold() { reset(); }

    [[nodiscard]] const HostImageReservation& get() const noexcept { return image_; }
    [[nodiscard]] HostImageReservation release() noexcept {
        index_ = nullptr;
        return std::move(image_);
    }
    void reset() noexcept {
        if (PrefixCacheIndex* index = std::exchange(index_, nullptr)) { index->release_host_image(std::move(image_)); }
        image_ = {};
    }

private:
    PrefixCacheIndex* index_ = nullptr;
    HostImageReservation image_;
};

// A Device block id the backend allocated (a snapshot tail), until a publication takes it.
class DeviceBlockHold {
public:
    DeviceBlockHold() noexcept = default;
    DeviceBlockHold(PrefixIndexBackend& backend, std::uint32_t id) noexcept : backend_(&backend), id_(id) {}
    DeviceBlockHold(DeviceBlockHold&& other) noexcept : backend_(std::exchange(other.backend_, nullptr)), id_(other.id_) {}
    DeviceBlockHold& operator=(DeviceBlockHold&& other) noexcept {
        if (this != &other) {
            reset();
            backend_ = std::exchange(other.backend_, nullptr);
            id_      = other.id_;
        }
        return *this;
    }
    DeviceBlockHold(const DeviceBlockHold&)            = delete;
    DeviceBlockHold& operator=(const DeviceBlockHold&) = delete;
    ~DeviceBlockHold() { reset(); }

    [[nodiscard]] bool held() const noexcept { return backend_ != nullptr; }
    [[nodiscard]] std::uint32_t get() const noexcept { return id_; }
    std::uint32_t release() noexcept {
        backend_ = nullptr;
        return id_;
    }
    void reset() noexcept {
        if (PrefixIndexBackend* backend = std::exchange(backend_, nullptr)) { backend->release_device_block(id_); }
    }

private:
    PrefixIndexBackend* backend_ = nullptr;
    std::uint32_t id_            = kNoId;
};

} // namespace infernix::runtime::prefix_cache
