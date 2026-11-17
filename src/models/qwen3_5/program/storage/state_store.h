#pragma once

#include "models/qwen3_5/state/state_image.h"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace infernix::models::qwen3_5::detail {

class StateImageStore;

class StateImageHandle {
public:
    StateImageHandle() noexcept = default;

    [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

    [[nodiscard]] friend bool operator==(StateImageHandle, StateImageHandle) noexcept = default;

private:
    StateImageHandle(const StateImageStore* owner, std::uint32_t index,
                     std::uint32_t generation) noexcept
        : owner_(owner), index_(index), generation_(generation) {}

    const StateImageStore* owner_ = nullptr;
    std::uint32_t index_          = 0;
    std::uint32_t generation_     = 0;

    friend class StateImageStore;
};

enum class StateImageRole : std::uint8_t {
    Free,
    // A lane's state, updated in place by execution.
    ActiveMutable,
    // A prefix-cache snapshot: never written again.
    SnapshotImmutable,
    // Reserved for a stream-ordered copy or reset that has not been activated yet.
    ReservedDestination,
};

struct StateImageSelectors {
    std::int32_t source      = -1;
    std::int32_t destination = -1;
};

// Program-private ownership of the Device StateImage slots. Each image owns one slot for its whole
// life; a handle's generation detects reuse of the slot by a later image.
class StateImageStore {
public:
    explicit StateImageStore(qwen3_5::StateImageDevicePool& device)
        : device_(&device), objects_(static_cast<std::size_t>(device.slot_count())),
          free_objects_(objects_.size()), free_count_(static_cast<std::uint32_t>(objects_.size())) {
        if (objects_.empty()) { throw std::invalid_argument("StateImageStore has no slots"); }
        for (std::uint32_t index = 0; index < free_count_; ++index) {
            free_objects_[index] = free_count_ - 1U - index;
        }
    }

    StateImageStore(const StateImageStore&)            = delete;
    StateImageStore& operator=(const StateImageStore&) = delete;
    StateImageStore(StateImageStore&&)                 = delete;
    StateImageStore& operator=(StateImageStore&&)      = delete;

    [[nodiscard]] std::uint32_t device_capacity() const noexcept {
        return static_cast<std::uint32_t>(objects_.size());
    }

    [[nodiscard]] std::uint32_t device_occupied() const noexcept {
        return device_capacity() - free_count_;
    }

    [[nodiscard]] std::optional<StateImageHandle> reserve_destination() noexcept {
        return allocate(StateImageRole::ReservedDestination);
    }

    [[nodiscard]] std::optional<StateImageHandle> reserve_reset(cudaStream_t stream) {
        std::optional<StateImageHandle> handle = allocate(StateImageRole::ActiveMutable);
        if (!handle) { return std::nullopt; }
        try {
            device_->zero_slot(static_cast<std::int32_t>(handle->index_), stream);
        } catch (...) {
            (void)release(*handle);
            throw;
        }
        return handle;
    }

    void activate_reset(StateImageHandle handle, cudaStream_t stream) {
        Object& object = require_role(handle, StateImageRole::ReservedDestination,
                                      "StateImage reset reservation is not activatable");
        device_->zero_slot(static_cast<std::int32_t>(handle.index_), stream);
        object.role = StateImageRole::ActiveMutable;
    }

    // A reserved destination filled by a stream-ordered copy (prefix cache admission and tap
    // capture) becomes the lane's active image or an immutable cached snapshot.
    void activate_copied(StateImageHandle handle) {
        require_role(handle, StateImageRole::ReservedDestination,
                     "StateImage copied destination is not activatable")
            .role = StateImageRole::ActiveMutable;
    }

    void publish_copied_snapshot(StateImageHandle handle) {
        require_role(handle, StateImageRole::ReservedDestination,
                     "StateImage copied destination is not publishable")
            .role = StateImageRole::SnapshotImmutable;
    }

    // A lane's image becomes a snapshot in place (a finishing lane's endpoint), or back.
    void freeze(StateImageHandle handle) {
        require_role(handle, StateImageRole::ActiveMutable, "StateImage active image is not freezable")
            .role = StateImageRole::SnapshotImmutable;
    }

    void thaw(StateImageHandle handle) {
        require_role(handle, StateImageRole::SnapshotImmutable, "StateImage snapshot is not thawable")
            .role = StateImageRole::ActiveMutable;
    }

    [[nodiscard]] bool valid(StateImageHandle handle) const noexcept {
        return handle.owner_ == this && handle.index_ < objects_.size() &&
               objects_[handle.index_].role != StateImageRole::Free &&
               objects_[handle.index_].generation == handle.generation_;
    }

    [[nodiscard]] StateImageRole role(StateImageHandle handle) const {
        if (!valid(handle)) { throw std::invalid_argument("StateImage handle is stale"); }
        return objects_[handle.index_].role;
    }

    [[nodiscard]] std::int32_t physical_slot(StateImageHandle handle) const {
        if (!valid(handle)) { throw std::invalid_argument("StateImage handle is stale"); }
        return static_cast<std::int32_t>(handle.index_);
    }

    [[nodiscard]] bool release(StateImageHandle handle) noexcept {
        if (!valid(handle)) { return false; }
        Object& object = objects_[handle.index_];
        object.role    = StateImageRole::Free;
        if (++object.generation == 0) { ++object.generation; }
        free_objects_[free_count_++] = handle.index_;
        return true;
    }

private:
    struct Object {
        std::uint32_t generation = 1;
        StateImageRole role      = StateImageRole::Free;
    };

    [[nodiscard]] std::optional<StateImageHandle> allocate(StateImageRole role) noexcept {
        if (free_count_ == 0) { return std::nullopt; }
        const std::uint32_t index = free_objects_[--free_count_];
        Object& object            = objects_[index];
        object.role               = role;
        return StateImageHandle(this, index, object.generation);
    }

    [[nodiscard]] Object& require_role(StateImageHandle handle, StateImageRole role,
                                       const char* message) {
        if (!valid(handle) || objects_[handle.index_].role != role) {
            throw std::logic_error(message);
        }
        return objects_[handle.index_];
    }

    qwen3_5::StateImageDevicePool* device_ = nullptr;
    std::vector<Object> objects_;
    std::vector<std::uint32_t> free_objects_;
    std::uint32_t free_count_ = 0;
};

// A slot reserved in a scope that can still fail before its owner (a binding transaction, a pending tap) holds it:
// released when the hold dies still holding it, handed over by release(). The prefix cache's holds
// (runtime/prefix_cache/prefix_holds.h) are the same family.
class StateSlotHold {
public:
    StateSlotHold() noexcept = default;
    StateSlotHold(StateImageStore& store, StateImageHandle handle) noexcept : store_(&store), handle_(handle) {}
    StateSlotHold(StateSlotHold&& other) noexcept
        : store_(std::exchange(other.store_, nullptr)), handle_(other.handle_) {}
    StateSlotHold& operator=(StateSlotHold&& other) noexcept {
        if (this != &other) {
            reset();
            store_  = std::exchange(other.store_, nullptr);
            handle_ = other.handle_;
        }
        return *this;
    }
    StateSlotHold(const StateSlotHold&)            = delete;
    StateSlotHold& operator=(const StateSlotHold&) = delete;
    ~StateSlotHold() { reset(); }

    [[nodiscard]] bool held() const noexcept { return store_ != nullptr; }
    [[nodiscard]] StateImageHandle get() const noexcept { return handle_; }
    StateImageHandle release() noexcept {
        store_ = nullptr;
        return handle_;
    }
    void reset() noexcept {
        if (StateImageStore* store = std::exchange(store_, nullptr)) { (void)store->release(handle_); }
    }

private:
    StateImageStore* store_ = nullptr;
    StateImageHandle handle_;
};

// A lane's state: execution reads and writes the same image.
struct ActiveStateBinding {
    StateImageHandle read;
    StateImageHandle write;
};

} // namespace infernix::models::qwen3_5::detail
