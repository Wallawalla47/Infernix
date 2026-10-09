#pragma once

#include "models/qwen3_5/program/storage/logical_kv_store.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace infernix::models::qwen3_5::detail {

class KVAddressSpaceStore;

class KVAddressSpaceHandle {
public:
    KVAddressSpaceHandle() noexcept = default;

    [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

    [[nodiscard]] friend bool operator==(KVAddressSpaceHandle,
                                         KVAddressSpaceHandle) noexcept = default;

private:
    KVAddressSpaceHandle(const KVAddressSpaceStore* owner, std::uint32_t index,
                         std::uint32_t generation) noexcept
        : owner_(owner), index_(index), generation_(generation) {}

    const KVAddressSpaceStore* owner_ = nullptr;
    std::uint32_t index_              = 0;
    std::uint32_t generation_         = 0;

    friend class KVAddressSpaceStore;
};

class KVActivationReservation {
public:
    KVActivationReservation() noexcept = default;

    KVActivationReservation(KVActivationReservation&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), address_(other.address_),
          growth_pages_(other.growth_pages_), page_reservation_(std::move(other.page_reservation_)),
          row_(std::move(other.row_)) {}

    KVActivationReservation& operator=(KVActivationReservation&&)      = delete;
    KVActivationReservation(const KVActivationReservation&)            = delete;
    KVActivationReservation& operator=(const KVActivationReservation&) = delete;

private:
    KVActivationReservation(KVAddressSpaceStore& owner, KVAddressSpaceHandle address,
                            std::uint32_t growth_pages, DeviceKVPageReservation&& page_reservation,
                            KVExecutionRowLease&& row) noexcept
        : owner_(&owner), address_(address), growth_pages_(growth_pages),
          page_reservation_(std::move(page_reservation)), row_(std::move(row)) {}

    KVAddressSpaceStore* owner_ = nullptr;
    KVAddressSpaceHandle address_;
    std::uint32_t growth_pages_ = 0;
    DeviceKVPageReservation page_reservation_;
    std::optional<KVExecutionRowLease> row_;

    friend class KVAddressSpaceStore;
};

// Hybrid prefix cache activation of an empty address space from cached pages owned by the prefix
// index (docs/maintainer/hybrid-prefix-cache-spec.md §6.4). Full pages are shared by reference; a
// partial tail is copied by the caller into a private destination page before commit.
class KVPagePrefixForkReservation {
public:
    KVPagePrefixForkReservation() noexcept = default;
    ~KVPagePrefixForkReservation();

    KVPagePrefixForkReservation(KVPagePrefixForkReservation&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), destination_(other.destination_),
          full_pages_(std::move(other.full_pages_)), tail_source_(other.tail_source_),
          tail_columns_(other.tail_columns_), growth_pages_(other.growth_pages_),
          page_reservation_(std::move(other.page_reservation_)), row_(std::move(other.row_)),
          tail_destination_(std::exchange(other.tail_destination_, std::nullopt)) {}

    KVPagePrefixForkReservation& operator=(KVPagePrefixForkReservation&&)      = delete;
    KVPagePrefixForkReservation(const KVPagePrefixForkReservation&)            = delete;
    KVPagePrefixForkReservation& operator=(const KVPagePrefixForkReservation&) = delete;

    [[nodiscard]] bool needs_tail_copy() const noexcept { return tail_columns_ != 0; }

private:
    KVAddressSpaceStore* owner_ = nullptr;
    KVAddressSpaceHandle destination_;
    std::vector<LogicalKVPageHandle> full_pages_;
    std::optional<LogicalKVPageHandle> tail_source_;
    std::uint32_t tail_columns_ = 0;
    std::uint32_t growth_pages_ = 0;
    DeviceKVPageReservation page_reservation_;
    std::optional<KVExecutionRowLease> row_;
    std::optional<LogicalKVPageHandle> tail_destination_;

    friend class KVAddressSpaceStore;
};

// A logical page occurs once in an address and keeps its position in prefix aliases.
// Growth/COW introduce fresh handles; truncation removes a suffix without moving pages.
class KVAddressSpaceStore {
public:
    KVAddressSpaceStore(LogicalKVPageStore& pages, KVExecutionTablePool& tables,
                        std::uint32_t address_capacity, std::uint32_t page_capacity)
        : pages_(&pages), tables_(&tables), page_capacity_(page_capacity),
          addresses_(address_capacity), free_(address_capacity), free_count_(address_capacity) {
        if (address_capacity == 0 || page_capacity == 0 ||
            page_capacity != tables.logical_page_capacity()) {
            throw std::invalid_argument("KV address-space geometry is invalid");
        }
        for (std::uint32_t index = 0; index < address_capacity; ++index) {
            free_[index] = address_capacity - 1U - index;
        }
        publish_scratch_.reserve(page_capacity_);
        materialization_scratch_.reserve(page_capacity_);
        while (directory_capacity_ < page_capacity_) { directory_capacity_ *= 2; }
    }

    KVAddressSpaceStore(const KVAddressSpaceStore&)            = delete;
    KVAddressSpaceStore& operator=(const KVAddressSpaceStore&) = delete;
    KVAddressSpaceStore(KVAddressSpaceStore&&)                 = delete;
    KVAddressSpaceStore& operator=(KVAddressSpaceStore&&)      = delete;

    [[nodiscard]] std::uint32_t capacity() const noexcept {
        return static_cast<std::uint32_t>(addresses_.size());
    }

    [[nodiscard]] std::uint32_t occupied() const noexcept { return capacity() - free_count_; }

    [[nodiscard]] std::optional<KVAddressSpaceHandle>
    create_active(std::uint32_t growth_pages, std::int32_t execution_row, cudaStream_t stream) {
        if (growth_pages > page_capacity_) { return std::nullopt; }
        std::optional<KVAddressSpaceHandle> handle = create_inactive();
        if (!handle) { return std::nullopt; }
        try {
            activate(*handle, growth_pages, execution_row, stream);
            return handle;
        } catch (...) {
            (void)release(*handle);
            throw;
        }
    }

    [[nodiscard]] std::optional<KVAddressSpaceHandle> create_inactive() noexcept {
        if (free_count_ == 0) { return std::nullopt; }
        const std::uint32_t index = free_[--free_count_];
        Address& address          = addresses_[index];
        if (address.occupied) {
            free_[free_count_++] = index;
            return std::nullopt;
        }
        address.occupied = true;
        return KVAddressSpaceHandle(this, index, address.generation);
    }

    [[nodiscard]] bool valid(KVAddressSpaceHandle handle) const noexcept {
        return handle.owner_ == this && handle.index_ < addresses_.size() &&
               addresses_[handle.index_].occupied &&
               addresses_[handle.index_].generation == handle.generation_;
    }

    void activate(KVAddressSpaceHandle handle, std::uint32_t growth_pages,
                  std::int32_t execution_row, cudaStream_t stream) {
        auto reservation = prepare_activation(handle, growth_pages, execution_row);
        commit_activation(std::move(reservation), stream);
    }

    [[nodiscard]] KVActivationReservation prepare_activation(KVAddressSpaceHandle handle,
                                                             std::uint32_t growth_pages,
                                                             std::int32_t execution_row) {
        Address& address = require(handle);
        if (address.active || address.row || address.reservation.valid() ||
            growth_pages > page_capacity_ - address.page_count) {
            throw std::logic_error("KV address space is not reservable for activation");
        }
        DeviceKVPageReservation reservation = pages_->physical_pool().make_empty_reservation();
        pages_->physical_pool().resize_reservation(reservation, growth_pages);
        KVExecutionRowLease row = tables_->acquire(execution_row);
        return KVActivationReservation(*this, handle, growth_pages, std::move(reservation),
                                       std::move(row));
    }

    void commit_activation(KVActivationReservation&& activation, cudaStream_t stream) {
        if (activation.owner_ != this) {
            throw std::logic_error("KV activation reservation belongs to another store");
        }
        if (!activation.row_) {
            throw std::logic_error("KV activation reservation has no execution row");
        }
        if (!valid(activation.address_)) {
            throw std::logic_error("KV activation reservation address is stale");
        }
        Address& address = require(activation.address_);
        if (address.active || address.row || address.reservation.valid() ||
            activation.growth_pages_ > page_capacity_ - address.page_count) {
            throw std::logic_error("KV activation destination changed after reservation");
        }
        const std::uint32_t expected = activation.growth_pages_;
        if (!activation.page_reservation_.belongs_to(pages_->physical_pool()) ||
            activation.page_reservation_.pages() != expected) {
            throw std::logic_error("KV activation capacity reservation changed");
        }
        for (std::uint32_t page = 0; page < address.page_count; ++page) {
            const LogicalKVPageHandle logical = membership(address, page);
            if (!pages_->device_resident(logical) || (pages_->address_references(logical) == 1 &&
                                                      !pages_->can_set_writer(logical, true))) {
                throw std::logic_error("KV activation has an unavailable Device page");
            }
        }
        publish_scratch_.clear();
        for (std::uint32_t page = 0; page < address.page_count; ++page) {
            publish_scratch_.push_back(pages_->physical(membership(address, page)));
        }
        tables_->publish(activation.row_->handle(), 0, publish_scratch_, stream);
        address.reservation = std::move(activation.page_reservation_);
        address.row.emplace(std::move(*activation.row_));
        activation.row_.reset();
        address.active    = true;
        activation.owner_ = nullptr;
        for (std::uint32_t page = 0; page < address.page_count; ++page) {
            const LogicalKVPageHandle logical = membership(address, page);
            if (pages_->address_references(logical) == 1) { pages_->set_writer(logical, true); }
            pages_->retain_active_reference(logical);
        }
    }

    void deactivate(KVAddressSpaceHandle handle) {
        Address& address = require_active(handle);
        for (std::uint32_t page = 0; page < address.page_count; ++page) {
            const LogicalKVPageHandle logical = membership(address, page);
            if (pages_->writer_references(logical) != 0 &&
                !pages_->can_set_writer(logical, false)) {
                throw std::logic_error("KV deactivation has an invalid writer reference");
            }
        }
        for (std::uint32_t page = 0; page < address.page_count; ++page) {
            const LogicalKVPageHandle logical = membership(address, page);
            if (pages_->writer_references(logical) != 0) { pages_->set_writer(logical, false); }
            pages_->release_active_reference(logical);
        }
        address.row.reset();
        address.reservation.release();
        address.active = false;
    }

    // `growth_pages` are reserved beyond the cached prefix (its full pages plus the private tail).
    [[nodiscard]] KVPagePrefixForkReservation prepare_page_prefix_fork(
        KVAddressSpaceHandle destination_handle, std::span<const LogicalKVPageHandle> full_pages,
        std::optional<LogicalKVPageHandle> tail_source, std::uint32_t tail_columns,
        std::uint32_t growth_pages, std::int32_t execution_row) {
        Address& destination = require(destination_handle);
        if (destination.active || destination.row || destination.reservation.valid() ||
            destination.page_count != 0 || destination.committed_frontier != 0) {
            throw std::logic_error("KV cached-prefix destination is not empty");
        }
        const auto page_size = static_cast<std::uint32_t>(kPagedKVPageSize);
        if (tail_columns >= page_size || tail_source.has_value() != (tail_columns != 0)) {
            throw std::invalid_argument("KV cached-prefix tail is inconsistent");
        }
        const auto full                    = static_cast<std::uint32_t>(full_pages.size());
        const std::uint32_t required_pages = full + (tail_columns != 0 ? 1U : 0U);
        if (required_pages == 0 || required_pages > page_capacity_ ||
            growth_pages > page_capacity_ - required_pages) {
            throw std::invalid_argument("KV cached-prefix geometry is invalid");
        }
        for (const LogicalKVPageHandle logical : full_pages) {
            if (!pages_->device_resident(logical) ||
                pages_->committed_columns(logical) != page_size ||
                !pages_->can_pin_source(logical) || !pages_->can_retain_reference(logical, false)) {
                throw std::logic_error("KV cached-prefix page is not stable");
            }
        }
        if (tail_source && (!pages_->device_resident(*tail_source) ||
                            pages_->committed_columns(*tail_source) < tail_columns ||
                            !pages_->can_pin_source(*tail_source))) {
            throw std::logic_error("KV cached-prefix tail source is not stable");
        }
        KVPagePrefixForkReservation fork;
        fork.page_reservation_ = pages_->physical_pool().make_empty_reservation();
        pages_->physical_pool().resize_reservation(fork.page_reservation_,
                                                   growth_pages + (tail_columns != 0 ? 1U : 0U));
        fork.row_.emplace(tables_->acquire(execution_row));
        if (tail_columns != 0) {
            fork.tail_destination_ =
                pages_->materialize_transfer_destination(fork.page_reservation_, tail_columns);
        }
        fork.full_pages_.assign(full_pages.begin(), full_pages.end());
        fork.tail_source_  = tail_source;
        fork.tail_columns_ = tail_columns;
        fork.growth_pages_ = growth_pages;
        fork.destination_  = destination_handle;
        for (const LogicalKVPageHandle logical : full_pages) { pages_->pin_source(logical); }
        if (tail_source) { pages_->pin_source(*tail_source); }
        fork.owner_ = this;
        return fork;
    }

    [[nodiscard]] DeviceKVPageHandle
    page_prefix_fork_tail_source(const KVPagePrefixForkReservation& fork) const {
        require_page_prefix_fork(fork);
        if (!fork.tail_source_) { throw std::logic_error("KV cached-prefix fork has no tail"); }
        return pages_->physical(*fork.tail_source_);
    }

    [[nodiscard]] DeviceKVPageHandle
    page_prefix_fork_tail_destination(const KVPagePrefixForkReservation& fork) const {
        require_page_prefix_fork(fork);
        if (!fork.tail_destination_) {
            throw std::logic_error("KV cached-prefix fork has no tail destination");
        }
        return pages_->physical(*fork.tail_destination_);
    }

    void commit_page_prefix_fork(KVPagePrefixForkReservation&& fork, cudaStream_t stream) {
        require_page_prefix_fork(fork);
        Address& destination               = require(fork.destination_);
        const auto full                    = static_cast<std::uint32_t>(fork.full_pages_.size());
        const std::uint32_t required_pages = full + (fork.tail_columns_ != 0 ? 1U : 0U);
        if (!fork.row_ || destination.active || destination.row ||
            destination.reservation.valid() || destination.page_count != 0 ||
            fork.page_reservation_.pages() != fork.growth_pages_ ||
            (fork.tail_columns_ != 0 &&
             (!fork.tail_destination_ || !pages_->valid(*fork.tail_destination_)))) {
            throw std::logic_error("KV cached-prefix fork changed before publication");
        }
        publish_scratch_.clear();
        for (const LogicalKVPageHandle logical : fork.full_pages_) {
            if (pages_->source_pins(logical) == 0 || !pages_->device_resident(logical)) {
                throw std::logic_error("KV cached-prefix page changed before publication");
            }
            publish_scratch_.push_back(pages_->physical(logical));
        }
        if (fork.tail_destination_) {
            publish_scratch_.push_back(pages_->physical(*fork.tail_destination_));
        }
        // The whole directory is built before any physical ownership moves.
        Directory directory;
        for (std::uint32_t page = 0; page < full; ++page) {
            directory_slot(directory, page) = fork.full_pages_[page];
        }
        if (fork.tail_destination_) { directory_slot(directory, full) = *fork.tail_destination_; }
        tables_->publish(fork.row_->handle(), 0, publish_scratch_, stream);

        for (const LogicalKVPageHandle logical : fork.full_pages_) {
            pages_->retain_reference(logical, false);
            pages_->protect_coverage(logical, static_cast<std::uint32_t>(kPagedKVPageSize));
        }
        if (fork.tail_destination_) {
            pages_->publish_transfer_destination(*fork.tail_destination_, true);
            fork.tail_destination_.reset();
        }
        destination.directory  = std::move(directory);
        destination.page_count = required_pages;
        destination.committed_frontier =
            full * static_cast<std::uint32_t>(kPagedKVPageSize) + fork.tail_columns_;
        destination.reservation = std::move(fork.page_reservation_);
        destination.row.emplace(std::move(*fork.row_));
        fork.row_.reset();
        destination.active = true;
        for (std::uint32_t page = 0; page < required_pages; ++page) {
            pages_->retain_active_reference(membership(destination, page));
        }
        for (const LogicalKVPageHandle logical : fork.full_pages_) {
            pages_->unpin_source(logical);
        }
        if (fork.tail_source_) { pages_->unpin_source(*fork.tail_source_); }
        fork.owner_ = nullptr;
    }

    void abort_page_prefix_fork(KVPagePrefixForkReservation& fork) noexcept {
        if (fork.owner_ != this) { return; }
        for (const LogicalKVPageHandle logical : fork.full_pages_) {
            if (pages_->valid(logical) && pages_->source_pins(logical) != 0) {
                pages_->unpin_source(logical);
            }
        }
        if (fork.tail_source_ && pages_->valid(*fork.tail_source_) &&
            pages_->source_pins(*fork.tail_source_) != 0) {
            pages_->unpin_source(*fork.tail_source_);
        }
        if (fork.tail_destination_) {
            pages_->abort_transfer_destination(*fork.tail_destination_, fork.page_reservation_);
            fork.tail_destination_.reset();
        }
        fork.page_reservation_.release();
        fork.row_.reset();
        fork.owner_ = nullptr;
    }

    // A reservation covers a finite execution or recovery interval. Resizing is atomic at the pool;
    // callers combining several addresses retain their old counts until the group is acquired.
    void reserve_growth(KVAddressSpaceHandle handle, std::uint32_t growth_pages) {
        Address& address = require_active(handle);
        if (growth_pages > page_capacity_ - address.page_count) {
            throw std::invalid_argument("KV growth exceeds address capacity");
        }
        pages_->physical_pool().resize_reservation(address.reservation, growth_pages);
    }

    [[nodiscard]] std::uint32_t growth_pages_for_tokens(KVAddressSpaceHandle handle,
                                                        std::uint32_t tokens) const {
        const Address& address     = require(handle);
        const std::uint32_t target = pages_for_tokens(tokens);
        if (target > page_capacity_) {
            throw std::invalid_argument("KV unit exceeds address capacity");
        }
        return target > address.page_count ? target - address.page_count : 0U;
    }

    void settle_growth(KVAddressSpaceHandle handle, std::uint32_t frontier,
                       std::uint32_t retain_through = 0) {
        if (frontier >= committed_frontier(handle)) { commit_frontier(handle, frontier); }
        destructive_truncate(handle, frontier);
        reserve_growth(handle, growth_pages_for_tokens(handle, retain_through));
    }

    // Coverage is a lower bound. A speculative mapping may already extend beyond this stage's
    // needs; only an explicit truncate releases it, and commit_frontier publishes valid tokens.
    void ensure_mapped_to_tokens(KVAddressSpaceHandle handle, std::uint32_t tokens,
                                 cudaStream_t stream) {
        Address& address           = require_active(handle);
        const std::uint32_t target = pages_for_tokens(tokens);
        if (target > address.page_count + address.reservation.pages()) {
            throw std::invalid_argument(
                "KV coverage exceeds reserved unit growth: tokens=" + std::to_string(tokens) +
                " required_pages=" + std::to_string(target) +
                " mapped_pages=" + std::to_string(address.page_count) +
                " reserved_pages=" + std::to_string(address.reservation.pages()));
        }
        if (target <= address.page_count) { return; }
        const std::uint32_t begin = address.page_count;
        const std::uint32_t count = target - begin;
        // Prepare only the append paths. Existing shared prefix nodes remain immutable.
        for (std::uint32_t page = begin; page < target; ++page) {
            (void)directory_slot(address.directory, page);
        }
        materialization_scratch_.assign(count, LogicalKVPageHandle{});
        std::span<LogicalKVPageHandle> added(materialization_scratch_);
        const std::optional<LogicalKVPageHandle> predecessor =
            begin == 0 ? std::nullopt
                       : std::optional<LogicalKVPageHandle>(membership(address, begin - 1U));
        pages_->materialize(address.reservation, added, predecessor);
        try {
            publish_scratch_.clear();
            for (const LogicalKVPageHandle page : added) {
                publish_scratch_.push_back(pages_->physical(page));
            }
            tables_->publish(address.row->handle(), begin, publish_scratch_, stream);
        } catch (...) {
            for (LogicalKVPageHandle& page : added) {
                pages_->dematerialize(page, address.reservation);
                page = {};
            }
            throw;
        }
        for (std::uint32_t offset = 0; offset < count; ++offset) {
            pages_->retain_active_reference(added[offset]);
            directory_slot(address.directory, begin + offset) = added[offset];
        }
        address.page_count = target;
    }

    void commit_frontier(KVAddressSpaceHandle handle, std::uint32_t frontier) {
        Address& address = require_active(handle);
        if (frontier < address.committed_frontier ||
            pages_for_tokens(frontier) > address.page_count) {
            throw std::invalid_argument("KV committed frontier is invalid");
        }
        if (frontier == address.committed_frontier) { return; }
        const std::uint32_t page_size          = static_cast<std::uint32_t>(kPagedKVPageSize);
        const std::uint32_t first_changed_page = address.committed_frontier / page_size;
        const std::uint32_t final_changed_page = (frontier - 1U) / page_size;
        for (std::uint32_t page = first_changed_page; page <= final_changed_page; ++page) {
            const std::uint32_t begin   = page * static_cast<std::uint32_t>(kPagedKVPageSize);
            const std::uint32_t columns = std::min(page_size, frontier - begin);
            if (columns > pages_->committed_columns(membership(address, page))) {
                pages_->commit_coverage(membership(address, page), columns);
            }
        }
        address.committed_frontier = frontier;
    }

    void destructive_truncate(KVAddressSpaceHandle handle, std::uint32_t frontier) {
        Address& address = require_active(handle);
        if (frontier > address.committed_frontier) {
            throw std::invalid_argument("KV destructive truncate extends the frontier");
        }
        const std::uint32_t target = pages_for_tokens(frontier);
        if (frontier == address.committed_frontier && target == address.page_count) { return; }
        for (std::uint32_t page = target; page < address.page_count; ++page) {
            if (!pages_->can_dematerialize(membership(address, page))) {
                throw std::logic_error("KV truncate would partially release a protected page");
            }
        }
        if (target != 0) {
            const std::uint32_t columns =
                frontier - (target - 1U) * static_cast<std::uint32_t>(kPagedKVPageSize);
            if (columns != pages_->committed_columns(membership(address, target - 1U)) &&
                !pages_->can_destructive_truncate(membership(address, target - 1U), columns)) {
                throw std::logic_error("KV truncate would overwrite protected coverage");
            }
        }
        while (address.page_count > target) {
            const std::uint32_t index = --address.page_count;
            LogicalKVPageHandle page  = membership(address, index);
            pages_->release_active_reference(page);
            pages_->dematerialize(page, address.reservation);
        }
        if (target != 0) {
            const std::uint32_t columns =
                frontier - (target - 1U) * static_cast<std::uint32_t>(kPagedKVPageSize);
            if (columns != pages_->committed_columns(membership(address, target - 1U))) {
                pages_->destructive_truncate(membership(address, target - 1U), columns);
            }
        }
        trim_unique_suffix(address.directory, directory_capacity_, target);
        address.committed_frontier = frontier;
    }

    [[nodiscard]] std::uint32_t mapped_pages(KVAddressSpaceHandle handle) const {
        return require(handle).page_count;
    }

    [[nodiscard]] std::uint32_t reserved_growth_pages(KVAddressSpaceHandle handle) const {
        const Address& address = require(handle);
        return address.reservation.valid() ? address.reservation.pages() : 0U;
    }

    [[nodiscard]] std::uint32_t committed_frontier(KVAddressSpaceHandle handle) const {
        return require(handle).committed_frontier;
    }

    [[nodiscard]] std::int32_t bound_row(KVAddressSpaceHandle handle) const noexcept {
        if (!valid(handle) || !addresses_[handle.index_].row) { return -1; }
        return addresses_[handle.index_].row->row_index();
    }

    [[nodiscard]] bool active(KVAddressSpaceHandle handle) const noexcept {
        return valid(handle) && addresses_[handle.index_].active;
    }

    [[nodiscard]] const KVExecutionRowLease& execution_row(KVAddressSpaceHandle handle) const {
        const Address& address = require(handle);
        if (!address.active || !address.row) {
            throw std::logic_error("KV address space has no execution row");
        }
        return *address.row;
    }

    [[nodiscard]] DeviceKVPageHandle physical_page(KVAddressSpaceHandle handle,
                                                   std::uint32_t logical_page) const {
        const Address& address = require(handle);
        if (logical_page >= address.page_count) {
            throw std::out_of_range("KV logical page is outside the address space");
        }
        return pages_->physical(membership(address, logical_page));
    }

    [[nodiscard]] LogicalKVPageHandle logical_page(KVAddressSpaceHandle handle,
                                                   std::uint32_t logical_page) const {
        const Address& address = require(handle);
        if (logical_page >= address.page_count) {
            throw std::out_of_range("KV logical page is outside the address space");
        }
        return membership(address, logical_page);
    }

    [[nodiscard]] bool can_release(KVAddressSpaceHandle handle) const noexcept {
        if (!valid(handle)) { return false; }
        const Address& address = addresses_[handle.index_];
        if (address.active || address.row || address.reservation.valid()) { return false; }
        for (std::uint32_t page = 0; page < address.page_count; ++page) {
            if (!pages_->can_release_reference(membership(address, page), false)) { return false; }
        }
        return true;
    }

    [[nodiscard]] bool can_release_after_deactivate(KVAddressSpaceHandle handle) const noexcept {
        if (!valid(handle)) { return false; }
        const Address& address = addresses_[handle.index_];
        if (!address.active) { return can_release(handle); }
        if (!address.row) { return false; }
        for (std::uint32_t page = 0; page < address.page_count; ++page) {
            if (!pages_->can_release_reference_after_active_reference(membership(address, page))) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] bool release_after_deactivate(KVAddressSpaceHandle handle) noexcept {
        if (!can_release_after_deactivate(handle)) { return false; }
        try {
            if (addresses_[handle.index_].active) { deactivate(handle); }
        } catch (...) { std::terminate(); }
        if (!release(handle)) { std::terminate(); }
        return true;
    }

    [[nodiscard]] bool release(KVAddressSpaceHandle handle) noexcept {
        if (!can_release(handle)) { return false; }
        Address& address = addresses_[handle.index_];
        for (std::uint32_t page = 0; page < address.page_count; ++page) {
            if (!pages_->release_reference(membership(address, page), false)) { std::terminate(); }
        }
        const std::uint32_t index      = handle.index_;
        const std::uint32_t generation = next_generation(address.generation);
        address                        = Address{};
        address.generation             = generation;
        free_[free_count_++]           = index;
        return true;
    }

private:
    // Address roots share immutable chunks. Nodes hold non-owning logical capabilities; the
    // address store, rather than shared_ptr destruction, owns every physical page reference.
    // Thus dropping an ancestor cannot invalidate descendants. Only paths containing changed
    // membership are copied.
    static constexpr std::uint32_t kDirectoryChunkPages = 32;

    struct DirectoryNode {
        std::shared_ptr<DirectoryNode> left;
        std::shared_ptr<DirectoryNode> right;
        std::array<LogicalKVPageHandle, kDirectoryChunkPages> pages{};
    };

    using Directory = std::shared_ptr<DirectoryNode>;

    struct Address {
        Directory directory;
        std::uint32_t generation         = 1;
        std::uint32_t page_count         = 0;
        std::uint32_t committed_frontier = 0;
        DeviceKVPageReservation reservation;
        std::optional<KVExecutionRowLease> row;
        bool occupied = false;
        bool active   = false;
    };

    [[nodiscard]] static std::uint32_t pages_for_tokens(std::uint32_t tokens) noexcept {
        return tokens == 0 ? 0U : 1U + (tokens - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
    }

    [[nodiscard]] static std::uint32_t next_generation(std::uint32_t generation) noexcept {
        ++generation;
        return generation == 0 ? 1 : generation;
    }

    [[nodiscard]] Address& require(KVAddressSpaceHandle handle) {
        if (!valid(handle)) { throw std::invalid_argument("KV address-space handle is stale"); }
        return addresses_[handle.index_];
    }

    [[nodiscard]] const Address& require(KVAddressSpaceHandle handle) const {
        if (!valid(handle)) { throw std::invalid_argument("KV address-space handle is stale"); }
        return addresses_[handle.index_];
    }

    void require_page_prefix_fork(const KVPagePrefixForkReservation& fork) const {
        if (fork.owner_ != this || !valid(fork.destination_)) {
            throw std::logic_error("KV cached-prefix fork reservation is stale");
        }
    }

    [[nodiscard]] Address& require_active(KVAddressSpaceHandle handle) {
        Address& address = require(handle);
        if (!address.active || !address.row || !address.reservation.valid()) {
            throw std::logic_error("KV address space is not active");
        }
        return address;
    }

    [[nodiscard]] const Address& require_active(KVAddressSpaceHandle handle) const {
        const Address& address = require(handle);
        if (!address.active || !address.row || !address.reservation.valid()) {
            throw std::logic_error("KV address space is not active");
        }
        return address;
    }

    [[nodiscard]] LogicalKVPageHandle membership(const Address& address,
                                                 std::uint32_t page) const noexcept {
        const DirectoryNode* node = address.directory.get();
        std::uint64_t width       = directory_capacity_;
        while (width > kDirectoryChunkPages) {
            width /= 2;
            if (page < width) {
                node = node->left.get();
            } else {
                page -= static_cast<std::uint32_t>(width);
                node = node->right.get();
            }
        }
        return node->pages[page];
    }

    [[nodiscard]] LogicalKVPageHandle& directory_slot(Directory& root, std::uint32_t page) {
        Directory* node     = &root;
        std::uint64_t width = directory_capacity_;
        for (;;) {
            if (!*node) {
                *node = std::make_shared<DirectoryNode>();
            } else if (node->use_count() != 1) {
                *node = std::make_shared<DirectoryNode>(**node);
            }
            if (width == kDirectoryChunkPages) { return (*node)->pages[page]; }
            width /= 2;
            if (page < width) {
                node = &(*node)->left;
            } else {
                page -= static_cast<std::uint32_t>(width);
                node = &(*node)->right;
            }
        }
    }

    static void trim_unique_suffix(Directory& node, std::uint64_t width,
                                   std::uint32_t count) noexcept {
        if (!node || count >= width) { return; }
        if (count == 0) {
            node.reset();
            return;
        }
        // A shared node contains only retained prefix pages: appending a suffix made its
        // changed path private. It needs no pruning when all removed pages lived elsewhere.
        if (node.use_count() != 1) { return; }
        if (width == kDirectoryChunkPages) {
            std::fill(node->pages.begin() + count, node->pages.end(), LogicalKVPageHandle{});
        } else {
            width /= 2;
            if (count <= width) {
                trim_unique_suffix(node->left, width, count);
                node->right.reset();
            } else {
                trim_unique_suffix(node->right, width, count - static_cast<std::uint32_t>(width));
            }
        }
    }

    LogicalKVPageStore* pages_    = nullptr;
    KVExecutionTablePool* tables_ = nullptr;
    std::uint32_t page_capacity_  = 0;
    std::vector<Address> addresses_;
    std::vector<std::uint32_t> free_;
    std::uint64_t directory_capacity_ = kDirectoryChunkPages;
    std::vector<LogicalKVPageHandle> materialization_scratch_;
    std::vector<DeviceKVPageHandle> publish_scratch_;
    std::uint32_t free_count_ = 0;
};

inline KVPagePrefixForkReservation::~KVPagePrefixForkReservation() {
    if (owner_ != nullptr) { owner_->abort_page_prefix_fork(*this); }
}

} // namespace infernix::models::qwen3_5::detail
