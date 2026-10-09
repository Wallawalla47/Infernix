#include "core/device.h"
#include "models/qwen3_5/program/storage/kv_address_space.h"
#include "models/qwen3_5/program/storage/state_store.h"

#include "models/qwen3_5/state/state_image.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace q36   = infernix::models::qwen3_5;
namespace store = infernix::models::qwen3_5::detail;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver;
}

std::vector<std::int32_t> read_block_table(const infernix::KVExecutionTablePool& tables,
                                           std::int32_t row, std::size_t count) {
    const infernix::Tensor source = tables.matrix().slice(1, row, 1).view(
        {static_cast<std::int32_t>(tables.logical_page_capacity())});
    std::vector<std::int32_t> values(count);
    CUDA_CHECK(cudaMemcpy(values.data(), source.data, values.size() * sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost));
    return values;
}

void test_state_store(infernix::DeviceContext& device) {
    q36::StateImageSpec spec{
        .linear =
            {
                .layers         = 1,
                .conv_channels  = 8,
                .conv_width     = 3,
                .value_heads    = 2,
                .value_head_dim = 4,
                .key_head_dim   = 4,
                .slot_count     = 4,
                .conv_dtype     = infernix::DType::BF16,
            },
        .hidden = 8,
        .dflash_local =
            q36::DFlashLocalStateSpec{.layers = 1, .capacity = 8, .kv_heads = 2, .head_dim = 4},
    };
    infernix::LayoutBuilder builder;
    const q36::StateImageDeviceLayout layout = q36::plan_state_image_device_pool(builder, spec);
    infernix::DeviceArena arena(builder.finish(256));
    q36::StateImageDevicePool physical({arena.base(), arena.capacity()}, layout);
    store::StateImageStore images(physical);
    expect(images.device_capacity() == 4 && images.device_occupied() == 0, "state store capacity");

    // A lane's image freezes into a prefix-cache snapshot in place, and thaws back.
    const auto lane = images.reserve_reset(device.stream);
    expect(lane.has_value() && images.role(*lane) == store::StateImageRole::ActiveMutable,
           "reset image is active");
    const std::int32_t lane_slot = images.physical_slot(*lane);
    images.freeze(*lane);
    expect(images.role(*lane) == store::StateImageRole::SnapshotImmutable &&
               images.physical_slot(*lane) == lane_slot,
           "freezing keeps the image's slot");
    images.thaw(*lane);
    expect(images.role(*lane) == store::StateImageRole::ActiveMutable, "thaw reactivates");

    // A copied destination becomes an active image or a snapshot; the copy is the caller's.
    const auto copied   = images.reserve_destination();
    const auto snapshot = images.reserve_destination();
    expect(copied && snapshot &&
               images.role(*copied) == store::StateImageRole::ReservedDestination &&
               images.physical_slot(*copied) != lane_slot &&
               images.physical_slot(*snapshot) != images.physical_slot(*copied),
           "destinations reserve distinct slots");
    const std::array<std::uint16_t, 8> content{1, 3, 5, 7, 9, 11, 13, 15};
    CUDA_CHECK(cudaMemcpyAsync(physical.continuation_hidden_slot(lane_slot).data, content.data(),
                               sizeof(content), cudaMemcpyHostToDevice, device.stream));
    physical.copy_slot(lane_slot, images.physical_slot(*copied), device.stream);
    images.activate_copied(*copied);
    physical.copy_slot(lane_slot, images.physical_slot(*snapshot), device.stream);
    images.publish_copied_snapshot(*snapshot);
    std::array<std::uint16_t, 8> observed{};
    CUDA_CHECK(cudaMemcpyAsync(observed.data(),
                               physical.continuation_hidden_slot(images.physical_slot(*snapshot)).data,
                               sizeof(observed), cudaMemcpyDeviceToHost, device.stream));
    device.synchronize();
    expect(images.role(*copied) == store::StateImageRole::ActiveMutable &&
               images.role(*snapshot) == store::StateImageRole::SnapshotImmutable &&
               observed == content,
           "copied destinations publish with the source content");
    bool rejected = false;
    try {
        images.activate_copied(*copied);
    } catch (const std::logic_error&) { rejected = true; }
    expect(rejected, "an activated image was activated again");

    // The store is exhausted at its slot count, and a released slot is reused under a new
    // generation.
    const auto last = images.reserve_destination();
    expect(last.has_value() && !images.reserve_destination() && images.device_occupied() == 4,
           "the store hands out exactly its slots");
    const std::int32_t last_slot = images.physical_slot(*last);
    expect(images.release(*last) && !images.valid(*last) && !images.release(*last),
           "a released image is stale");
    const auto reused = images.reserve_reset(device.stream);
    expect(reused && *reused != *last && images.physical_slot(*reused) == last_slot,
           "a reused slot carries a new generation");
    expect(images.release(*reused) && images.release(*lane) && images.release(*copied) &&
               images.release(*snapshot) && images.device_occupied() == 0,
           "every image releases");
}

struct KVFixture {
    KVFixture(std::uint32_t page_groups, std::uint32_t logical_page_capacity,
              std::uint32_t table_rows, std::uint32_t address_capacity) {
        infernix::LayoutBuilder builder;
        const infernix::DeviceKVPagePoolLayout page_layout = infernix::plan_device_kv_page_pool(
            builder,
            {.page_group_count = page_groups,
             .geometry         = {
                         .page_tokens        = static_cast<std::uint32_t>(infernix::kPagedKVPageSize),
                         .device_plane_order = infernix::PagedKVPlaneOrder::PageMajor,
                         .planes = {{.dtype = infernix::DType::BF16, .leading_extent = 8,
                                     .head_extent = 2}}}});
        const infernix::KVExecutionTableLayout table_layout = infernix::plan_kv_execution_tables(
            builder, {.logical_page_capacity = logical_page_capacity,
                      .table_rows            = static_cast<std::int32_t>(table_rows)});
        arena.emplace(builder.finish(256));
        const infernix::DeviceSpan backing{arena->base(), arena->capacity()};
        physical_pages.emplace(backing, page_layout);
        physical_tables.emplace(backing, table_layout, *physical_pages);
        pages.emplace(*physical_pages, physical_pages->capacity_pages());
        addresses.emplace(*pages, *physical_tables, address_capacity, logical_page_capacity);
    }

    std::optional<infernix::DeviceArena> arena;
    std::optional<infernix::DeviceKVPagePool> physical_pages;
    std::optional<infernix::KVExecutionTablePool> physical_tables;
    std::optional<store::LogicalKVPageStore> pages;
    std::optional<store::KVAddressSpaceStore> addresses;
};

void test_kv_store(infernix::DeviceContext& device) {
    KVFixture fixture(8, 4, 2, 4);
    infernix::DeviceKVPagePool& physical_pages         = *fixture.physical_pages;
    const infernix::KVExecutionTablePool& physical_tables = *fixture.physical_tables;
    store::LogicalKVPageStore& pages                   = *fixture.pages;
    store::KVAddressSpaceStore& addresses              = *fixture.addresses;

    const auto address = addresses.create_active(3, 0, device.stream);
    expect(address.has_value(), "active KV address allocation");
    addresses.ensure_mapped_to_tokens(*address, 65, device.stream);
    device.synchronize();
    expect(addresses.mapped_pages(*address) == 2 && addresses.committed_frontier(*address) == 0,
           "physical KV append does not publish canonical coverage before commit");
    expect(read_block_table(physical_tables, 0, 2) == std::vector<std::int32_t>({0, 1}),
           "batched KV materialization publishes the exact execution mapping");
    addresses.commit_frontier(*address, 65);
    expect(addresses.mapped_pages(*address) == 2 &&
               addresses.reserved_growth_pages(*address) == 1 &&
               addresses.committed_frontier(*address) == 65 && addresses.bound_row(*address) == 0,
           "KV address tracks mapped pages, remaining growth, frontier, and execution row");
    expect(pages.active_address_references(addresses.logical_page(*address, 0)) == 1 &&
               pages.active_address_references(addresses.logical_page(*address, 1)) == 1,
           "active KV membership is counted on each logical page");
    addresses.ensure_mapped_to_tokens(*address, 129, device.stream);
    device.synchronize();
    expect(read_block_table(physical_tables, 0, 3) == std::vector<std::int32_t>({0, 1, 2}),
           "incremental KV materialization preserves the existing mapping prefix");
    addresses.destructive_truncate(*address, 65);
    expect(addresses.mapped_pages(*address) == 2 &&
               addresses.reserved_growth_pages(*address) == 1 &&
               addresses.committed_frontier(*address) == 65,
           "same-frontier trim releases uncommitted materialized suffix pages");

    addresses.deactivate(*address);
    expect(addresses.bound_row(*address) == -1 && addresses.reserved_growth_pages(*address) == 0,
           "an inactive KV address owns no execution row or growth reservation");
    const std::array logical_pages{addresses.logical_page(*address, 0),
                                   addresses.logical_page(*address, 1)};
    expect(pages.active_address_references(logical_pages[0]) == 0 &&
               pages.active_address_references(logical_pages[1]) == 0 &&
               pages.writer_references(logical_pages[0]) == 0 &&
               pages.writer_references(logical_pages[1]) == 0,
           "KV deactivation clears active references and the writer");
    auto activation = addresses.prepare_activation(*address, 1, 1);
    expect(addresses.bound_row(*address) == -1 && addresses.reserved_growth_pages(*address) == 0 &&
               physical_pages.reserved_pages() == 1,
           "prepared KV activation preserves the inactive mapping until publication");
    addresses.commit_activation(std::move(activation), device.stream);
    device.synchronize();
    expect(pages.active_address_references(logical_pages[0]) == 1 &&
               pages.active_address_references(logical_pages[1]) == 1 &&
               pages.writer_references(logical_pages[1]) == 1 && addresses.bound_row(*address) == 1 &&
               read_block_table(physical_tables, 1, 2) == std::vector<std::int32_t>({0, 1}),
           "KV reactivation republishes the mapping, active references and the writer");
    addresses.destructive_truncate(*address, 32);
    expect(addresses.mapped_pages(*address) == 1 &&
               addresses.reserved_growth_pages(*address) == 2 &&
               addresses.committed_frontier(*address) == 32 &&
               pages.committed_columns(logical_pages[0]) == 32,
           "destructive rewrite truncates coverage and returns the suffix page to growth");
    addresses.deactivate(*address);
    expect(addresses.release(*address), "inactive KV address releases");
    expect(!addresses.valid(*address) && pages.occupied() == 0 &&
               physical_pages.allocated_pages() == 0 && physical_pages.reserved_pages() == 0,
           "KV release invalidates generations and closes physical ownership");

    const auto occupied_prefix = addresses.create_active(1, 0, device.stream);
    expect(occupied_prefix.has_value(), "growth rollback retained prefix allocation");
    addresses.ensure_mapped_to_tokens(*occupied_prefix, 1, device.stream);
    addresses.settle_growth(*occupied_prefix, 1);
    addresses.deactivate(*occupied_prefix);
    const auto occupied_page = addresses.logical_page(*occupied_prefix, 0);
    addresses.activate(*occupied_prefix, 0, 0, device.stream);
    expect(addresses.active(*occupied_prefix) && addresses.mapped_pages(*occupied_prefix) == 1 &&
               addresses.logical_page(*occupied_prefix, 0) == occupied_page &&
               addresses.reserved_growth_pages(*occupied_prefix) == 0 &&
               addresses.growth_pages_for_tokens(*occupied_prefix, 64) == 0 &&
               pages.writer_references(occupied_page) == 1 && physical_pages.reserved_pages() == 0,
           "zero-growth reactivation retains its existing page and restores the writer");
    addresses.deactivate(*occupied_prefix);
    const auto first_empty  = addresses.create_active(0, 0, device.stream);
    const auto second_empty = addresses.create_active(0, 1, device.stream);
    expect(first_empty && second_empty && addresses.active(*first_empty) &&
               addresses.active(*second_empty) && addresses.bound_row(*first_empty) == 0 &&
               addresses.bound_row(*second_empty) == 1 &&
               addresses.mapped_pages(*first_empty) == 0 &&
               addresses.mapped_pages(*second_empty) == 0 &&
               addresses.reserved_growth_pages(*first_empty) == 0 &&
               addresses.reserved_growth_pages(*second_empty) == 0 &&
               physical_pages.allocated_pages() == 1 && physical_pages.reserved_pages() == 0,
           "zero-growth active addresses own rows without reserving physical pages");
    addresses.ensure_mapped_to_tokens(*first_empty, 0, device.stream);
    addresses.reserve_growth(*first_empty, 4);
    bool growth_rejected = false;
    try {
        addresses.reserve_growth(*second_empty, 4);
    } catch (const std::bad_alloc&) { growth_rejected = true; }
    expect(growth_rejected && addresses.reserved_growth_pages(*first_empty) == 4 &&
               addresses.reserved_growth_pages(*second_empty) == 0 &&
               addresses.mapped_pages(*first_empty) == 0 &&
               addresses.mapped_pages(*second_empty) == 0 && pages.occupied() == 1 &&
               addresses.occupied() == 3 && physical_pages.allocated_pages() == 1 &&
               physical_pages.reserved_pages() == 4 && physical_pages.available_pages() == 3,
           "failed later-address growth preserves prior reservations without partial ownership");
    addresses.reserve_growth(*first_empty, 0);
    addresses.reserve_growth(*second_empty, 0);
    expect(addresses.active(*first_empty) && addresses.active(*second_empty) &&
               addresses.reserved_growth_pages(*first_empty) == 0 &&
               addresses.reserved_growth_pages(*second_empty) == 0 &&
               physical_pages.allocated_pages() == 1 && physical_pages.reserved_pages() == 0 &&
               physical_pages.available_pages() == 7,
           "multi-address growth rollback releases all reservations and preserves active rows");
    addresses.reserve_growth(*first_empty, 3);
    addresses.reserve_growth(*second_empty, 4);
    expect(addresses.reserved_growth_pages(*first_empty) == 3 &&
               addresses.reserved_growth_pages(*second_empty) == 4 &&
               physical_pages.reserved_pages() == 7 && physical_pages.available_pages() == 0,
           "growth retry can reserve all capacity returned by rollback");
    addresses.deactivate(*first_empty);
    addresses.deactivate(*second_empty);
    expect(addresses.release(*first_empty) && addresses.release(*second_empty) &&
               addresses.release(*occupied_prefix) && addresses.occupied() == 0 &&
               pages.occupied() == 0 && physical_pages.allocated_pages() == 0 &&
               physical_pages.reserved_pages() == 0 && physical_pages.available_pages() == 8,
           "zero-growth and failed reservation fixtures close all ownership");

    // Verify crosses a page boundary, but the terminal commit consumes only its first column.
    const auto terminal = addresses.create_active(3, 0, device.stream);
    expect(terminal.has_value(), "terminal boundary KV address allocation");
    addresses.ensure_mapped_to_tokens(*terminal, 63, device.stream);
    addresses.commit_frontier(*terminal, 63);
    addresses.ensure_mapped_to_tokens(*terminal, 71, device.stream);
    device.synchronize();
    const auto terminal_mapping   = read_block_table(physical_tables, 0, 2);
    const auto unchanged_terminal = [&] {
        return addresses.mapped_pages(*terminal) == 2 &&
               addresses.committed_frontier(*terminal) == 63 &&
               addresses.reserved_growth_pages(*terminal) == 1 &&
               physical_pages.allocated_pages() == 2 && physical_pages.reserved_pages() == 1 &&
               physical_pages.available_pages() == 5 &&
               read_block_table(physical_tables, 0, 2) == terminal_mapping;
    };
    addresses.ensure_mapped_to_tokens(*terminal, 71, device.stream);
    addresses.ensure_mapped_to_tokens(*terminal, 64, device.stream);
    device.synchronize();
    expect(unchanged_terminal(), "covered KV requests preserve speculative mappings and ownership");
    bool exceeded = false;
    try {
        addresses.ensure_mapped_to_tokens(*terminal, 193, device.stream);
    } catch (const std::invalid_argument&) { exceeded = true; }
    expect(exceeded && unchanged_terminal(),
           "KV coverage beyond reserved growth fails without mutation");
    addresses.commit_frontier(*terminal, 64);
    addresses.settle_growth(*terminal, 64);
    expect(addresses.mapped_pages(*terminal) == 1 &&
               addresses.committed_frontier(*terminal) == 64 &&
               addresses.reserved_growth_pages(*terminal) == 0 &&
               physical_pages.allocated_pages() == 1 && physical_pages.reserved_pages() == 0 &&
               physical_pages.available_pages() == 7 && addresses.active(*terminal),
           "growth settlement returns speculative and unused pages while keeping the row active");
    expect(addresses.growth_pages_for_tokens(*terminal, 64) == 0 &&
               addresses.growth_pages_for_tokens(*terminal, 65) == 1,
           "next unit reserves only pages beyond the settled mapping");
    addresses.reserve_growth(*terminal, addresses.growth_pages_for_tokens(*terminal, 65));
    expect(addresses.growth_pages_for_tokens(*terminal, 65) == 1 &&
               addresses.reserved_growth_pages(*terminal) == 1,
           "unit growth demand is independent of an already acquired reservation");
    addresses.ensure_mapped_to_tokens(*terminal, 65, device.stream);
    addresses.settle_growth(*terminal, 65);
    expect(addresses.mapped_pages(*terminal) == 2 &&
               addresses.committed_frontier(*terminal) == 65 &&
               addresses.reserved_growth_pages(*terminal) == 0 &&
               physical_pages.allocated_pages() == 2 && physical_pages.reserved_pages() == 0,
           "settled active address reserves and commits the following unit independently");
    addresses.deactivate(*terminal);
    expect(addresses.release(*terminal) && physical_pages.allocated_pages() == 0 &&
               physical_pages.reserved_pages() == 0 && physical_pages.available_pages() == 8,
           "terminal settlement releases both mappings and unused growth");
}

// The prefix cache starts a history from cached pages: full pages are shared by reference and a
// partial tail is copied into a private page before publication.
void test_page_prefix_fork(infernix::DeviceContext& device) {
    KVFixture fixture(8, 4, 2, 4);
    infernix::DeviceKVPagePool& physical_pages         = *fixture.physical_pages;
    const infernix::KVExecutionTablePool& physical_tables = *fixture.physical_tables;
    store::LogicalKVPageStore& pages                   = *fixture.pages;
    store::KVAddressSpaceStore& addresses              = *fixture.addresses;

    const auto shared = addresses.create_active(3, 0, device.stream);
    expect(shared.has_value(), "cached-prefix source allocation");
    addresses.ensure_mapped_to_tokens(*shared, 65, device.stream);
    addresses.commit_frontier(*shared, 65);
    device.synchronize();
    const auto shared_mapping = read_block_table(physical_tables, 0, 2);
    addresses.deactivate(*shared);
    const auto shared_full = addresses.logical_page(*shared, 0);
    const auto shared_tail = addresses.logical_page(*shared, 1);
    const std::array full_pages{shared_full};

    const auto branch = addresses.create_inactive();
    expect(branch.has_value(), "cached-prefix branch allocation");
    {
        auto aborted = addresses.prepare_page_prefix_fork(*branch, full_pages, shared_tail, 1, 1, 1);
        expect(aborted.needs_tail_copy() && pages.source_pins(shared_full) == 1 &&
                   pages.source_pins(shared_tail) == 1 && physical_pages.allocated_pages() == 3 &&
                   physical_pages.reserved_pages() == 1,
               "a page prefix fork pins its sources and reserves its tail copy and growth");
    }
    expect(pages.source_pins(shared_full) == 0 && pages.source_pins(shared_tail) == 0 &&
               physical_pages.allocated_pages() == 2 && physical_pages.reserved_pages() == 0 &&
               !addresses.active(*branch) && addresses.mapped_pages(*branch) == 0,
           "an aborted page prefix fork releases its pins, tail and growth");

    auto fork = addresses.prepare_page_prefix_fork(*branch, full_pages, shared_tail, 1, 1, 1);
    physical_pages.copy_page(addresses.page_prefix_fork_tail_source(fork),
                             addresses.page_prefix_fork_tail_destination(fork),
                             device.transfer_stream);
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    addresses.commit_page_prefix_fork(std::move(fork), device.stream);
    device.synchronize();
    const auto branch_mapping = read_block_table(physical_tables, 1, 2);
    const auto branch_tail    = addresses.logical_page(*branch, 1);
    expect(addresses.logical_page(*branch, 0) == shared_full &&
               branch_mapping[0] == shared_mapping[0] && branch_mapping[1] != shared_mapping[1] &&
               branch_tail != shared_tail && pages.address_references(shared_full) == 2 &&
               pages.writer_references(shared_full) == 0 &&
               pages.writer_references(branch_tail) == 1 &&
               addresses.committed_frontier(*branch) == 65 &&
               addresses.reserved_growth_pages(*branch) == 1 &&
               pages.source_pins(shared_full) == 0 && pages.source_pins(shared_tail) == 0,
           "a page prefix fork shares the full page and publishes a private partial tail");
    addresses.ensure_mapped_to_tokens(*branch, 66, device.stream);
    addresses.commit_frontier(*branch, 66);
    expect(pages.committed_columns(shared_tail) == 1 && pages.committed_columns(branch_tail) == 2,
           "branch writes do not extend the shared partial tail");
    bool shared_truncate_rejected = false;
    try {
        addresses.destructive_truncate(*branch, 32);
    } catch (const std::logic_error&) { shared_truncate_rejected = true; }
    expect(shared_truncate_rejected && addresses.mapped_pages(*branch) == 2 &&
               addresses.committed_frontier(*branch) == 66 &&
               pages.committed_columns(shared_full) == 64,
           "a branch cannot truncate into the shared page's protected coverage");
    addresses.deactivate(*branch);
    expect(addresses.release(*shared) && !pages.valid(shared_tail) && pages.valid(shared_full) &&
               pages.address_references(shared_full) == 1 &&
               physical_pages.allocated_pages() == 2,
           "the branch keeps the shared page after its source releases");
    expect(addresses.release(*branch) && pages.occupied() == 0 &&
               physical_pages.allocated_pages() == 0 && physical_pages.reserved_pages() == 0,
           "releasing the branch closes all ownership");

    // A fork that cannot reserve its tail and growth leaves its sources untouched.
    const auto filler = addresses.create_active(4, 0, device.stream);
    expect(filler.has_value(), "full-capacity filler allocation");
    addresses.ensure_mapped_to_tokens(*filler, 193, device.stream);
    addresses.commit_frontier(*filler, 193);
    addresses.deactivate(*filler);
    const auto source = addresses.create_active(4, 0, device.stream);
    expect(source.has_value(), "full-capacity cached-prefix source allocation");
    addresses.ensure_mapped_to_tokens(*source, 65, device.stream);
    addresses.commit_frontier(*source, 65);
    addresses.deactivate(*source);
    const auto source_full = addresses.logical_page(*source, 0);
    const auto source_tail = addresses.logical_page(*source, 1);
    const std::array source_full_pages{source_full};
    const auto destination = addresses.create_inactive();
    expect(destination.has_value() && physical_pages.allocated_pages() == 6 &&
               physical_pages.available_pages() == 2,
           "the fixture lacks space for the tail copy and growth together");
    bool fork_rejected = false;
    try {
        auto rejected =
            addresses.prepare_page_prefix_fork(*destination, source_full_pages, source_tail, 1, 2, 1);
        (void)rejected;
    } catch (const std::bad_alloc&) { fork_rejected = true; }
    expect(fork_rejected && !addresses.active(*destination) &&
               addresses.mapped_pages(*destination) == 0 &&
               addresses.bound_row(*destination) == -1 &&
               pages.address_references(source_full) == 1 && pages.source_pins(source_full) == 0 &&
               pages.source_pins(source_tail) == 0 && physical_pages.allocated_pages() == 6 &&
               physical_pages.reserved_pages() == 0 && physical_pages.available_pages() == 2,
           "a failed page prefix fork releases every destination claim");
    expect(addresses.release(*filler) && physical_pages.available_pages() == 6,
           "releasing unrelated ownership makes the fork feasible");
    auto retried =
        addresses.prepare_page_prefix_fork(*destination, source_full_pages, source_tail, 1, 2, 1);
    physical_pages.copy_page(addresses.page_prefix_fork_tail_source(retried),
                             addresses.page_prefix_fork_tail_destination(retried),
                             device.transfer_stream);
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    addresses.commit_page_prefix_fork(std::move(retried), device.stream);
    expect(addresses.logical_page(*destination, 0) == source_full &&
               addresses.logical_page(*destination, 1) != source_tail &&
               addresses.reserved_growth_pages(*destination) == 2 &&
               physical_pages.allocated_pages() == 3 && physical_pages.reserved_pages() == 2,
           "a retried fork publishes a private tail and the complete growth reservation");
    addresses.deactivate(*destination);
    expect(addresses.release(*destination) && addresses.release(*source) &&
               pages.occupied() == 0 && physical_pages.allocated_pages() == 0 &&
               physical_pages.reserved_pages() == 0,
           "fork failure and retry close all logical and physical ownership");
}

// A cached prefix longer than one directory chunk keeps every page identity and mapping.
void test_long_page_prefix_fork(infernix::DeviceContext& device) {
    constexpr std::uint32_t page_tokens = infernix::kPagedKVPageSize;
    constexpr std::uint32_t full_count  = 65;
    constexpr std::uint32_t frontier    = full_count * page_tokens + 1;
    KVFixture fixture(72, 68, 2, 3);
    infernix::DeviceKVPagePool& physical_pages         = *fixture.physical_pages;
    const infernix::KVExecutionTablePool& physical_tables = *fixture.physical_tables;
    store::LogicalKVPageStore& pages                   = *fixture.pages;
    store::KVAddressSpaceStore& addresses              = *fixture.addresses;

    const auto source = addresses.create_active(full_count + 1, 0, device.stream);
    expect(source.has_value(), "long cached-prefix source allocation");
    addresses.ensure_mapped_to_tokens(*source, frontier, device.stream);
    addresses.settle_growth(*source, frontier);
    device.synchronize();
    const auto source_mapping = read_block_table(physical_tables, 0, full_count + 1);
    addresses.deactivate(*source);
    std::vector<store::LogicalKVPageHandle> shared_pages;
    for (std::uint32_t page = 0; page < full_count; ++page) {
        shared_pages.push_back(addresses.logical_page(*source, page));
    }
    const auto source_tail = addresses.logical_page(*source, full_count);
    const auto branch      = addresses.create_inactive();
    expect(branch.has_value(), "long cached-prefix branch allocation");
    auto fork = addresses.prepare_page_prefix_fork(*branch, shared_pages, source_tail, 1, 1, 1);
    physical_pages.copy_page(addresses.page_prefix_fork_tail_source(fork),
                             addresses.page_prefix_fork_tail_destination(fork),
                             device.transfer_stream);
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    addresses.commit_page_prefix_fork(std::move(fork), device.stream);
    device.synchronize();
    const auto branch_mapping    = read_block_table(physical_tables, 1, full_count + 1);
    bool fork_identity_preserved = true;
    for (std::uint32_t page = 0; page < full_count; ++page) {
        fork_identity_preserved = fork_identity_preserved &&
                                  addresses.logical_page(*branch, page) == shared_pages[page] &&
                                  branch_mapping[page] == source_mapping[page] &&
                                  pages.address_references(shared_pages[page]) == 2;
    }
    const auto branch_tail = addresses.logical_page(*branch, full_count);
    expect(fork_identity_preserved && branch_tail != source_tail &&
               branch_mapping.back() != source_mapping.back() &&
               physical_pages.allocated_pages() == full_count + 2 &&
               addresses.reserved_growth_pages(*branch) == 1,
           "a long fork shares every full-page identity and copies only the partial tail");
    expect(addresses.release(*source) && !pages.valid(source_tail) &&
               physical_pages.allocated_pages() == full_count + 1,
           "a long fork keeps the shared pages after its source releases");
    const std::uint32_t next_frontier = (full_count + 1) * page_tokens + 1;
    addresses.ensure_mapped_to_tokens(*branch, next_frontier, device.stream);
    addresses.settle_growth(*branch, next_frontier);
    device.synchronize();
    const auto grown_mapping = read_block_table(physical_tables, 1, full_count + 2);
    expect(std::equal(branch_mapping.begin(), branch_mapping.end(), grown_mapping.begin()) &&
               addresses.mapped_pages(*branch) == full_count + 2,
           "appending past a directory chunk preserves the inherited mapping");
    addresses.deactivate(*branch);
    expect(addresses.release(*branch) && pages.occupied() == 0 &&
               physical_pages.allocated_pages() == 0 && physical_pages.reserved_pages() == 0,
           "the long branch closes all ownership");
}

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err) || count == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    CUDA_CHECK(count_err);

    try {
        infernix::DeviceContext device(0);
        test_state_store(device);
        test_kv_store(device);
        test_page_prefix_fork(device);
        test_long_page_prefix_fork(device);
        device.synchronize();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
        return 1;
    }
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
