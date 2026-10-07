#include "core/device.h"
#include "core/host_kv_arena.h"
#include "core/paged_kv_cache.h"
#include "core/vmm_arena.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

struct PlannedCache {
    infernix::DeviceKVPagePoolLayout pages;
    infernix::KVExecutionTableLayout tables;
    std::size_t bytes = 0;
};

PlannedCache plan_cache(std::uint32_t physical_pages, std::uint32_t logical_pages,
                        std::int32_t rows, infernix::KVPageGeometry geometry) {
    infernix::LayoutBuilder builder;
    PlannedCache out;
    out.pages = infernix::plan_device_kv_page_pool(
        builder, {.page_group_count = physical_pages, .geometry = std::move(geometry)});
    out.tables = infernix::plan_kv_execution_tables(
        builder, {.logical_page_capacity = logical_pages, .table_rows = rows});
    out.bytes = builder.finish(256);
    return out;
}

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

int expect(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

int expect_size(std::size_t actual, std::size_t expected, const std::string& label) {
    return expect(actual == expected, label + " expected " + std::to_string(expected) + ", got " +
                                          std::to_string(actual));
}

std::vector<infernix::DeviceKVPageLease> materialize(infernix::DeviceKVPagePool& pool,
                                                   std::uint32_t pages) {
    std::optional<infernix::DeviceKVPageReservation> reservation = pool.reserve(pages);
    if (!reservation) { throw std::bad_alloc(); }
    std::vector<infernix::DeviceKVPageLease> out;
    out.reserve(pages);
    pool.materialize(*reservation, pages, out);
    return out;
}

std::vector<infernix::DeviceKVPageHandle> handles(std::span<const infernix::DeviceKVPageLease> pages) {
    std::vector<infernix::DeviceKVPageHandle> out;
    out.reserve(pages.size());
    for (const infernix::DeviceKVPageLease& page : pages) { out.push_back(page.handle()); }
    return out;
}

std::vector<std::int32_t> read_mapping(const infernix::Tensor& row, std::size_t count) {
    std::vector<std::int32_t> out(count);
    const cudaError_t err =
        cudaMemcpy(out.data(), row.data, out.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("block-table read failed: ") +
                                 cudaGetErrorString(err));
    }
    return out;
}

std::vector<std::vector<unsigned char>>
fill_device_pool(infernix::DeviceKVPagePool& pool, cudaStream_t stream, unsigned char seed = 0) {
    std::vector<std::vector<unsigned char>> bytes;
    bytes.reserve(pool.plane_count());
    for (std::size_t plane_index = 0; plane_index < pool.plane_count(); ++plane_index) {
        const infernix::Tensor& plane = pool.plane(plane_index);
        std::vector<unsigned char> host(plane.bytes());
        for (std::size_t index = 0; index < host.size(); ++index) {
            host[index] =
                static_cast<unsigned char>(((index * 29U) ^ (index >> 8U) ^ (index >> 16U) ^
                                            (plane_index * 61U + seed + 17U)) &
                                           0xffU);
        }
        const cudaError_t err =
            cudaMemcpyAsync(plane.data, host.data(), host.size(), cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("device pool fill failed: ") +
                                     cudaGetErrorString(err));
        }
        bytes.push_back(std::move(host));
    }
    return bytes;
}

std::vector<std::byte>
expected_host_records(const infernix::DeviceKVPagePool& pool,
                      std::span<const std::int32_t> physical_pages,
                      const infernix::HostKVPageLayout& host_layout,
                      const std::vector<std::vector<unsigned char>>& device_planes) {
    std::vector<std::byte> out(host_layout.page_stride * physical_pages.size(), std::byte{0});
    for (std::size_t logical = 0; logical < physical_pages.size(); ++logical) {
        const std::int32_t physical = physical_pages[logical];
        for (std::size_t plane_index = 0; plane_index < pool.plane_count(); ++plane_index) {
            const infernix::Tensor& plane                 = pool.plane(plane_index);
            const infernix::HostKVPlaneLayout& host_plane = host_layout.planes[plane_index];
            std::byte* destination =
                out.data() + logical * host_layout.page_stride + host_plane.offset;
            if (pool.geometry().device_plane_order == infernix::PagedKVPlaneOrder::PageMajor) {
                const unsigned char* source = device_planes[plane_index].data() +
                                              static_cast<std::size_t>(physical) * plane.nb[3];
                std::memcpy(destination, source, host_plane.page_payload_bytes);
            } else {
                for (std::int32_t head = 0; head < plane.ne[3]; ++head) {
                    const unsigned char* source = device_planes[plane_index].data() +
                                                  static_cast<std::size_t>(head) * plane.nb[3] +
                                                  static_cast<std::size_t>(physical) * plane.nb[2];
                    std::memcpy(destination +
                                    static_cast<std::size_t>(head) * host_plane.head_payload_bytes,
                                source, host_plane.head_payload_bytes);
                }
            }
        }
    }
    return out;
}

bool page_payload_equal(infernix::HostKVAllocationConstView left_view, std::uint32_t left,
                        infernix::HostKVAllocationConstView right_view, std::uint32_t right) {
    const infernix::HostKVPageLayout& layout = left_view.layout();
    if (right_view.layout() != layout) { return false; }
    for (const infernix::HostKVPlaneLayout& plane : layout.planes) {
        const std::byte* a =
            left_view.data() + static_cast<std::size_t>(left) * layout.page_stride + plane.offset;
        const std::byte* b =
            right_view.data() + static_cast<std::size_t>(right) * layout.page_stride + plane.offset;
        if (std::memcmp(a, b, plane.page_payload_bytes) != 0) { return false; }
    }
    return true;
}

bool page_payload_zero(infernix::HostKVAllocationConstView view, std::uint32_t page) {
    const infernix::HostKVPageLayout& layout = view.layout();
    for (const infernix::HostKVPlaneLayout& plane : layout.planes) {
        const std::byte* data =
            view.data() + static_cast<std::size_t>(page) * layout.page_stride + plane.offset;
        for (std::size_t index = 0; index < plane.page_payload_bytes; ++index) {
            if (data[index] != std::byte{0}) { return false; }
        }
    }
    return true;
}

int exercise_reservation_and_mapping(infernix::DeviceContext& context) {
    int failures = 0;
    infernix::KVPageGeometry geometry{
        .planes = {{infernix::DType::I8, 8, 2, 256}},
    };
    PlannedCache plan = plan_cache(4, 4, 1, geometry);
    infernix::DeviceArena arena(plan.bytes);
    infernix::DeviceKVPagePool pool({arena.base(), arena.capacity()}, plan.pages);
    infernix::KVExecutionTablePool tables({arena.base(), arena.capacity()}, plan.tables, pool);

    std::optional<infernix::DeviceKVPageReservation> reservation = pool.reserve(3);
    failures += expect(reservation.has_value(), "three-page reservation failed");
    failures += expect_size(pool.reserved_pages(), 3, "reserved pages");
    failures += expect_size(pool.available_pages(), 1, "available after reserve");

    std::vector<infernix::DeviceKVPageLease> pages;
    pages.reserve(3);
    pool.materialize(*reservation, 2, pages);
    failures += expect_size(pool.allocated_pages(), 2, "allocated after materialize");
    failures += expect_size(pool.reserved_pages(), 1, "remaining reservation");
    failures += expect(!pool.reserve(2).has_value(), "over-capacity reservation succeeded");

    pool.dematerialize(*reservation, 1, pages);
    failures += expect_size(pool.allocated_pages(), 1, "allocated after dematerialize");
    failures += expect_size(pool.reserved_pages(), 2, "reservation after dematerialize");
    failures +=
        expect_size(pool.available_pages(), 1, "available capacity changed after dematerialize");
    pool.materialize(*reservation, 2, pages);

    infernix::KVExecutionRowLease row = tables.acquire(0);
    tables.publish(row.handle(), 0, pages, context.stream);
    context.synchronize();
    const std::vector<std::int32_t> mapping = read_mapping(tables.row(row.handle()), pages.size());
    failures += expect(mapping == std::vector<std::int32_t>({0, 1}),
                       "execution mapping did not preserve logical order");
    row.release();
    failures += expect_size(pool.allocated_pages(), 2, "row release changed page ownership");

    reservation->clear();
    failures += expect_size(pool.reserved_pages(), 0, "reservation clear");
    const infernix::DeviceKVPageHandle stale = pages.back().handle();
    pages.back().release();
    bool stale_rejected = false;
    try {
        pool.zero_pages(std::span<const infernix::DeviceKVPageHandle>(&stale, 1), context.stream);
    } catch (const std::invalid_argument&) { stale_rejected = true; }
    failures += expect(stale_rejected, "released page capability remained usable");

    PlannedCache other_plan = plan_cache(1, 1, 1, geometry);
    infernix::DeviceArena other_arena(other_plan.bytes);
    infernix::DeviceKVPagePool other({other_arena.base(), other_arena.capacity()}, other_plan.pages);
    const std::uint32_t before                                = pool.reserved_pages();
    const infernix::DeviceKVPageReservationRequest impossible[] = {
        {.pool = &pool, .pages = 2},
        {.pool = &other, .pages = 2},
    };
    bool bundle_failed = false;
    try {
        auto unused = infernix::reserve_device_kv_page_bundle(impossible);
        (void)unused;
    } catch (const std::bad_alloc&) { bundle_failed = true; }
    failures += expect(bundle_failed, "impossible multi-pool reservation succeeded");
    failures += expect_size(pool.reserved_pages(), before, "failed bundle changed the first pool");
    return failures;
}

// An elastic pool (design §19.3.11): only pages below the backed limit are handed out, first fit keeps
// them low, the limit rises freely and falls only over free pages.
int exercise_backed_limit(infernix::DeviceContext& context) {
    int failures = 0;
    infernix::KVPageGeometry geometry{
        .planes = {{infernix::DType::I8, 8, 2, 256}},
    };
    PlannedCache plan = plan_cache(8, 8, 1, geometry);
    infernix::DeviceArena arena(plan.bytes);
    infernix::DeviceKVPagePool pool({arena.base(), arena.capacity()}, plan.pages);
    infernix::KVExecutionTablePool tables({arena.base(), arena.capacity()}, plan.tables, pool);
    failures += expect_size(pool.backed_pages(), 8, "a new pool backs its capacity");

    pool.set_backed_pages(3);
    failures += expect_size(pool.available_pages(), 3, "available under the limit");
    failures += expect(!pool.reserve(4).has_value(), "reservation past the backed limit succeeded");
    std::optional<infernix::DeviceKVPageReservation> first = pool.reserve(3);
    failures += expect(first.has_value(), "reservation within the limit failed");
    std::vector<infernix::DeviceKVPageLease> low;
    low.reserve(3);
    pool.materialize(*first, 3, low); // pages 0..2
    failures += expect(pool.can_back(3) && !pool.can_back(2), "can_back ignored pages in use");
    bool refused = false;
    try {
        pool.set_backed_pages(2);
    } catch (const std::logic_error&) { refused = true; }
    failures += expect(refused, "the limit fell over a page in use");

    pool.set_backed_pages(8);
    failures += expect_size(pool.available_pages(), 5, "available after growth");
    std::optional<infernix::DeviceKVPageReservation> second = pool.reserve(5);
    failures += expect(second.has_value(), "reservation of the grown pages failed");
    std::vector<infernix::DeviceKVPageLease> high;
    high.reserve(5);
    pool.materialize(*second, 5, high); // pages 3..7

    infernix::KVExecutionRowLease row = tables.acquire(0);
    tables.publish(row.handle(), 0, std::span<const infernix::DeviceKVPageLease>(low), context.stream);
    tables.publish(row.handle(), 3, std::span<const infernix::DeviceKVPageLease>(high), context.stream);
    context.synchronize();
    failures += expect(read_mapping(tables.row(row.handle()), 8) == std::vector<std::int32_t>({0, 1, 2, 3, 4, 5, 6, 7}),
                       "pages were not handed out lowest first across growth");
    row.release();

    // Free the top five: the limit may fall back to three, not below.
    high.clear();
    failures += expect(pool.can_back(3) && !pool.can_back(2), "can_back after freeing the top");
    pool.set_backed_pages(3);
    failures += expect_size(pool.available_pages(), 0, "available after shrinking to the pages in use");
    failures += expect(!pool.reserve(1).has_value(), "reservation past the lowered limit succeeded");

    // A free page below a used one does not let the limit fall past the used one.
    pool.set_backed_pages(8);
    low.pop_back(); // frees page 2
    std::optional<infernix::DeviceKVPageReservation> third = pool.reserve(2);
    std::vector<infernix::DeviceKVPageLease> mid;
    mid.reserve(2);
    pool.materialize(*third, 2, mid); // pages 2 and 3
    low.clear();                      // frees 0 and 1
    failures += expect(pool.can_back(4) && !pool.can_back(3), "a low hole let the limit fall past a used page");
    mid.clear();
    pool.set_backed_pages(0);
    failures += expect_size(pool.available_pages(), 0, "an empty pool shrank to nothing");
    pool.set_backed_pages(8);
    failures += expect_size(pool.available_pages(), 8, "an empty pool grew back to its capacity");
    return failures;
}

// Chunks map and unmap individually anywhere in the reserved range and keep their addresses.
int exercise_vmm_range() {
    if (!infernix::VmmArena::supported(0)) { return 0; }
    int failures            = 0;
    const std::size_t chunk = 2ULL << 20;
    infernix::VmmRange range(0, 4 * chunk, chunk);
    failures += expect_size(range.chunk_count(), 4, "VMM range chunk count");
    failures += expect_size(range.mapped_bytes(), 0, "a new VMM range maps nothing");
    auto* base = static_cast<std::byte*>(range.base());
    failures += expect(range.map(2) && range.map(0), "mapping VMM range chunks failed");
    failures += expect(range.mapped(0) && !range.mapped(1) && range.mapped(2), "VMM range mapped state");
    failures += expect_size(range.mapped_bytes(), 2 * chunk, "VMM range mapped bytes");
    std::vector<std::uint8_t> host(chunk, 0x5A), back(chunk, 0);
    if (cudaMemcpy(base + 2 * chunk, host.data(), chunk, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(back.data(), base + 2 * chunk, chunk, cudaMemcpyDeviceToHost) != cudaSuccess) {
        return failures + expect(false, "VMM range chunk copy failed");
    }
    failures += expect(back == host, "VMM range chunk did not hold its bytes");
    range.unmap(2);
    failures += expect(!range.mapped(2), "VMM range unmap");
    failures += expect(range.map(2), "remapping a VMM range chunk failed");
    failures += expect(range.base() == base, "the VMM range moved");
    return failures;
}

void CUDART_CB hold_stream(void* milliseconds) {
    std::this_thread::sleep_for(std::chrono::milliseconds(*static_cast<int*>(milliseconds)));
}

// A publication's copy reads the row's pinned shadow when the stream reaches it. Work queued
// behind it must see that publication even when the row is released and republished by its next
// owner before the stream gets there (an abort mid-prefill followed by the next admission).
int exercise_republish_while_copy_queued(infernix::DeviceContext& context) {
    int failures = 0;
    infernix::KVPageGeometry geometry{
        .planes = {{infernix::DType::I8, 8, 2, 256}},
    };
    PlannedCache plan = plan_cache(4, 4, 1, geometry);
    infernix::DeviceArena arena(plan.bytes);
    infernix::DeviceKVPagePool pool({arena.base(), arena.capacity()}, plan.pages);
    infernix::KVExecutionTablePool tables({arena.base(), arena.capacity()}, plan.tables, pool);
    std::optional<infernix::DeviceKVPageReservation> reservation = pool.reserve(3);
    std::vector<infernix::DeviceKVPageLease> pages;
    pages.reserve(3);
    pool.materialize(*reservation, 3, pages);
    const std::vector<infernix::DeviceKVPageHandle> first{pages[0].handle(), pages[1].handle()};
    const std::vector<infernix::DeviceKVPageHandle> second{pages[2].handle(), pages[0].handle()};

    std::int32_t* observed = nullptr;
    if (cudaMalloc(&observed, 2 * sizeof(std::int32_t)) != cudaSuccess) {
        throw std::runtime_error("observation buffer allocation failed");
    }
    infernix::KVExecutionRowLease owner = tables.acquire(0);
    const infernix::Tensor device_row   = tables.row(owner.handle());
    int hold_ms                       = 300;
    if (cudaLaunchHostFunc(context.stream, hold_stream, &hold_ms) != cudaSuccess) {
        throw std::runtime_error("stream hold failed");
    }
    tables.publish(owner.handle(), 0, first, context.stream);
    // Stands for the owner's queued kernels, which read the table in stream order.
    if (cudaMemcpyAsync(observed, device_row.data, 2 * sizeof(std::int32_t),
                        cudaMemcpyDeviceToDevice, context.stream) != cudaSuccess) {
        throw std::runtime_error("table observation failed");
    }
    owner.release();
    infernix::KVExecutionRowLease next = tables.acquire(0);
    tables.publish(next.handle(), 0, second, context.stream);
    context.synchronize();

    std::vector<std::int32_t> seen(2);
    cudaMemcpy(seen.data(), observed, seen.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost);
    cudaFree(observed);
    failures += expect(seen == std::vector<std::int32_t>({0, 1}),
                       "queued work saw the next owner's mapping instead of its own");
    failures += expect(read_mapping(tables.row(next.handle()), 2) ==
                           std::vector<std::int32_t>({2, 0}),
                       "the next owner's publication did not land");

    // A disjoint extension of a row with a copy still queued does not wait for it.
    if (cudaLaunchHostFunc(context.stream, hold_stream, &hold_ms) != cudaSuccess) {
        throw std::runtime_error("stream hold failed");
    }
    tables.publish(next.handle(), 0, second, context.stream);
    const auto started = std::chrono::steady_clock::now();
    tables.publish(next.handle(), 2, std::span<const infernix::DeviceKVPageHandle>(first.data(), 1),
                   context.stream);
    const auto waited = std::chrono::steady_clock::now() - started;
    context.synchronize();
    failures += expect(waited < std::chrono::milliseconds(100),
                       "a disjoint publication waited for an unrelated queued copy");
    failures += expect(read_mapping(tables.row(next.handle()), 3) ==
                           std::vector<std::int32_t>({2, 0, 0}),
                       "disjoint extension did not land");
    return failures;
}

int exercise_layout_and_transfer(infernix::DeviceContext& context, infernix::KVPageGeometry geometry,
                                 const std::string& label,
                                 const std::array<std::uint32_t, 4>& run_operations) {
    int failures                  = 0;
    PlannedCache source_plan      = plan_cache(10, 8, 2, geometry);
    PlannedCache destination_plan = plan_cache(10, 8, 1, geometry);
    infernix::DeviceArena source_arena(source_plan.bytes);
    infernix::DeviceArena destination_arena(destination_plan.bytes);
    infernix::DeviceKVPagePool source({source_arena.base(), source_arena.capacity()},
                                    source_plan.pages);
    infernix::KVExecutionTablePool tables({source_arena.base(), source_arena.capacity()},
                                        source_plan.tables, source);
    infernix::DeviceKVPagePool destination({destination_arena.base(), destination_arena.capacity()},
                                         destination_plan.pages);

    failures += expect_size(source.plane_count(), geometry.planes.size(), label + " plane count");
    const infernix::Tensor& first_plane = source.plane(0);
    if (geometry.device_plane_order == infernix::PagedKVPlaneOrder::PageMajor) {
        failures += expect_size(first_plane.ne[2], geometry.planes[0].head_extent,
                                label + " PageMajor heads");
        failures += expect_size(first_plane.ne[3], 10, label + " PageMajor pages");
    } else {
        failures += expect_size(first_plane.ne[2], 10, label + " HeadMajor pages");
        failures += expect_size(first_plane.ne[3], geometry.planes[0].head_extent,
                                label + " HeadMajor heads");
    }

    std::vector<infernix::DeviceKVPageLease> prefix   = materialize(source, 3);
    std::vector<infernix::DeviceKVPageLease> blockers = materialize(source, 3);
    prefix.clear();
    std::vector<infernix::DeviceKVPageLease> fragmented            = materialize(source, 5);
    const std::vector<infernix::DeviceKVPageHandle> source_handles = handles(fragmented);
    infernix::KVExecutionRowLease row                              = tables.acquire(1);
    tables.publish(row.handle(), 0, source_handles, context.stream);
    context.synchronize();
    const std::vector<std::int32_t> physical_mapping =
        read_mapping(tables.row(row.handle()), source_handles.size());
    failures += expect(physical_mapping == std::vector<std::int32_t>({0, 1, 2, 6, 7}),
                       label + " execution row differs from logical page order");
    const std::uint32_t allocated_before_row_release = source.allocated_pages();
    row.release();
    failures += expect_size(source.allocated_pages(), allocated_before_row_release,
                            label + " row ownership isolation");

    const std::vector<std::vector<unsigned char>> device_bytes =
        fill_device_pool(source, context.stream);
    context.synchronize();

    const infernix::HostKVPageLayout host_layout =
        infernix::plan_host_kv_page_layout(source.geometry());
    const infernix::HostKVPageLayout layouts[] = {host_layout};
    infernix::HostContextArena host_backing(host_layout.page_stride * 24, host_layout.page_stride);
    infernix::HostKVArena host_arena(host_backing,
                                   std::span<const infernix::HostKVPageLayout>(layouts));
    failures += expect(!host_arena.can_allocate(host_layout, 25),
                       label + " oversized Host extent was reported allocatable");
    std::optional<infernix::HostKVAllocation> host =
        host_arena.allocate(host_layout, static_cast<std::uint32_t>(source_handles.size()));
    failures += expect(host.has_value(), label + " Host allocation failed");
    infernix::HostKVAllocationView host_view = host_arena.writable_view(*host);
    std::memset(host_view.data(), 0, host_layout.page_stride * host_view.page_count());
    std::uint64_t page_payload = 0;
    for (const auto& plane : host_layout.planes) { page_payload += plane.page_payload_bytes; }
    failures += expect(source.host_transfer_run_work(0) == infernix::TransferWork{},
                       label + " empty run has transfer work");
    for (std::uint32_t pages = 1; pages <= run_operations.size(); ++pages) {
        failures +=
            expect(source.host_transfer_run_work(pages) ==
                       infernix::TransferWork{page_payload * pages, run_operations[pages - 1]},
                   label + " contiguous-run quote differs from its physical copy work");
    }
    const auto export_work = source.copy_to_host(source_handles, host_view, context.stream);
    failures += expect(export_work == infernix::TransferWork{5 * page_payload,
                                                           run_operations[2] + run_operations[1]},
                       label + " D2H work missed fragmentation or counts Host padding");
    context.synchronize();

    const std::vector<std::byte> expected =
        expected_host_records(source, physical_mapping, host_layout, device_bytes);
    failures += expect(std::memcmp(host_view.data(), expected.data(), expected.size()) == 0,
                       label + " D2H canonical page records differ from Device payload");

    std::vector<infernix::DeviceKVPageLease> restored                = materialize(destination, 5);
    const std::vector<infernix::DeviceKVPageHandle> restored_handles = handles(restored);
    destination.zero_pages(restored_handles, context.stream);
    const auto restore_work =
        destination.copy_from_host(host_arena.view(*host), restored_handles, context.stream);
    failures +=
        expect(restore_work == infernix::TransferWork{5 * page_payload, run_operations.back()},
               label + " H2D work differs from the submitted contiguous run");

    const std::array duplicate_destinations{restored[0].handle(), restored[0].handle()};
    bool duplicate_zero_rejected = false;
    try {
        destination.zero_pages(duplicate_destinations, context.stream);
    } catch (const std::invalid_argument&) { duplicate_zero_rejected = true; }
    failures += expect(duplicate_zero_rejected, label + " duplicate zero destination accepted");
    bool duplicate_restore_rejected = false;
    try {
        destination.copy_from_host(host_arena.view(*host).subview(0, 2), duplicate_destinations,
                                   context.stream);
    } catch (const std::invalid_argument&) { duplicate_restore_rejected = true; }
    failures += expect(duplicate_restore_rejected, label + " duplicate H2D destination accepted");

    std::optional<infernix::HostKVAllocation> roundtrip = host_arena.allocate(host_layout, 5);
    failures += expect(roundtrip.has_value(), label + " roundtrip Host allocation failed");
    infernix::HostKVAllocationView roundtrip_view = host_arena.writable_view(*roundtrip);
    std::memset(roundtrip_view.data(), 0, host_layout.page_stride * roundtrip_view.page_count());
    destination.copy_to_host(restored_handles, roundtrip_view, context.stream);
    context.synchronize();
    failures += expect(std::memcmp(roundtrip_view.data(), host_view.data(), expected.size()) == 0,
                       label + " Device -> Host -> Device roundtrip changed bytes");

    const auto local_work =
        destination.copy_page(restored[0].handle(), restored[4].handle(), context.stream);
    const infernix::TransferWork expected_local{page_payload,
                                              static_cast<std::uint32_t>(geometry.planes.size())};
    failures +=
        expect(local_work == expected_local && destination.device_copy_work(1) == expected_local &&
                   destination.device_copy_work(0) == infernix::TransferWork{},
               label + " D2D work differs from one full page-group copy");
    failures += expect(destination.copy_page(restored[0].handle(), restored[0].handle(),
                                             context.stream) == infernix::TransferWork{},
                       label + " D2D self-copy must submit no work");
    // Record-addressed copies into caller-owned slabs (the hybrid prefix cache Host tier): records
    // 3, 4, 5 advance by one constant pitch larger than a page record and coalesce into one
    // strided run over consecutive physical pages; the rest are scattered and descending.
    {
        const std::size_t pitch = host_layout.page_stride + 4096;
        infernix::PinnedHostBuffer slabs(pitch * 10);
        std::memset(slabs.data(), 0xa5, pitch * 10);
        auto* base = static_cast<std::byte*>(slabs.data());
        const std::array<std::size_t, 5> slab_of{3, 4, 5, 9, 0};
        std::vector<std::byte*> records;
        for (const std::size_t slab : slab_of) { records.push_back(base + slab * pitch); }
        source.copy_to_host_records(source_handles, records, {}, host_layout, context.stream);
        context.synchronize();
        // Plane payloads only: alignment padding between planes is not written (the slabs keep
        // their 0xa5 fill there), as in every other Host record.
        bool records_match = true;
        for (std::size_t page = 0; page < records.size(); ++page) {
            for (const infernix::HostKVPlaneLayout& plane : host_layout.planes) {
                records_match = records_match &&
                                std::memcmp(records[page] + plane.offset,
                                            expected.data() + page * host_layout.page_stride +
                                                plane.offset,
                                            plane.page_payload_bytes) == 0;
            }
        }
        failures += expect(records_match, label + " D2H page records differ from Device payload");
        failures += expect(std::to_integer<unsigned>(*(base + 1 * pitch)) == 0xa5U,
                           label + " D2H records wrote an unselected slab");

        std::vector<infernix::DeviceKVPageLease> from_records          = materialize(destination, 5);
        const std::vector<infernix::DeviceKVPageHandle> record_handles = handles(from_records);
        destination.zero_pages(record_handles, context.stream);
        std::vector<const std::byte*> const_records(records.begin(), records.end());
        destination.copy_from_host_records(const_records, {}, record_handles, host_layout,
                                           context.stream);
        std::optional<infernix::HostKVAllocation> record_roundtrip =
            host_arena.allocate(host_layout, 5);
        infernix::HostKVAllocationView record_view = host_arena.writable_view(*record_roundtrip);
        std::memset(record_view.data(), 0, host_layout.page_stride * 5);
        destination.copy_to_host(record_handles, record_view, context.stream);
        context.synchronize();
        failures += expect(std::memcmp(record_view.data(), expected.data(), expected.size()) == 0,
                           label + " Device -> records -> Device roundtrip changed bytes");

        // Plane ranges: one plane at a time, last plane first, rebuilds the same pages (a hybrid
        // restore lands each model layer's planes separately).
        destination.zero_pages(record_handles, context.stream);
        for (std::size_t plane = destination.plane_count(); plane-- > 0;) {
            destination.copy_from_host_records(const_records, {}, record_handles, host_layout,
                                               plane, plane + 1, context.stream);
        }
        std::memset(record_view.data(), 0, host_layout.page_stride * 5);
        destination.copy_to_host(record_handles, record_view, context.stream);
        context.synchronize();
        failures += expect(std::memcmp(record_view.data(), expected.data(), expected.size()) == 0,
                           label + " per-plane record copies differ from the whole-record copy");
        bool range_rejected = false;
        try {
            destination.copy_from_host_records(const_records, {}, record_handles, host_layout, 0,
                                               destination.plane_count() + 1, context.stream);
        } catch (const std::invalid_argument&) { range_rejected = true; }
        failures += expect(range_rejected, label + " out-of-range plane range accepted");
        bool mismatched_rejected = false;
        try {
            destination.copy_from_host_records(
                std::span<const std::byte* const>(const_records.data(), 2), {}, record_handles,
                host_layout, context.stream);
        } catch (const std::invalid_argument&) { mismatched_rejected = true; }
        failures += expect(mismatched_rejected, label + " record count mismatch accepted");

        // Records in two separate pinned allocations (the hybrid Host tier pins its slabs in
        // chunks): consecutive pages whose records lie in different groups must not form one
        // strided transfer across both allocations, whatever the distance between them.
        infernix::PinnedHostBuffer first_chunk(host_layout.page_stride * 3);
        infernix::PinnedHostBuffer second_chunk(host_layout.page_stride * 2);
        auto* first_base  = static_cast<std::byte*>(first_chunk.data());
        auto* second_base = static_cast<std::byte*>(second_chunk.data());
        std::vector<std::byte*> split{first_base, first_base + host_layout.page_stride,
                                      first_base + 2 * host_layout.page_stride, second_base,
                                      second_base + host_layout.page_stride};
        const std::array<std::uint32_t, 5> split_groups{0, 0, 0, 1, 1};
        source.copy_to_host_records(source_handles, split, split_groups, host_layout,
                                    context.stream);
        destination.zero_pages(record_handles, context.stream);
        const std::vector<const std::byte*> const_split(split.begin(), split.end());
        destination.copy_from_host_records(const_split, split_groups, record_handles, host_layout,
                                           context.stream);
        std::memset(record_view.data(), 0, host_layout.page_stride * 5);
        destination.copy_to_host(record_handles, record_view, context.stream);
        context.synchronize();
        failures += expect(std::memcmp(record_view.data(), expected.data(), expected.size()) == 0,
                           label + " records split over two pinned allocations changed bytes");
        bool groups_rejected = false;
        try {
            destination.copy_from_host_records(
                const_split, std::span<const std::uint32_t>(split_groups.data(), 2), record_handles,
                host_layout, context.stream);
        } catch (const std::invalid_argument&) { groups_rejected = true; }
        failures += expect(groups_rejected, label + " record group count mismatch accepted");
        record_roundtrip->release();
    }

    destination.copy_page(restored[0].handle(), restored[4].handle(), context.stream);
    const infernix::DeviceKVPageHandle copied[] = {restored[0].handle(), restored[4].handle()};
    std::optional<infernix::HostKVAllocation> copied_host = host_arena.allocate(host_layout, 2);
    infernix::HostKVAllocationView copied_view            = host_arena.writable_view(*copied_host);
    std::memset(copied_view.data(), 0, host_layout.page_stride * 2);
    destination.copy_to_host(copied, copied_view, context.stream);
    context.synchronize();
    const infernix::HostKVAllocationConstView copied_contents = host_arena.view(*copied_host);
    failures += expect(page_payload_equal(copied_contents, 0, copied_contents, 1),
                       label + " D2D copy did not cover the complete page-group");
    failures += expect(page_payload_equal(copied_contents, 0, host_arena.view(*host), 0),
                       label + " D2D copy changed its source page");

    const infernix::DeviceKVPageHandle zeroed[] = {restored[1].handle()};
    destination.zero_pages(zeroed, context.stream);
    const infernix::DeviceKVPageHandle zero_observation[] = {restored[0].handle(),
                                                           restored[1].handle()};
    std::optional<infernix::HostKVAllocation> zero_host   = host_arena.allocate(host_layout, 2);
    infernix::HostKVAllocationView zero_view              = host_arena.writable_view(*zero_host);
    std::memset(zero_view.data(), 0x5a, host_layout.page_stride * 2);
    destination.copy_to_host(zero_observation, zero_view, context.stream);
    context.synchronize();
    const infernix::HostKVAllocationConstView zero_contents = host_arena.view(*zero_host);
    failures += expect(page_payload_equal(zero_contents, 0, host_arena.view(*host), 0),
                       label + " selective zero changed an unselected page");
    failures += expect(page_payload_zero(zero_contents, 1),
                       label + " selective zero left page payload bytes");

    {
        infernix::KVExecutionTablePool destination_tables(
            {destination_arena.base(), destination_arena.capacity()}, destination_plan.tables,
            destination);
        auto destination_row = destination_tables.acquire(0);
        const std::array scattered{restored[4].handle(), restored[1].handle(), restored[3].handle(),
                                   restored[0].handle(), restored[2].handle()};
        destination_tables.publish(destination_row.handle(), 0, scattered, context.stream);
        context.synchronize();
        const auto destination_mapping =
            read_mapping(destination_tables.row(destination_row.handle()), scattered.size());

        auto ordered_host = host_arena.allocate(host_layout, 7);
        if (!ordered_host) { throw std::bad_alloc(); }
        const auto guarded = host_arena.writable_view(*ordered_host);
        std::memset(guarded.data(), 0xa5, host_layout.page_stride * guarded.page_count());
        const auto ordered_view = guarded.subview(1, 5);

        // Both producers precede the transfers. H2D reads pinned Host bytes written by the earlier
        // D2H in this stream; no CPU wait or read separates those dependent operations.
        const auto ordered_source = fill_device_pool(source, context.stream, 89);
        auto expected_device      = fill_device_pool(destination, context.stream, 37);
        source.copy_to_host(source_handles, ordered_view, context.stream);
        const auto scattered_work = destination.copy_from_host(
            host_arena.view(*ordered_host).subview(1, 5), scattered, context.stream);
        failures += expect(scattered_work ==
                               infernix::TransferWork{5 * page_payload, 5 * run_operations.front()},
                           label + " scattered H2D work missed the single-page lowering");
        std::vector<std::vector<unsigned char>> observed_device(destination.plane_count());
        for (std::size_t index = 0; index < destination.plane_count(); ++index) {
            const auto& plane = destination.plane(index);
            auto& observed    = observed_device[index];
            observed.resize(plane.bytes());
            CUDA_CHECK(cudaMemcpyAsync(observed.data(), plane.data, observed.size(),
                                       cudaMemcpyDeviceToHost, context.stream));
        }
        context.synchronize();

        const auto ordered_expected =
            expected_host_records(source, physical_mapping, host_layout, ordered_source);
        for (std::size_t logical = 0; logical < scattered.size(); ++logical) {
            const auto physical = destination_mapping[logical];
            for (std::size_t index = 0; index < destination.plane_count(); ++index) {
                const auto& plane          = destination.plane(index);
                const auto& plane_geometry = geometry.planes[index];
                const auto& host_plane     = host_layout.planes[index];
                const auto element_bytes   = infernix::dtype_size(plane_geometry.dtype);
                const auto* canonical =
                    ordered_expected.data() + logical * host_layout.page_stride + host_plane.offset;
                failures +=
                    expect(std::memcmp(ordered_view.data() + logical * host_layout.page_stride +
                                           host_plane.offset,
                                       canonical, host_plane.page_payload_bytes) == 0,
                           label + " stream-ordered D2H payload differs from its producer");
                for (std::int32_t head = 0; head < plane_geometry.head_extent; ++head) {
                    const auto device_head =
                        geometry.device_plane_order == infernix::PagedKVPlaneOrder::PageMajor
                            ? head * plane.nb[2] + physical * plane.nb[3]
                            : physical * plane.nb[2] + head * plane.nb[3];
                    for (std::int32_t token = 0; token < infernix::kPagedKVPageSize; ++token) {
                        for (std::int32_t element = 0; element < plane_geometry.leading_extent;
                             ++element) {
                            for (std::size_t byte = 0; byte < element_bytes; ++byte) {
                                const auto host_offset =
                                    head * host_plane.head_payload_bytes +
                                    (token * plane_geometry.leading_extent + element) *
                                        element_bytes +
                                    byte;
                                const auto device_offset = device_head + token * plane.nb[1] +
                                                           element * plane.nb[0] + byte;
                                expected_device[index][device_offset] =
                                    std::to_integer<unsigned char>(canonical[host_offset]);
                            }
                        }
                    }
                }
            }
        }
        failures += expect(observed_device == expected_device,
                           label + " scattered H2D raw Device bytes differ from the scalar oracle");
        const auto is_guard = [&](std::uint32_t page) {
            const auto* first = guarded.data() + page * host_layout.page_stride;
            return std::all_of(first, first + host_layout.page_stride,
                               [](std::byte value) { return value == std::byte{0xa5}; });
        };
        failures += expect(is_guard(0) && is_guard(6),
                           label + " Host transfer overwrote a page outside its subview");
    }

    std::optional<infernix::HostKVAllocation> split_source = host_arena.allocate(host_layout, 4);
    failures += expect(split_source.has_value(), label + " split source allocation failed");
    infernix::HostKVAllocationView stale_view = host_arena.writable_view(*split_source);
    auto [left, right]                      = host_arena.split(std::move(*split_source), 1);
    failures += expect(!stale_view.valid(), label + " split did not invalidate the old capability");
    failures += expect_size(left.page_count(), 1, label + " split left pages");
    failures += expect_size(right.page_count(), 3, label + " split right pages");
    auto [middle, tail] = host_arena.split(std::move(right), 1);
    middle.release();
    std::optional<infernix::HostKVAllocation> reused = host_arena.allocate(host_layout, 1);
    failures += expect(reused.has_value(), label + " released Host subextent was not reusable");

    (void)left;
    (void)tail;
    (void)blockers;
    return failures;
}

} // namespace

int main() {
    int device_count              = 0;
    const cudaError_t count_error = cudaGetDeviceCount(&device_count);
    if (cuda_unavailable(count_error) || (count_error == cudaSuccess && device_count == 0)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (count_error != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_error) << '\n';
        return 1;
    }

    try {
        infernix::DeviceContext context(0);
        int failures = exercise_reservation_and_mapping(context);
        failures += exercise_backed_limit(context);
        failures += exercise_vmm_range();
        failures += exercise_republish_while_copy_queued(context);
        failures += exercise_layout_and_transfer(
            context,
            infernix::KVPageGeometry{
                .device_plane_order = infernix::PagedKVPlaneOrder::PageMajor,
                .planes =
                    {
                        {infernix::DType::I8, 8, 2, 256},
                        {infernix::DType::FP16, 1, 2, 256},
                    },
            },
            "PageMajor", {2, 2, 2, 2});
        failures += exercise_layout_and_transfer(
            context,
            infernix::KVPageGeometry{
                .device_plane_order = infernix::PagedKVPlaneOrder::HeadMajor,
                .planes =
                    {
                        {infernix::DType::BF16, 8, 3, 256},
                        {infernix::DType::FP16, 2, 3, 256},
                    },
            },
            "HeadMajor", {2, 4, 6, 6});
        failures += exercise_layout_and_transfer(
            context,
            infernix::KVPageGeometry{
                .device_plane_order = infernix::PagedKVPlaneOrder::PageMajor,
                .planes =
                    {
                        {infernix::DType::FP8_E4M3FN, 256, 2, 256},
                        {infernix::DType::U8, 128, 2, 256},
                        {infernix::DType::FP16, 1, 2, 256},
                        {infernix::DType::U8, 16, 2, 256},
                    },
            },
            "K8V4 asymmetric PageMajor", {4, 4, 4, 4});
        failures += exercise_layout_and_transfer(
            context,
            infernix::KVPageGeometry{
                .device_plane_order = infernix::PagedKVPlaneOrder::PageMajor,
                .planes =
                    {
                        {infernix::DType::I8, 8, 2, 256},
                        {infernix::DType::I8, 8, 2, 256},
                        {infernix::DType::FP16, 1, 2, 256},
                        {infernix::DType::FP16, 1, 2, 256},
                    },
            },
            "Grouped PageMajor", {2, 4, 4, 4});
        failures += exercise_layout_and_transfer(
            context,
            infernix::KVPageGeometry{
                .device_plane_order = infernix::PagedKVPlaneOrder::PageMajor,
                .planes =
                    {
                        {infernix::DType::I8, 1, 3, 256},
                        {infernix::DType::I8, 1, 3, 256},
                        {infernix::DType::I8, 1, 3, 256},
                        {infernix::DType::I8, 1, 3, 256},
                    },
            },
            "Grouped padded PageMajor", {1, 2, 3, 4});
        if (failures != 0) {
            std::cerr << failures << " Paged KV physical-container checks failed\n";
            return 1;
        }
        std::cout << "Paged KV physical-container checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Paged KV physical-container test failed: " << error.what() << '\n';
        return 1;
    }
}
