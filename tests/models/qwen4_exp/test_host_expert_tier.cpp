// The SSD tier's host level (design §19.3.7, R10 host part): HostExpertTier over a synthetic two-file
// "artifact" with a record straddling the files, plain aligned memory as slots. Checks against a
// closed-form byte oracle: the startup pre-fill, demand reads from another thread (the CPU
// service's role) including a straddling record, a resident key served without a read, duplicate
// demands sharing a ticket, admission of hot landings at the next boundary (the victims lose their
// host copy, the dirty set names both), ring exhaustion reusing finished slots, a failed read,
// destruction with reads in flight, unpinned records (a refused pin chunk becoming unpinned
// records, their startup fill, CPU demands served in place and never admitted), a RAM resize
// (pinned chunks and unpinned records given back and taken again, refilled from the file), and an
// EngineHost's slot pool (tiers keeping the records their reused chunks hold). CPU and disk only,
// except with --fetch: the tier as the responder
// of a fetch channel (mapped words, so a CUDA context), this thread playing the device's half of the
// protocol (ops/offloaded_sparse_moe/cpu/fetch_request.h): records landed from RAM and read from disk,
// more records than ring slots (slots reused as the "device" consumes them), the heartbeat, a request
// failing on an unreadable record, and admission of fetched landings at the next boundary. An
// expert stripe moves each record's tail to another file; records still land whole. CSF slots (Options::csf, real
// Flash-Next record geometry): the fill and the refills compress records in their slots (one that cannot hold its
// table stays expanded), published tagged; CPU demands, admissions and copied demotions; an EngineHost's pool keeping
// each slot's format; with --fetch, a fetch of a compressed resident landing an expanded copy.
#include "test_temp_path.h"
#include "core/host_memory_worker.h"
#include "runtime/expert_offload/expert_store.h"
#include "runtime/expert_offload/host_expert_tier.h"
#include "runtime/expert_offload/host_slot_pool.h"
#include "ops/offloaded_sparse_moe/cpu/fetch_channel.h"
#include "ops/offloaded_sparse_moe/csf_record.h"

#include <algorithm>
#include <atomic>
#include <memory>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#ifdef _WIN32
#    include <windows.h>
#endif

namespace {

using infernix::runtime::expert_offload::ExpertStore;
using infernix::runtime::expert_offload::HostExpertTier;
namespace expert_cache = infernix::runtime::expert_offload::expert_cache;

constexpr std::uint32_t kLayers = 3, kExperts = 40, kKeys = kLayers * kExperts;
constexpr std::uint64_t kRecord = 8192; // two 4 KiB blocks
constexpr std::uint32_t kStraddle = 60; // the key whose record spans both files

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::cerr << "FAIL: " << what << '\n';
    }
}

// A RAM resize's memory work (releasing, page-locking, decommitting, committing) runs on a worker, as in the Engine;
// these wait for it as the next maintenance would find it done. The worker makes no CUDA device current.
infernix::HostMemoryWorker& ram_worker() {
    static infernix::HostMemoryWorker worker;
    return worker;
}

void settle_worker() {
    while (!ram_worker().idle()) { std::this_thread::yield(); }
}

std::uint64_t shrink(HostExpertTier& tier, std::uint64_t bytes) {
    const std::uint64_t released = tier.shrink(bytes, ram_worker());
    settle_worker();
    return released;
}

// A growth as the idle Engine runs it: started, linked once it landed, and the room filled from the artifact.
std::uint64_t grow(HostExpertTier& tier, std::uint64_t bytes) {
    if (tier.begin_growth(bytes, ram_worker()) == 0) { return 0; }
    while (!tier.growth_landed()) { std::this_thread::yield(); }
    const std::uint64_t linked = tier.link_growth();
    (void)tier.fill_room();
    return linked;
}

std::uint8_t oracle(std::uint32_t key, std::uint64_t i) {
    return static_cast<std::uint8_t>((key * 131U + i * 7U + (i >> 9U)) & 0xFFU);
}

bool matches(const std::uint8_t* p, std::uint32_t key) {
    if (p == nullptr) { return false; }
    for (std::uint64_t i = 0; i < kRecord; ++i) {
        if (p[i] != oracle(key, i)) { return false; }
    }
    return true;
}

// Keys [0, kStraddle) and the first half of kStraddle in file A after a 4 KiB header; the second half
// of kStraddle and the rest in file B. Key kKeys - 1 points past file B's end (an unreadable record).
struct Files {
    std::filesystem::path a, b;
    std::vector<std::vector<ExpertStore::Segment>> records;
};

Files make_files() {
    Files f;
    f.a = infernix::test::unique_temp_path("infernix_host_tier_a.bin");
    f.b = infernix::test::unique_temp_path("infernix_host_tier_b.bin");
    std::ofstream a(f.a, std::ios::binary | std::ios::trunc), b(f.b, std::ios::binary | std::ios::trunc);
    std::vector<char> header(4096, 0x11);
    a.write(header.data(), static_cast<std::streamsize>(header.size()));
    std::vector<char> rec(kRecord);
    f.records.resize(kKeys);
    std::uint64_t at_a = 4096, at_b = 0;
    for (std::uint32_t key = 0; key + 1 < kKeys; ++key) {
        for (std::uint64_t i = 0; i < kRecord; ++i) { rec[i] = static_cast<char>(oracle(key, i)); }
        if (key < kStraddle) {
            a.write(rec.data(), static_cast<std::streamsize>(kRecord));
            f.records[key] = {{0, at_a, 0, static_cast<std::uint32_t>(kRecord)}};
            at_a += kRecord;
        } else if (key == kStraddle) {
            a.write(rec.data(), 4096);
            b.write(rec.data() + 4096, 4096);
            f.records[key] = {{0, at_a, 0, 4096}, {1, 0, 4096, 4096}};
            at_a += 4096;
            at_b += 4096;
        } else {
            b.write(rec.data(), static_cast<std::streamsize>(kRecord));
            f.records[key] = {{1, at_b, 0, static_cast<std::uint32_t>(kRecord)}};
            at_b += kRecord;
        }
    }
    f.records[kKeys - 1] = {{1, at_b + 64 * kRecord, 0, static_cast<std::uint32_t>(kRecord)}}; // past the end
    if (!a || !b) { throw std::runtime_error("cannot write the test files"); }
    return f;
}

HostExpertTier::Options options() {
    HostExpertTier::Options o;
    o.slots           = 64; // ring 8 + prefetch 2 + demotion 4 + 50 resident slots
    o.ring            = 8;
    o.prefetch        = 2;
    o.demotion        = 4;
    o.half_life       = 8;
    o.slots_per_chunk = 16;
    o.allocate = [](std::size_t bytes) { return ::operator new(bytes, std::align_val_t{4096}); };
    o.release  = [](void* p) { ::operator delete(p, std::align_val_t{4096}); };
    return o;
}

// Demands `keys` from a separate thread as the CPU service does, one layer call of `per_call` keys
// at a time: every demand of the call first, then each wait and done. A record's bytes are checked
// before done (`valid`), as the service consumes them then; a finished landing's slot may be reused.
std::vector<const std::uint8_t*> serve(HostExpertTier& tier, const std::vector<std::uint32_t>& keys,
                                       std::vector<std::uint32_t>& statuses, std::vector<bool>& valid,
                                       std::size_t per_call = 6) {
    std::vector<const std::uint8_t*> out(keys.size());
    statuses.assign(keys.size(), 0);
    valid.assign(keys.size(), false);
    std::thread service([&] {
        for (std::size_t first = 0; first < keys.size(); first += per_call) {
            const std::size_t last = std::min(keys.size(), first + per_call);
            std::vector<std::uint32_t> tickets;
            for (std::size_t i = first; i < last; ++i) {
                tickets.push_back(tier.demand(static_cast<int>(keys[i] / kExperts), static_cast<int>(keys[i] % kExperts)));
            }
            for (std::size_t i = first; i < last; ++i) {
                out[i]   = tier.wait(tickets[i - first], statuses[i]);
                valid[i] = statuses[i] == 0 && matches(out[i], keys[i]);
                tier.done(tickets[i - first]);
            }
        }
    });
    service.join();
    return out;
}

// The device's half of the fetch protocol, played on the channel's mapped words.
struct DeviceSide {
    infernix::ops::MoeFetchChannel channel;
    std::uint32_t sequence = 0;

    void publish(std::uint32_t layer, const std::vector<std::int32_t>& experts) {
        for (std::size_t i = 0; i < experts.size(); ++i) { channel.request->expert[i] = experts[i]; }
        channel.request->layer = static_cast<std::int32_t>(layer);
        channel.request->count = static_cast<std::int32_t>(experts.size());
        std::atomic_thread_fence(std::memory_order_seq_cst);
        *reinterpret_cast<volatile std::uint32_t*>(&channel.request->sequence) = ++sequence;
    }
    [[nodiscard]] std::uint32_t beat() const { return *reinterpret_cast<const volatile std::uint32_t*>(channel.heartbeat); }
    // Waits (up to 5 s) until record i of the current request has landed: its address, or null once
    // the request failed (`status`) or the wait timed out.
    const std::uint8_t* wait(std::uint32_t i, std::uint32_t& status) const {
        const auto* response = reinterpret_cast<const volatile infernix::ops::offloaded_moe::FetchResponse*>(channel.response);
        const auto until     = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (std::chrono::steady_clock::now() < until) {
            if (response->sequence == sequence) {
                std::atomic_thread_fence(std::memory_order_seq_cst);
                status = response->status;
                if (status != 0) { return nullptr; }
                if (response->landed > i) {
                    std::atomic_thread_fence(std::memory_order_seq_cst);
                    return reinterpret_cast<const std::uint8_t*>(response->record[i]);
                }
            }
            std::this_thread::yield();
        }
        status = ETIMEDOUT;
        return nullptr;
    }
    void consume(std::uint32_t records) {
        *reinterpret_cast<volatile std::uint64_t*>(channel.consumed) = static_cast<std::uint64_t>(sequence) << 32U | records;
    }
};

void test_fetch(const ExpertStore& store) {
    infernix::ops::offloaded_moe::FetchChannel channel;
    auto o  = options();
    o.fetch = &channel;
    HostExpertTier tier(store, o);
    std::vector<std::uint32_t> ranked(kKeys);
    for (std::uint32_t k = 0; k < kKeys; ++k) { ranked[k] = k; }
    check(tier.prefill(ranked) == 50, "fetch: pre-fill fills the resident slots");
    const std::uint64_t runs_before = tier.stats().run_reads, run_records_before = tier.stats().run_records;
    DeviceSide device{channel.channel(0)};
    tier.begin_round(32);

    // Layer 1, experts 0..19 = keys 40..59: 40..49 are in RAM, 50..59 read from disk through the
    // 8-slot ring, consumed one record at a time.
    std::vector<std::int32_t> experts;
    for (std::int32_t e = 0; e < 20; ++e) { experts.push_back(e); }
    const std::uint32_t beat_before = device.beat();
    device.publish(1, experts);
    bool all = true, from_ram = true;
    for (std::uint32_t i = 0; i < experts.size(); ++i) {
        std::uint32_t status = 0;
        const std::uint8_t* record = device.wait(i, status);
        all = all && status == 0 && matches(record, kExperts + i);
        if (i < 10) { from_ram = from_ram && record == tier.record(kExperts + i); }
        device.consume(i + 1);
    }
    check(all, "fetch: 20 records (10 from RAM, 10 through an 8-slot ring) land in job order with their bytes");
    tier.end_round();
    tier.begin_round(32); // publishes the agent's counters
    check(tier.stats().run_reads > runs_before &&
              tier.stats().run_records - run_records_before > tier.stats().run_reads - runs_before,
          "fetch: neighbouring records of one request are read together");
    check(from_ram, "fetch: a key in RAM lands at its RAM slot without a read");
    check(device.beat() != beat_before, "fetch: the heartbeat advances while a round is open");

    // Layer 2, experts 30..39 = keys 110..119: key 119 cannot be read, so the request fails.
    experts.clear();
    for (std::int32_t e = 30; e < 40; ++e) { experts.push_back(e); }
    device.publish(2, experts);
    std::uint32_t status = 0, wrong = 0;
    for (std::uint32_t i = 0; i < experts.size(); ++i) {
        const std::uint8_t* record = device.wait(i, status);
        if (record == nullptr) { break; }
        wrong += matches(record, 2 * kExperts + 30 + i) ? 0U : 1U;
        device.consume(i + 1);
    }
    check(status == EIO, "fetch: a request with an unreadable record fails with EIO");
    check(wrong == 0, "fetch: every record landed before the failure holds its bytes");
    tier.end_round();

    // The hot fetched keys are admitted at the next boundary (records still in their ring slots).
    std::vector<std::uint32_t> hot;
    for (std::uint32_t k = 2 * kExperts + 30; k < 2 * kExperts + 39; ++k) { hot.push_back(k); }
    tier.record_uses(1, hot, 50.0);
    tier.begin_round(32);
    std::uint32_t admitted = 0;
    for (const auto k : hot) { admitted += matches(tier.record(k), k) ? 1U : 0U; }
    std::cout << "fetch: " << tier.stats().fetch_requests << " requests, " << tier.stats().fetch_records
              << " records (" << tier.stats().fetch_from_ram << " from RAM), " << tier.stats().demand_reads
              << " reads, " << admitted << " of 9 hot landings admitted\n";
    check(tier.stats().fetch_requests == 2 && tier.stats().fetch_from_ram == 10, "fetch: request statistics");
    check(admitted > 0, "fetch: hot fetched landings are admitted into RAM");
    tier.controller().check();
    tier.end_round();
}

// One request of `layer`'s `experts`, consumed record by record: whether every record landed with
// its bytes.
bool fetch_all(DeviceSide& device, std::uint32_t layer, const std::vector<std::int32_t>& experts) {
    device.publish(layer, experts);
    bool all = true;
    for (std::uint32_t i = 0; i < experts.size(); ++i) {
        std::uint32_t status = 0;
        const std::uint8_t* record = device.wait(i, status);
        all = all && status == 0 && matches(record, layer * kExperts + static_cast<std::uint32_t>(experts[i]));
        device.consume(i + 1);
    }
    return all;
}

// Read-ahead (Options::ahead_min, ahead): a wide request also reads the next layer's first records
// without a host copy, behind its own; the next request adopts them, so each record is read once
// and lands whole. A request naming none of them releases their slots, and the round still ends.
void test_read_ahead(const ExpertStore& store) {
    infernix::ops::offloaded_moe::FetchChannel channel;
    auto o      = options();
    o.fetch     = &channel;
    o.ahead_min = 16;
    o.ahead     = 6;
    HostExpertTier tier(store, o);
    std::vector<std::uint32_t> ranked(kKeys);
    for (std::uint32_t k = 0; k < kKeys; ++k) { ranked[k] = k; }
    check(tier.prefill(ranked) == 50, "read-ahead: pre-fill fills the resident slots (keys 0-49)");
    DeviceSide device{channel.channel(0)};
    tier.begin_round(32);
    const std::uint64_t reads_before = tier.stats().demand_reads;
    std::vector<std::int32_t> experts;
    for (std::int32_t e = 0; e < 20; ++e) { experts.push_back(e); }
    // Layer 1, keys 40..59 (50..59 from disk): wide, so layer 2's keys 80..85 are read ahead.
    check(fetch_all(device, 1, experts), "read-ahead: the wide request's records land with their bytes");
    // Layer 2, keys 80..89: 80..85 adopted from the read-ahead, 86..89 read now.
    experts.resize(10);
    check(fetch_all(device, 2, experts), "read-ahead: adopted records land with their bytes");
    tier.end_round();
    tier.begin_round(32); // publishes the agent's counters
    std::cout << "read-ahead: " << tier.stats().demand_reads - reads_before << " reads for 20 records from disk\n";
    check(tier.stats().demand_reads - reads_before == 20, "read-ahead: an adopted record is read once");
    // A wide request of layer 0 (keys 20..39, in RAM) reads ahead layer 1's records without a host
    // copy; the next request (layer 2, keys 110..118) names none of them.
    experts.clear();
    for (std::int32_t e = 20; e < 40; ++e) { experts.push_back(e); }
    check(fetch_all(device, 0, experts), "read-ahead: a request served from RAM lands");
    experts.clear();
    for (std::int32_t e = 30; e < 39; ++e) { experts.push_back(e); }
    check(fetch_all(device, 2, experts), "read-ahead: a request naming no read-ahead record lands whole");
    tier.end_round();
    tier.begin_round(32); // drains every read, the unadopted read-ahead included
    tier.controller().check();
    tier.end_round();
}

// Stream slots (Options::stream), lent from the resident slots: lending evicts the lowest-ranked
// residents; a read lands with the file's bytes whether or not a round is open and whatever the
// key's RAM state; the slots run out and come back on release; the loan returns them to RAM.
void test_stream_reads(const ExpertStore& store) {
    auto o   = options();
    o.stream = 4;
    HostExpertTier tier(store, o);
    std::vector<std::uint32_t> ranked(kKeys);
    for (std::uint32_t k = 0; k < kKeys; ++k) { ranked[k] = k; }
    check(tier.prefill(ranked) == 50, "stream reads: pre-fill fills the resident slots (keys 0-49)");
    check(tier.stream_slots() == 0 && tier.stream_read(60) == -1, "stream reads: none before a loan");
    check(tier.lend_stream(10) == 4 && tier.stream_slots() == 4, "stream reads: the loan is capped by the option");
    std::uint32_t in_ram = 0;
    for (std::uint32_t k = 0; k < 50; ++k) { in_ram += tier.record(k) != nullptr ? 1U : 0U; }
    check(in_ram == 46, "stream reads: the loan evicts four residents");
    tier.controller().check();
    // Between rounds: keys 60..62 (SSD-only) and 3 (in RAM).
    const std::uint32_t keys[] = {60, 61, 62, 3};
    std::int32_t slots[4];
    for (int i = 0; i < 4; ++i) { slots[i] = tier.stream_read(keys[i]); }
    check(slots[0] >= 0 && slots[1] >= 0 && slots[2] >= 0 && slots[3] >= 0, "stream reads: four slots for four reads");
    check(tier.stream_read(63) == -1, "stream reads: none left for a fifth");
    bool all = true;
    for (int i = 0; i < 4; ++i) { all = all && matches(tier.stream_wait(slots[i]), keys[i]); }
    check(all, "stream reads: every record lands with the file's bytes");
    // During an open round, after a release.
    tier.stream_release(slots[1]);
    tier.begin_round(32);
    const std::int32_t again = tier.stream_read(70);
    check(again >= 0 && matches(tier.stream_wait(again), 70), "stream reads: a released slot reads again during a round");
    tier.end_round();
    tier.begin_round(32);
    std::cout << "stream reads: " << tier.stats().stream_reads << " records read into stream slots\n";
    check(tier.stats().stream_reads == 5, "stream reads: the agent counts every stream read");
    tier.end_round();
    bool refused = false;
    try {
        tier.return_stream();
    } catch (const std::logic_error&) { refused = true; }
    check(refused, "stream reads: the loan cannot return while a slot is in use");
    for (const std::int32_t s : {slots[0], slots[2], slots[3], again}) { tier.stream_release(s); }
    tier.return_stream();
    tier.controller().check();
    check(tier.stream_slots() == 0 && tier.prefill(ranked) == 4, "stream reads: the returned slots refill RAM");
    // Destruction with a stream read in flight.
    check(tier.lend_stream(4) == 4, "stream reads: a second loan");
    (void)tier.stream_read(71);
}

// A GGUF import's layers: records of different pitches (layer 0 two blocks, layer 1 one) and no
// scale tail. Slots take the largest pitch; a record shorter than its slot lands whole, alone and in
// a scatter run, and the bytes past it are never read as its own.
void test_mixed_strides() {
    constexpr std::uint32_t kExpertsHere = 20;
    const std::uint64_t pitch[2] = {kRecord, kRecord / 2};
    const auto path = infernix::test::unique_temp_path("infernix_host_tier_mixed.bin");
    std::vector<std::vector<ExpertStore::Segment>> records(2 * kExpertsHere);
    {
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        std::uint64_t at = 0;
        std::vector<char> rec(kRecord);
        for (std::uint32_t key = 0; key < 2 * kExpertsHere; ++key) {
            const std::uint64_t bytes = pitch[key / kExpertsHere];
            for (std::uint64_t i = 0; i < bytes; ++i) { rec[i] = static_cast<char>(oracle(key, i)); }
            f.write(rec.data(), static_cast<std::streamsize>(bytes));
            records[key] = {{0, at, 0, static_cast<std::uint32_t>(bytes)}};
            at += bytes;
        }
        if (!f) { throw std::runtime_error("cannot write the mixed-stride test file"); }
    }
    const auto holds = [&](const std::uint8_t* p, std::uint32_t key) {
        if (p == nullptr) { return false; }
        for (std::uint64_t i = 0; i < pitch[key / kExpertsHere]; ++i) {
            if (p[i] != oracle(key, i)) { return false; }
        }
        return true;
    };
    const ExpertStore store({path}, {pitch[0], pitch[1]}, kExpertsHere, records, {{}, {}});
    check(store.record_bytes() == kRecord && store.record_bytes(1) == kRecord / 2 && store.multipliers(1) == nullptr,
          "mixed strides: the largest pitch, each layer its own, no scale tail");
    HostExpertTier::Options o = options();
    // Transit: ring 8 + prefetch 2 + bounce 32 at the largest pitch; each class (one layer each) 2 demotion + 5
    // resident slots at its own pitch.
    o.slots = 42 + 2 * (2 + 5);
    check(HostExpertTier::slots_for_bytes(store, 42 * kRecord + 7 * kRecord + 7 * (kRecord / 2), o) == o.slots,
          "mixed strides: slots_for_bytes prices class slots at the mean record");
    HostExpertTier tier(store, o);
    check(tier.controller().class_count() == 2 && tier.pinned_slots() == o.slots &&
              tier.pinned_bytes() == 42 * kRecord + 7 * kRecord + 7 * (kRecord / 2),
          "mixed strides: a class per record size, each slot its record's size");
    // Layer 1's keys 20..29 follow each other in the file: one scatter run of half-size records into its 5 slots.
    std::vector<std::uint32_t> ranked;
    for (std::uint32_t k = kExpertsHere; k < kExpertsHere + 10; ++k) { ranked.push_back(k); }
    check(tier.prefill(ranked) == 5, "mixed strides: the pre-fill fills layer 1's class");
    bool all = true;
    for (std::uint32_t k = kExpertsHere; k < kExpertsHere + 5; ++k) { all = all && holds(tier.record(k), k); }
    check(all && tier.stats().run_reads == 1 && tier.stats().run_records == 5,
          "mixed strides: a run of short records lands in its class's slots");
    check(tier.record(kExpertsHere + 1) - tier.record(kExpertsHere) == static_cast<std::ptrdiff_t>(kRecord / 2),
          "mixed strides: half-size records are packed at their own pitch");
    std::vector<std::uint32_t> layer0;
    for (std::uint32_t k = 10; k < kExpertsHere; ++k) { layer0.push_back(k); }
    check(tier.prefill(layer0) == 5, "mixed strides: layer 0's class takes its own 5 keys");
    tier.begin_round(32);
    std::vector<const std::uint8_t*> got(3);
    std::vector<std::uint32_t> status(3);
    const std::uint32_t keys[3] = {3, kExpertsHere + 15, 7};
    std::thread service([&] {
        std::uint32_t tickets[3];
        for (int i = 0; i < 3; ++i) {
            tickets[i] = tier.demand(static_cast<int>(keys[i] / kExpertsHere), static_cast<int>(keys[i] % kExpertsHere));
        }
        for (int i = 0; i < 3; ++i) {
            got[i] = tier.wait(tickets[i], status[i]);
            got[i] = status[i] == 0 && holds(got[i], keys[i]) ? got[i] : nullptr;
            tier.done(tickets[i]);
        }
    });
    service.join();
    check(got[0] != nullptr && got[1] != nullptr && got[2] != nullptr,
          "mixed strides: demanded records of either pitch equal the file bytes");
    tier.end_round();
    tier.controller().check();
    // Hot landings are admitted at the next boundary: layer 1's (half size) by a copy into its class's slots,
    // layer 0's (the transit size) by swapping its ring slot into the class.
    const std::vector<std::uint32_t> hot{3, kExpertsHere + 15};
    tier.record_uses(1, hot, 50.0);
    tier.begin_round(32);
    const std::uint8_t* copied = tier.record(kExpertsHere + 15);
    const std::ptrdiff_t from_packed = copied - tier.record(kExpertsHere + 1);
    check(holds(copied, kExpertsHere + 15) && holds(tier.record(3), 3) && tier.stats().copied_admissions == 1,
          "mixed strides: both landings admitted with their bytes, the short one by a copy");
    check(from_packed % static_cast<std::ptrdiff_t>(kRecord / 2) == 0 && std::abs(from_packed) < 7 * static_cast<std::ptrdiff_t>(kRecord / 2),
          "mixed strides: the copied record sits in its class's packed slots");
    tier.end_round();
    tier.controller().check();
    {
        // The warm start's refill with record-size classes: the first pass fills layer 1's class with keys 20-24 and
        // passes over 25-29, then fills layer 0's with 0-4; the frames take 20 and 21, whose released slots go to the
        // best-ranked keys of their class the first pass passed over (25, 26), not to keys after where it stopped.
        HostExpertTier warm(store, o);
        std::vector<std::uint32_t> order;
        for (std::uint32_t k = kExpertsHere; k < kExpertsHere + 10; ++k) { order.push_back(k); }
        for (std::uint32_t k = 0; k < 10; ++k) { order.push_back(k); }
        check(warm.prefill(order) == 10 && warm.record(kExpertsHere + 5) == nullptr,
              "mixed strides: the first pass passes over a full class's keys");
        auto& host = warm.controller();
        for (const std::uint32_t k : {kExpertsHere, kExpertsHere + 1}) {
            host.queued(k);
            host.promotion_issued(k);
            host.promotion_completed(k, true);
        }
        check(host.release_shadows() == 2, "mixed strides: the seeds' RAM copies are released");
        const auto in_vram = [](std::uint32_t k) { return k == kExpertsHere || k == kExpertsHere + 1; };
        check(warm.prefill(order, in_vram) == 2 && holds(warm.record(kExpertsHere + 5), kExpertsHere + 5) &&
                  holds(warm.record(kExpertsHere + 6), kExpertsHere + 6) && warm.record(kExpertsHere + 7) == nullptr &&
                  warm.record(kExpertsHere) == nullptr,
              "mixed strides: the refill takes the passed-over keys first and skips the frames' keys");
        host.check();
    }
    {
        // A RAM resize with record-size classes, in chunks of two slots: the transit region stays, each class keeps
        // more slots than its demotion list (3 of 7) while the tier keeps min_slots (15 of 24), and growth refills both
        // classes. With no uses every chunk costs nothing, so the later chunks go first: class 1's slot 23 and class
        // 0's slot 16, then each class's next two.
        HostExpertTier::Options small = o;
        small.slots_per_chunk         = 2;
        std::vector<void*> freed;
        small.release = [&freed](void* p) {
            freed.push_back(p);
            ::operator delete(p, std::align_val_t{4096});
        };
        HostExpertTier resized(store, small);
        std::vector<std::uint32_t> all(2 * kExpertsHere);
        for (std::uint32_t k = 0; k < all.size(); ++k) { all[k] = k; }
        check(resized.prefill(all) == 10, "mixed resize: each class's 5 resident slots fill");
        std::vector<const std::uint8_t*> transit;
        for (const std::uint32_t s : resized.controller().ring()) { transit.push_back(resized.slot_bytes(s)); }
        for (const std::uint32_t s : resized.controller().prefetch_list()) { transit.push_back(resized.slot_bytes(s)); }
        const std::uint64_t expected = 3 * kRecord + 3 * (kRecord / 2);
        check(resized.shrinkable_bytes() == expected, "mixed resize: three slots of each class can go");
        check(shrink(resized, ~0ULL) == expected && freed.size() == 4 && resized.shrinkable_bytes() == 0,
              "mixed resize: four chunks given back");
        bool transit_kept = true;
        for (const void* p : freed) { transit_kept = transit_kept && std::find(transit.begin(), transit.end(), p) == transit.end(); }
        for (const std::uint32_t s : resized.controller().ring()) { transit_kept = transit_kept && resized.slot_bytes(s) != nullptr; }
        check(transit_kept, "mixed resize: the transit slots stay");
        bool kept = true;
        for (const std::uint32_t k : {0U, 1U, kExpertsHere, kExpertsHere + 1}) { kept = kept && holds(resized.record(k), k); }
        for (const std::uint32_t k : {2U, 3U, 4U, kExpertsHere + 2, kExpertsHere + 3, kExpertsHere + 4}) {
            kept = kept && resized.record(k) == nullptr;
        }
        check(kept, "mixed resize: the given-back slots' keys lost their records, the others keep theirs");
        resized.controller().check();
        check(grow(resized, ~0ULL) == expected && resized.growable_bytes() == 0, "mixed resize: growth takes it all back");
        bool refilled = true;
        for (std::uint32_t k = 0; k < 5; ++k) {
            refilled = refilled && holds(resized.record(k), k) && holds(resized.record(kExpertsHere + k), kExpertsHere + k);
        }
        check(refilled, "mixed resize: growth refills both classes from the artifact");
        resized.controller().check();
    }
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

// Writes an expert stripe of the test artifact: each record's tail past `head` bytes, layer by layer,
// after a 4 KiB header (format v1, expert_store.h). `stride` is written as each layer's stride.
void write_stripe(const std::filesystem::path& path, const infernix::artifact::ArtifactId& id, std::uint64_t head,
                  std::uint64_t stride, bool magic = true, std::uint32_t layers_written = kLayers) {
    std::vector<std::uint8_t> header(4096, 0);
    const auto put = [&](std::size_t at, std::uint64_t v, int bytes) {
        for (int i = 0; i < bytes; ++i) { header[at + static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(v >> (8 * i)); }
    };
    if (magic) { std::memcpy(header.data(), "INFXSTR1", 8); }
    put(8, 1, 4);
    put(12, kLayers, 4);
    put(16, kExperts, 4);
    put(20, 4096, 4);
    std::memcpy(&header[24], id.data(), id.size());
    const std::uint64_t tail = kRecord - head;
    for (std::uint32_t l = 0; l < kLayers; ++l) {
        put(40 + 24 * l, stride, 8);
        put(48 + 24 * l, head, 8);
        put(56 + 24 * l, 4096 + static_cast<std::uint64_t>(l) * kExperts * tail, 8);
    }
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(header.data()), static_cast<std::streamsize>(header.size()));
    std::vector<char> rec(tail);
    for (std::uint32_t key = 0; key < layers_written * kExperts; ++key) {
        for (std::uint64_t i = 0; i < tail; ++i) { rec[i] = static_cast<char>(oracle(key, head + i)); }
        f.write(rec.data(), static_cast<std::streamsize>(tail));
    }
    if (!f) { throw std::runtime_error("cannot write the stripe test file"); }
}

// An expert stripe (tools/flash_next/expert_stripe.py): each record's tail is read from the stripe
// file, its head from the artifact (the straddling record's head is its file-A part), and the record
// lands whole; a stripe of another artifact, of other strides, cut short or without its magic is refused.
void test_stripe(const ExpertStore& base) {
    using infernix::runtime::expert_offload::read_expert_stripe;
    using infernix::runtime::expert_offload::with_stripe;
    infernix::artifact::ArtifactId id{};
    for (std::size_t i = 0; i < id.size(); ++i) { id[i] = static_cast<std::byte>(i + 1); }
    const auto path = infernix::test::unique_temp_path("infernix_host_tier.stripe");
    write_stripe(path, id, 4096, kRecord);
    const ExpertStore store = with_stripe(base, read_expert_stripe(path), id);
    const auto straddler    = store.segments(kStraddle / kExperts, kStraddle % kExperts);
    const auto plain        = store.segments(1, 3);
    check(straddler.size() == 2 && straddler[0].file == 0 && straddler[0].bytes == 4096 && straddler[1].file == 2 &&
              plain.size() == 2 && plain[1].at == 4096 && plain[1].offset == 4096 + (kExperts + 3) * 4096ULL &&
              store.multipliers(2) != nullptr,
          "stripe: heads from the artifact (the straddler's from file A), tails at their stripe offsets");
    {
        HostExpertTier tier(store, options());
        std::vector<std::uint32_t> ranked(kKeys);
        for (std::uint32_t k = 0; k < kKeys; ++k) { ranked[k] = k; }
        check(tier.prefill(ranked) == 50, "stripe: the pre-fill fills the resident slots");
        bool all = true;
        for (std::uint32_t k = 0; k < 50; ++k) { all = all && matches(tier.record(k), k); }
        check(all && tier.stats().run_reads == 0, "stripe: pre-filled records equal the bytes, read segment by segment");
        tier.begin_round(32);
        std::vector<std::uint32_t> statuses;
        std::vector<bool> valid;
        (void)serve(tier, {55, kStraddle, 70, 90}, statuses, valid);
        check(valid[0] && valid[1] && valid[2] && valid[3], "stripe: demanded records (the straddler included) land whole");
        const auto bad = serve(tier, {kKeys - 1}, statuses, valid);
        check(bad[0] == nullptr && statuses[0] == EIO, "stripe: an unreadable head still fails its ticket");
        tier.begin_round(32);
    }
    const auto refused = [&](auto&& make) {
        try {
            (void)make();
        } catch (const infernix::artifact::ArtifactError&) { return true; }
        return false;
    };
    infernix::artifact::ArtifactId other = id;
    other[0] = std::byte{0x7F};
    check(refused([&] { return with_stripe(base, read_expert_stripe(path), other); }), "stripe: another artifact's is refused");
    write_stripe(path, id, 4096, 3 * 4096);
    check(refused([&] { return with_stripe(base, read_expert_stripe(path), id); }), "stripe: other strides are refused");
    write_stripe(path, id, 4096, kRecord, true, kLayers - 1);
    check(refused([&] { return read_expert_stripe(path); }), "stripe: a file without every tail is refused");
    write_stripe(path, id, 4096, kRecord, false);
    check(refused([&] { return read_expert_stripe(path); }), "stripe: a file without its magic is refused");
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

// Pinning refused after three of four chunks (the Windows page-locking limit): the tier gives back a
// chunk for later pins and keeps the slots it lost as unpinned records, beside those asked for.
HostExpertTier::Options refusing_options(int& chunks) {
    auto o            = options();
    o.min_slots       = 20;
    o.unpinned        = 14;
    o.bounce          = 4;
    o.pin_reserve     = 16 * kRecord;
    o.copy_threads    = 2;
    o.allocate        = [&chunks](std::size_t bytes) -> void* {
        if (chunks++ == 3) { return nullptr; }
        return ::operator new(bytes, std::align_val_t{4096});
    };
    return o;
}

void test_unpinned(const ExpertStore& store) {
    int chunks = 0;
    HostExpertTier tier(store, refusing_options(chunks));
    check(tier.pinned_slots() == 32 && tier.refused_slots() == 32, "a refused chunk: one chunk given back, 32 slots kept");
    check(tier.unpinned_records() == 46 && tier.bounce_slots() == 4, "the refused slots join the unpinned records asked for; 4 bounce slots");
    std::cout << "unpinned: lock " << (tier.unpinned_lock_failure().empty() ? "ok" : tier.unpinned_lock_failure()) << '\n';
    std::vector<std::uint32_t> ranked(kKeys);
    for (std::uint32_t k = 0; k < kKeys; ++k) { ranked[k] = k; }
    check(tier.prefill(ranked) == 14, "the 14 resident slots take the first keys");
    // The unpinned records take the next keys the slots do not hold, skipping key 20 (in VRAM),
    // including the record that straddles the files (key 60).
    const std::uint32_t filled = tier.fill_unpinned(ranked, [](std::uint32_t key) { return key == 20; });
    check(filled == 46, "the unpinned records fill up");
    bool exact = true, device_free = true;
    for (std::uint32_t k = 14; k <= kStraddle; ++k) { // 14..19 and 21..60: 46 records
        if (k == 20) { continue; }
        exact       = exact && matches(tier.unpinned_record(k), k) && matches(tier.host_record(k), k);
        device_free = device_free && tier.record(k) == nullptr && tier.controller().held(k);
    }
    check(exact, "unpinned records (straddling one included) equal the file bytes");
    check(device_free, "unpinned records are not in the device table, and their keys are held");
    check(tier.unpinned_record(20) == nullptr && tier.unpinned_record(kStraddle + 1) == nullptr,
          "skipped and later keys are not unpinned");
    check(tier.fill_unpinned(ranked, {}) == 0, "a full set of unpinned records takes nothing more");
    tier.controller().check();
    // Promotions stage unpinned records through the bounce slots (copy threads plus the caller).
    const std::vector<std::pair<std::uint8_t*, const std::uint8_t*>> bounce{
        {tier.bounce_slot(0), tier.unpinned_record(30)}, {tier.bounce_slot(1), tier.unpinned_record(kStraddle)},
        {tier.bounce_slot(2), tier.unpinned_record(15)}, {tier.bounce_slot(3), tier.unpinned_record(44)}};
    tier.copy_records(bounce);
    check(matches(tier.bounce_slot(0), 30) && matches(tier.bounce_slot(1), kStraddle) && matches(tier.bounce_slot(2), 15) &&
              matches(tier.bounce_slot(3), 44),
          "copy_records lands every record in its bounce slot");

    // CPU demands: unpinned records in place, without a read; a slot key from its slot; key 20 from disk.
    tier.begin_round(32);
    std::vector<std::uint32_t> statuses;
    std::vector<bool> valid;
    const std::vector<std::uint32_t> keys{18, kStraddle, 20, 5};
    const auto got = serve(tier, keys, statuses, valid);
    check(valid[0] && valid[1] && valid[2] && valid[3], "CPU demands of unpinned, disk and slot keys hold their bytes");
    check(got[0] == tier.unpinned_record(18) && got[1] == tier.unpinned_record(kStraddle),
          "an unpinned record is served in place");
    const std::vector<std::uint32_t> hot{18, kStraddle, 20};
    tier.record_uses(1, hot, 50.0);
    tier.begin_round(32);
    check(tier.stats().unpinned_served == 2 && tier.stats().demand_reads == 1, "two served in place, one read");
    check(tier.record(18) == nullptr && tier.record(kStraddle) == nullptr && matches(tier.record(20), 20),
          "only the read landing is admitted into a slot");
    tier.controller().check();
    tier.end_round();
}

// The device's fetch of unpinned records: copied into ring slots (the device reads pinned memory
// only) with their bytes, more records than ring slots, and never admitted.
void test_fetch_unpinned(const ExpertStore& store) {
    infernix::ops::offloaded_moe::FetchChannel channel;
    int chunks = 0;
    auto o     = refusing_options(chunks);
    o.fetch    = &channel;
    HostExpertTier tier(store, o);
    std::vector<std::uint32_t> ranked(kKeys);
    for (std::uint32_t k = 0; k < kKeys; ++k) { ranked[k] = k; }
    (void)tier.prefill(ranked);
    (void)tier.fill_unpinned(ranked, {});
    DeviceSide device{channel.channel(0)};
    tier.begin_round(32);
    // Layer 0, experts 10..29: 10..13 in slots, 14..29 unpinned (16 copies through the 8-slot ring).
    std::vector<std::int32_t> experts;
    for (std::int32_t e = 10; e < 30; ++e) { experts.push_back(e); }
    device.publish(0, experts);
    bool all = true, copied = true;
    for (std::uint32_t i = 0; i < experts.size(); ++i) {
        std::uint32_t status = 0;
        const std::uint8_t* record = device.wait(i, status);
        const std::uint32_t key    = 10 + i;
        all = all && status == 0 && matches(record, key);
        if (key >= 14) { copied = copied && record != nullptr && record != tier.unpinned_record(key); }
        device.consume(i + 1);
    }
    check(all, "fetch: slot keys and unpinned copies land with their bytes");
    check(copied, "fetch: an unpinned record lands as a copy in a ring slot");
    tier.end_round();
    const std::vector<std::uint32_t> hot{18, 19, 20};
    tier.record_uses(1, hot, 50.0);
    tier.begin_round(32);
    check(tier.stats().unpinned_copies == 16 && tier.stats().demand_reads == 0, "fetch: 16 copies, no reads");
    check(tier.record(18) == nullptr && tier.controller().held(18) && tier.stats().admitted == 0,
          "fetch: a copied unpinned record is not admitted");
    tier.controller().check();
    tier.end_round();
}

// A RAM resize (shrink, grow) of the pinned slots: four chunks of 16 slots; shrink gives back the chunk holding the
// least valuable residents through the release callback (here the lists' chunk: their replacements evict the lowest
// keys), passes over a chunk with a pinned slot or a refill's target, and stops at min_slots (15); growth allocates the
// chunks again and refills them with the best-ranked keys, and stops where the OS refuses.
void test_resize(const ExpertStore& store) {
    struct Calls {
        int allocated = 0, released = 0, refuse_after = -1;
        std::vector<void*> freed;
    } calls;
    {
        auto o     = options();
        o.allocate = [&calls](std::size_t bytes) -> void* {
            if (calls.refuse_after >= 0 && calls.allocated >= calls.refuse_after) { return nullptr; }
            ++calls.allocated;
            return ::operator new(bytes, std::align_val_t{4096});
        };
        o.release = [&calls](void* p) {
            ++calls.released;
            calls.freed.push_back(p);
            ::operator delete(p, std::align_val_t{4096});
        };
        HostExpertTier tier(store, o);
        auto& host = tier.controller();
        std::vector<std::uint32_t> ranked(kKeys);
        for (std::uint32_t k = 0; k < kKeys; ++k) { ranked[k] = k; }
        check(calls.allocated == 4 && tier.prefill(ranked) == 50, "resize: four chunks; keys 0-49 in slots 14-63");
        for (std::uint32_t k = 0; k < 50; ++k) {
            const std::vector<std::uint32_t> one{k};
            tier.record_uses(0, one, static_cast<double>(k + 1));
        }
        std::vector<std::uint32_t> dirty;
        tier.take_dirty(dirty);
        check(tier.growable_bytes() == 0 && tier.shrinkable_bytes() == 3 * 16 * kRecord,
              "resize: three chunks can go above min_slots");
        tier.begin_round(32);
        bool threw = false;
        try {
            (void)shrink(tier, 1);
        } catch (const std::logic_error&) { threw = true; }
        check(threw, "resize: no shrink while a round is open");
        tier.end_round();

        // Chunk 0 (slots 0-15: the lists and keys 0 and 1) is the cheapest; its lists' 14 replacements evict keys 2-15.
        const std::uint8_t* chunk0 = tier.slot_bytes(0);
        check(shrink(tier, 1) == 16 * kRecord && calls.released == 1 && calls.freed[0] == chunk0,
              "resize: shrink gives the least valuable chunk back through the release callback");
        bool retired = true, evicted = true, kept = true;
        for (std::uint32_t s = 0; s < 16; ++s) {
            retired = retired && tier.slot_bytes(s) == nullptr && host.list(s) == expert_cache::SlotList::kRetired;
        }
        for (std::uint32_t k = 0; k < 16; ++k) { evicted = evicted && tier.record(k) == nullptr; }
        for (std::uint32_t k = 16; k < 50; ++k) { kept = kept && matches(tier.record(k), k); }
        check(retired && evicted && kept, "resize: the chunk's keys and its lists' replacements lost their records, the rest keep theirs");
        dirty.clear();
        tier.take_dirty(dirty);
        std::sort(dirty.begin(), dirty.end());
        bool exact = dirty.size() == 16;
        for (std::uint32_t k = 0; k < 16 && exact; ++k) { exact = dirty[k] == k; }
        check(exact, "resize: the evicted keys are dirty");
        check(tier.pinned_slots() == 48 && tier.pinned_bytes() == 48 * kRecord && tier.growable_bytes() == 16 * kRecord,
              "resize: the pinned counts follow");
        host.check();
        // The ring's new slots serve the next round.
        tier.begin_round(32);
        std::vector<std::uint32_t> statuses;
        std::vector<bool> valid;
        (void)serve(tier, {3, 60, 70, 20}, statuses, valid);
        check(valid[0] && valid[1] && valid[2] && valid[3], "resize: demands land in the slots left");
        tier.end_round();

        // Key 16 (chunk 1, now the cheapest) is queued for VRAM: chunk 2 goes instead; then chunk 1, and chunk 3 holds
        // the floor.
        host.queued(16);
        const std::uint8_t* chunk2 = tier.slot_bytes(32);
        check(shrink(tier, 1) == 16 * kRecord && calls.freed.back() == chunk2 && tier.slot_bytes(16) != nullptr,
              "resize: a chunk with a pinned slot is passed over");
        host.unqueued(16);
        check(tier.shrinkable_bytes() == 16 * kRecord && shrink(tier, ~0ULL) == 16 * kRecord && tier.pinned_slots() == 16 &&
                  shrink(tier, ~0ULL) == 0 && tier.shrinkable_bytes() == 0,
              "resize: shrink stops at min_slots");
        host.check();

        // Growth: the three chunks come back, and the best-ranked keys without a host copy (0-47) fill them.
        dirty.clear();
        tier.take_dirty(dirty);
        check(grow(tier, 1) == 0, "resize: growth takes no chunk larger than asked");
        check(grow(tier, ~0ULL) == 48 * kRecord && calls.allocated == 7 && tier.growable_bytes() == 0 && tier.pinned_slots() == 64,
              "resize: growth allocates every chunk again");
        bool refilled = true;
        for (std::uint32_t k = 0; k < 50; ++k) { refilled = refilled && matches(tier.record(k), k); }
        dirty.clear();
        tier.take_dirty(dirty);
        check(refilled && tier.record(50) == nullptr && dirty.size() == 48, "resize: growth refills the best-ranked keys, dirty");
        host.check();

        // A refill's target (a demotion slot) keeps its chunk until the demotion completes.
        {
            const std::vector<std::uint32_t> hot{60};
            tier.record_uses(1, hot, 1000.0);
        }
        std::uint32_t target = 0;
        check(host.vram_evicted(60, true, target) == expert_cache::VramEviction::kDemote, "resize: a demotion");
        const std::uint32_t id = tier.refill(60, target);
        const std::uint8_t* target_chunk = tier.slot_bytes(target / 16 * 16);
        check(id != HostExpertTier::kNoRefill && shrink(tier, 1) == 16 * kRecord && calls.freed.back() != target_chunk &&
                  tier.slot_bytes(target) != nullptr,
              "resize: a refill's target keeps its chunk");
        while (tier.refill_state(id) == 0) { std::this_thread::yield(); }
        check(tier.refill_state(id) == 1, "resize: the refill lands");
        tier.refill_release(id);
        host.demotion_completed(60, target, false);
        check(matches(tier.record(60), 60), "resize: the refilled record is in RAM");
        host.check();
        // The OS refuses to pin more.
        calls.refuse_after = calls.allocated;
        check(grow(tier, ~0ULL) == 0 && tier.growable_bytes() == 16 * kRecord, "resize: growth stops when the OS refuses");
        host.check();
    }
    check(calls.released == calls.allocated, "resize: every chunk allocated is released once");
}

// A RAM resize of the unpinned records: the pinned chunks cannot go (chunk 0 holds min_slots, chunk 1 the bounce
// slots), so shrink gives back the lowest-ranked held keys' records (their pages decommitted), and growth commits them
// again and refills them with the best-ranked keys without a host copy.
void test_resize_unpinned(const ExpertStore& store) {
    int chunks = 0;
    HostExpertTier tier(store, refusing_options(chunks));
    std::vector<std::uint32_t> ranked(kKeys);
    for (std::uint32_t k = 0; k < kKeys; ++k) { ranked[k] = k; }
    check(tier.prefill(ranked) == 14 && tier.fill_unpinned(ranked, {}) == 46, "unpinned resize: keys 14-59 are held");
    check(tier.shrinkable_bytes() == 46 * kRecord, "unpinned resize: only the unpinned records can go");
    // Key 59 (the last filled) is the best ranked; the lowest are then 58, 57, 56, 55 and 54.
    {
        const std::vector<std::uint32_t> hot{59};
        tier.record_uses(0, hot, 100.0);
    }
    std::vector<std::uint32_t> dirty;
    tier.take_dirty(dirty);
    dirty.clear();
    const std::uint8_t* gone = tier.unpinned_record(54);
#ifdef _WIN32
    SIZE_T minimum_before = 0, maximum = 0;
    DWORD flags = 0;
    (void)::GetProcessWorkingSetSizeEx(::GetCurrentProcess(), &minimum_before, &maximum, &flags);
#endif
    check(shrink(tier, 5 * kRecord) == 5 * kRecord, "unpinned resize: five records given back");
    bool released = true;
    for (std::uint32_t k = 54; k < 59; ++k) {
        released = released && tier.unpinned_record(k) == nullptr && tier.host_record(k) == nullptr && !tier.controller().held(k);
    }
    check(released && matches(tier.unpinned_record(59), 59) && matches(tier.unpinned_record(53), 53),
          "unpinned resize: the lowest-ranked held keys' records go");
    tier.take_dirty(dirty);
    std::sort(dirty.begin(), dirty.end());
    check(dirty == std::vector<std::uint32_t>{54, 55, 56, 57, 58}, "unpinned resize: their keys are dirty");
    check(tier.growable_bytes() == 5 * kRecord && tier.shrinkable_bytes() == 41 * kRecord, "unpinned resize: the room follows");
#ifdef _WIN32
    MEMORY_BASIC_INFORMATION info{};
    check(::VirtualQuery(gone, &info, sizeof(info)) != 0 && info.State == MEM_RESERVE, "unpinned resize: the pages are decommitted");
    if (tier.unpinned_lock_failure().empty()) {
        SIZE_T minimum_after = 0;
        (void)::GetProcessWorkingSetSizeEx(::GetCurrentProcess(), &minimum_after, &maximum, &flags);
        check(minimum_before - minimum_after == 5 * kRecord, "unpinned resize: the minimum working set is lowered");
    }
#endif
    tier.controller().check();
    // An unheld key is read from the artifact.
    tier.begin_round(32);
    std::vector<std::uint32_t> statuses;
    std::vector<bool> valid;
    (void)serve(tier, {56}, statuses, valid);
    check(valid[0] && tier.unpinned_record(56) == nullptr, "unpinned resize: an unheld key is read from the artifact");
    tier.end_round();
    // Growth: the best-ranked keys without a host copy (100-104) fill the committed records.
    {
        const std::vector<std::uint32_t> wanted{100, 101, 102, 103, 104};
        tier.record_uses(0, wanted, 50.0);
    }
    check(grow(tier, 5 * kRecord) == 5 * kRecord && tier.growable_bytes() == 0, "unpinned resize: the records are committed again");
    bool refilled = true;
    for (std::uint32_t k = 100; k < 105; ++k) {
        refilled = refilled && matches(tier.unpinned_record(k), k) && tier.controller().held(k) && tier.record(k) == nullptr;
    }
    check(refilled, "unpinned resize: growth refills them with the best-ranked keys");
#ifdef _WIN32
    check(::VirtualQuery(gone, &info, sizeof(info)) != 0 && info.State == MEM_COMMIT, "unpinned resize: the pages are committed again");
#endif
    tier.controller().check();
}

// Record fills while rounds run: a growth linked between rounds (not filled from the artifact, as under load) leaves
// free unpinned records; a round's demand landings that no slot admits (every resident slot holds a better-ranked
// key) are copied into them on the copy threads, the best-ranked first and at most a sixteenth of the ring per
// boundary (at least one), and the boundary after holds each key in its record, with the file's bytes.
void test_record_fills(const ExpertStore& store) {
    int chunks = 0;
    HostExpertTier tier(store, refusing_options(chunks));
    std::vector<std::uint32_t> ranked(kKeys);
    for (std::uint32_t k = 0; k < kKeys; ++k) { ranked[k] = k; }
    check(tier.prefill(ranked) == 14 && tier.fill_unpinned(ranked, {}) == 46, "record fills: slots hold 0-13, records 14-59");
    {
        std::vector<std::uint32_t> residents;
        for (std::uint32_t k = 0; k < 14; ++k) { residents.push_back(k); }
        tier.record_uses(0, residents, 100.0); // no landing below outranks them
    }
    check(shrink(tier, 3 * kRecord) == 3 * kRecord, "record fills: three records given back");
    check(tier.begin_growth(3 * kRecord, ram_worker()) == 3 * kRecord, "record fills: a growth of the three records");
    while (!tier.growth_landed()) { std::this_thread::yield(); }
    check(tier.link_growth() == 3 * kRecord && tier.room_unfilled(), "record fills: linked, not filled");
    const std::vector<std::uint32_t> wanted{100, 101, 102, 103, 104};
    const auto round = [&](const std::vector<std::uint32_t>& keys) {
        tier.begin_round(32);
        std::vector<std::uint32_t> statuses;
        std::vector<bool> valid;
        (void)serve(tier, keys, statuses, valid);
        bool all = true;
        for (std::size_t i = 0; i < keys.size(); ++i) { all = all && valid[i] && statuses[i] == 0; }
        check(all, "record fills: the round's demands land");
        tier.end_round();
    };
    round(wanted);
    {
        const std::vector<std::uint32_t> warm{103};
        tier.record_uses(0, wanted, 1.0);
        tier.record_uses(0, warm, 5.0); // 103 ranks first among them, below every resident
    }
    // The boundary admits nothing and starts one fill (a ring of 8: a sixteenth rounds up to one), key 103's; the next
    // boundary holds it (and starts the next fill, of a key the round before read).
    round({100, 101});
    check(tier.unpinned_record(103) == nullptr, "record fills: a fill is held only at a later boundary");
    // A boundary never waits for a fill's copy: it is held at the first boundary after the copy finished, at the
    // latest at the eighth (which waits).
    const auto boundaries_until = [&](const auto& done) {
        for (int i = 0; i < 8 && !done(); ++i) {
            tier.begin_round(32);
            tier.end_round();
        }
    };
    boundaries_until([&] { return tier.unpinned_record(103) != nullptr; });
    check(tier.unpinned_record(103) != nullptr && matches(tier.unpinned_record(103), 103) &&
              tier.controller().held(103) && tier.record(103) == nullptr && tier.stats().record_fills == 1,
          "record fills: the best-ranked landing is held in a free record at the next boundary");
    check(tier.unpinned_record(104) == nullptr && tier.unpinned_record(102) == nullptr,
          "record fills: one per boundary with this ring");
    // Later rounds fill the rest of the free records from what they read, and then nothing more.
    for (int i = 0; i < 8 && tier.stats().record_fills < 3; ++i) {
        round(wanted);
        tier.begin_round(32); // an empty round: its boundary settles a finished fill or starts one from what was read
        tier.end_round();
    }
    std::uint32_t held = 0;
    bool exact         = true;
    for (const std::uint32_t k : wanted) {
        if (tier.unpinned_record(k) == nullptr) { continue; }
        ++held;
        exact = exact && matches(tier.unpinned_record(k), k) && tier.controller().held(k);
    }
    check(held == 3 && exact && tier.stats().record_fills == 3, "record fills: the three free records hold demanded keys");
    tier.controller().check();
}

// An EngineHost's reused slots (HostSlotPool): tiers one after another take the chunks the last gave back and keep the
// records their slots still hold, reading only the others; every record equals the file's bytes, and the fill places
// the keys a fill without the pool places. Another artifact's chunks are reused as memory only, a chunk of another
// size is not taken, and what a reused chunk held is known only until the first round.
void test_slot_pool(const ExpertStore& store) {
    struct Calls {
        int allocated = 0, released = 0;
    } calls;
    const auto allocate = [&calls](std::size_t bytes) -> void* {
        ++calls.allocated;
        return ::operator new(bytes, std::align_val_t{4096});
    };
    const auto release = [&calls](void* p) {
        ++calls.released;
        ::operator delete(p, std::align_val_t{4096});
    };
    auto pool = std::make_shared<infernix::runtime::expert_offload::HostSlotPool>(release);
    const auto pooled = [&](std::uint32_t ring, std::string identity) {
        auto o          = options();
        o.ring          = ring;
        o.allocate      = allocate;
        o.release       = release;
        o.pool          = pool;
        o.pool_identity = std::move(identity);
        return o;
    };
    std::vector<std::uint32_t> ranked(kKeys);
    for (std::uint32_t k = 0; k < kKeys; ++k) { ranked[k] = k; }
    const auto all_match = [&](const HostExpertTier& tier, std::uint32_t first, std::uint32_t last) {
        bool ok = true;
        for (std::uint32_t k = first; k < last; ++k) { ok = ok && matches(tier.record(k), k); }
        return ok;
    };
    {
        // The first tier reads every record and gives four chunks back; a round admits 55, 70 and 90 over three of
        // keys 0-49, so those three leave RAM.
        HostExpertTier tier(store, pooled(8, "A"));
        check(calls.allocated == 4 && tier.prefill(ranked) == 50 && tier.stats().kept == 0 && tier.stats().prefill_reads == 50,
              "pool: the first tier reads its 50 records");
        tier.begin_round(32);
        std::vector<std::uint32_t> statuses;
        std::vector<bool> valid;
        (void)serve(tier, {55, 70, 90}, statuses, valid);
        const std::vector<std::uint32_t> hot{55, 70, 90};
        tier.record_uses(1, hot, 50.0);
        tier.begin_round(32);
        check(tier.stats().admitted == 3, "pool: three landings admitted");
        tier.end_round();
    }
    check(pool->idle_chunks() == 4 && pool->idle_bytes() == 64 * kRecord && calls.released == 0,
          "pool: the tier's chunks are kept, not released");
    {
        // The same layout: the 47 keys of 0-49 still in RAM are kept, the three it lost are read.
        HostExpertTier tier(store, pooled(8, "A"));
        check(calls.allocated == 4 && pool->idle_chunks() == 0, "pool: the second tier takes the kept chunks");
        check(tier.prefill(ranked) == 50 && tier.stats().kept == 47 && tier.stats().prefill_reads == 3,
              "pool: 47 records kept, 3 read");
        check(all_match(tier, 0, 50) && tier.record(55) == nullptr && tier.record(70) == nullptr,
              "pool: the fill holds keys 0-49, each equal to the file's bytes");
        tier.controller().check();
    }
    {
        // A ring of 4 moves the controller's slots: 54 resident slots, keys 0-49 kept where they are, 50-53 read.
        HostExpertTier tier(store, pooled(4, "A"));
        check(tier.prefill(ranked) == 54 && tier.stats().kept == 50 && tier.stats().prefill_reads == 4 && all_match(tier, 0, 54),
              "pool: another layout keeps the records in its resident slots");
        tier.controller().check();
    }
    {
        // The startup fill's passes (2026-11-08: a reused Dense8 host gave other tokens): the first pass reads keys
        // 60-113 over every slot that held one of keys 0-53, the frames take 60 and 61 (their RAM copies become shadows
        // and are released), and the refill places the keys whose records those two slots held before the first pass.
        // Those slots now hold 60's and 61's bytes: the refill reads the keys, it keeps nothing.
        std::vector<const std::uint8_t*> was(kKeys, nullptr);
        {
            HostExpertTier tier(store, pooled(4, "A"));
            check(tier.prefill(ranked) == 54 && tier.stats().kept == 54, "pool passes: the last tier's 54 records kept");
            for (std::uint32_t k = 0; k < 54; ++k) { was[k] = tier.record(k); }
        }
        HostExpertTier tier(store, pooled(4, "A"));
        std::vector<std::uint32_t> others;
        for (std::uint32_t k = 60; k < 114; ++k) { others.push_back(k); }
        check(tier.prefill(others) == 54 && tier.stats().kept == 0 && all_match(tier, 60, 114),
              "pool passes: the first pass reads 54 other keys into every slot");
        std::vector<std::uint32_t> overwritten;
        for (const std::uint32_t k : {60U, 61U}) {
            for (std::uint32_t old = 0; old < 54; ++old) {
                if (was[old] == tier.record(k)) { overwritten.push_back(old); }
            }
        }
        auto& host = tier.controller();
        for (const std::uint32_t k : {60U, 61U}) {
            host.queued(k);
            host.promotion_issued(k);
            host.promotion_completed(k, true);
        }
        check(overwritten.size() == 2 && host.release_shadows() == 2, "pool passes: the two seeds' slots are released");
        check(tier.prefill(overwritten) == 2 && tier.stats().kept == 0 && tier.stats().prefill_reads == 56 &&
                  matches(tier.record(overwritten[0]), overwritten[0]) && matches(tier.record(overwritten[1]), overwritten[1]),
              "pool passes: the refill reads the keys whose slots the first pass overwrote");
        host.check();
    }
    {
        // Another artifact's records: the chunks are reused as memory, nothing is kept.
        HostExpertTier tier(store, pooled(8, "B"));
        check(calls.allocated == 4 && tier.prefill(ranked) == 50 && tier.stats().kept == 0 && tier.stats().prefill_reads == 50 &&
                  all_match(tier, 0, 50),
              "pool: another identity reads every record into the reused chunks");
    }
    {
        // Identity A again, after B wrote the chunks: nothing of A's is trusted.
        HostExpertTier tier(store, pooled(8, "A"));
        check(tier.prefill(ranked) == 50 && tier.stats().kept == 0 && all_match(tier, 0, 50),
              "pool: a chunk another identity wrote holds none of A's records");
    }
    {
        // The chunks now hold A's records (the last tier's), but a round begins before the fill: what they held is no
        // longer known.
        HostExpertTier tier(store, pooled(8, "A"));
        tier.begin_round(32);
        tier.end_round();
        check(tier.prefill(ranked) == 50 && tier.stats().kept == 0 && tier.stats().prefill_reads == 50 && all_match(tier, 0, 50),
              "pool: after the first round the fill reads everything");
    }
    {
        // Chunks of another size are not taken: the tier allocates its own, and the kept ones stay idle until released.
        auto o            = pooled(8, "A");
        o.slots_per_chunk = 32;
        HostExpertTier tier(store, o);
        check(calls.allocated == 6 && pool->idle_chunks() == 4, "pool: a tier of other chunk sizes allocates its own");
        pool->release_idle();
        check(calls.released == 4 && pool->idle_bytes() == 0, "pool: release_idle frees the idle chunks");
        check(tier.prefill(ranked) == 50 && tier.stats().kept == 0 && all_match(tier, 0, 50), "pool: its fill reads every record");
    }
    pool.reset();
    check(calls.released == calls.allocated, "pool: every chunk allocated is released once");

    // The CPU team's settled rates: a configuration's latest stands, another configuration has none, and a reading
    // of no rate is not kept.
    infernix::runtime::expert_offload::HostSlotPool rates([](void*) {});
    check(!rates.cpu_rates("a"), "pool: no CPU rates before a start measures them");
    rates.note_cpu_rates("a", {.experts = 18'000.0, .columns = 58'000.0, .note = "measured"});
    rates.note_cpu_rates("a", {.experts = 19'000.0, .columns = 59'000.0, .note = "again"});
    rates.note_cpu_rates("b", {.experts = 0.0, .columns = 59'000.0, .note = "none"});
    const auto a = rates.cpu_rates("a");
    check(a && a->experts == 19'000.0 && a->columns == 59'000.0 && a->note == "again",
          "pool: a configuration's latest CPU rates stand");
    check(!rates.cpu_rates("b") && !rates.cpu_rates("c"), "pool: another configuration, or a zero rate, has none");
}

// ---- CSF slots (HostExpertTier::Options::csf): Flash-Next records whose scales cluster per row as converted experts'
// do, and one record with random scales (it does not compress enough to hold its table: it stays expanded).
namespace csf = infernix::ops::offloaded_moe::csf;
constexpr std::uint32_t kCsfLayers = 2, kCsfExperts = 12, kCsfKeys = kCsfLayers * kCsfExperts;
constexpr std::uint32_t kRandomScales = 3; // the incompressible key

std::vector<std::uint8_t> csf_record(std::uint32_t key) {
    const auto& shape = infernix::ops::offloaded_moe::kFlashNextShape;
    std::vector<std::uint8_t> record(shape.record_bytes());
    std::uint64_t state = 0x9E3779B97F4A7C15ULL * (key + 1);
    const auto next = [&] {
        state ^= state << 13U;
        state ^= state >> 7U;
        state ^= state << 17U;
        return state;
    };
    for (auto& b : record) { b = static_cast<std::uint8_t>(next()); }
    const auto scales = [&](std::uint8_t* units, int rgs, int blocks) {
        for (int rg = 0; rg < rgs; ++rg) {
            int level[16];
            for (int& l : level) { l = 0x28 + static_cast<int>(next() % 32); }
            for (int b = 0; b < blocks; ++b) {
                std::uint8_t* s = units + (static_cast<std::size_t>(rg) * blocks + b) * 144 + 128;
                for (int r = 0; r < 16; ++r) {
                    const std::uint64_t v = next();
                    int code = level[r] + static_cast<int>(v % 9) - 4;
                    if ((v >> 8) % 50 == 0) { code += static_cast<int>((v >> 16) % 41) - 20; } // ~2 % far
                    s[r] = key == kRandomScales ? static_cast<std::uint8_t>(next() & 0x77U)
                                                : static_cast<std::uint8_t>(std::clamp(code, 1, 0x76));
                }
            }
        }
    };
    scales(record.data(), shape.gate_up_row_groups(), shape.gate_up_blocks());
    scales(record.data() + shape.gate_up_bytes(), shape.down_row_groups(), shape.down_blocks());
    return record;
}

// Whether a published address (tagged: a compressed slot; untagged: an expanded record) holds key's record.
bool csf_matches(const std::uint8_t* p, const std::vector<std::uint8_t>& want, std::uint64_t pitch) {
    if (p == nullptr) { return false; }
    const auto& shape = infernix::ops::offloaded_moe::kFlashNextShape;
    if (!csf::tagged(p)) { return std::memcmp(p, want.data(), want.size()) == 0; }
    const int words = csf::Geometry::of(shape).table_words();
    const std::uint32_t* table = csf::slot_table(p, words, pitch);
    if (!csf::valid_table(shape, table, csf::slot_table_offset(words, pitch))) { return false; }
    std::vector<std::uint8_t> out(want.size());
    csf::expand_record(shape, csf::untag(p), table, out.data());
    return out == want;
}

void test_csf(bool fetch) {
    const auto& shape = infernix::ops::offloaded_moe::kFlashNextShape;
    const std::uint64_t bytes = shape.record_bytes(); // 675 x 4 KiB: the slot pitch
    const auto path = infernix::test::unique_temp_path("infernix_host_tier_csf.bin");
    std::vector<std::vector<std::uint8_t>> want(kCsfKeys);
    std::vector<std::vector<ExpertStore::Segment>> segments(kCsfKeys);
    {
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        for (std::uint32_t k = 0; k < kCsfKeys; ++k) {
            want[k] = csf_record(k);
            file.write(reinterpret_cast<const char*>(want[k].data()), static_cast<std::streamsize>(bytes));
            segments[k] = {{0, k * bytes, 0, static_cast<std::uint32_t>(bytes)}};
        }
        if (!file) { throw std::runtime_error("cannot write the CSF test file"); }
    }
    std::vector<std::vector<float>> multipliers(kCsfLayers, std::vector<float>(3 * kCsfExperts, 1.0F));
    const ExpertStore store({path}, std::vector<std::uint64_t>(kCsfLayers, bytes), kCsfExperts, segments, multipliers);
    const auto csf_options = [&](bool compressed) {
        HostExpertTier::Options o;
        o.slots           = 16; // ring 4 + prefetch 2 + demotion 2 + 8 resident slots
        o.ring            = 4;
        o.prefetch        = 2;
        o.demotion        = 2;
        o.half_life       = 8;
        o.slots_per_chunk = 4;
        o.copy_threads    = 2;
        o.allocate = [](std::size_t n) { return ::operator new(n, std::align_val_t{4096}); };
        o.release  = [](void* p) { ::operator delete(p, std::align_val_t{4096}); };
        if (compressed) { o.csf = shape; }
        return o;
    };
    std::vector<std::uint32_t> ranked(kCsfKeys);
    for (std::uint32_t k = 0; k < kCsfKeys; ++k) { ranked[k] = k; }
    const auto key_matches = [&](const HostExpertTier& tier, std::uint32_t k) {
        return csf_matches(tier.record_tagged(k), want[k], tier.csf_pitch());
    };
    {
        HostExpertTier tier(store, csf_options(true));
        check(tier.prefill(ranked) == 8, "csf: the fill places 8 records");
        bool all = true, tags = true;
        for (std::uint32_t k = 0; k < 8; ++k) {
            all  = all && key_matches(tier, k);
            tags = tags && csf::tagged(tier.record_tagged(k)) == (k != kRandomScales) &&
                   (tier.record_csf(k) != nullptr) == (k != kRandomScales) && tier.record(k) == csf::untag(tier.record_tagged(k));
        }
        check(all, "csf: every filled slot expands to its file bytes");
        check(tags && tier.csf_slots() == 7 && tier.stats().csf_compressed == 7 && tier.stats().csf_expanded == 1,
              "csf: 7 slots compressed, the random-scale record left expanded");
        check(tier.csf_summary().find("CSF records in 7 RAM slots") != std::string::npos, "csf: the summary names them");
        // CPU demands: a compressed resident (tagged), the expanded one, and an SSD-only key (read expanded).
        tier.begin_round(32);
        std::vector<const std::uint8_t*> got(3);
        std::vector<std::uint32_t> status(3);
        const std::vector<std::uint32_t> keys{1, kRandomScales, 9};
        std::thread service([&] {
            std::vector<std::uint32_t> tickets;
            for (const std::uint32_t k : keys) { tickets.push_back(tier.demand(static_cast<int>(k / kCsfExperts), static_cast<int>(k % kCsfExperts))); }
            for (std::size_t i = 0; i < keys.size(); ++i) {
                got[i] = tier.wait(tickets[i], status[i]);
                tier.done(tickets[i]);
            }
        });
        service.join();
        check(status == std::vector<std::uint32_t>(3, 0) && csf::tagged(got[0]) && !csf::tagged(got[1]) && !csf::tagged(got[2]) &&
                  csf_matches(got[0], want[1], tier.csf_pitch()) && csf_matches(got[1], want[kRandomScales], tier.csf_pitch()) &&
                  csf_matches(got[2], want[9], tier.csf_pitch()),
              "csf: a CPU demand gets a compressed resident tagged, a read record expanded");
        // The landing is admitted expanded.
        const std::vector<std::uint32_t> hot{9};
        tier.record_uses(1, hot, 50.0);
        tier.begin_round(32);
        check(tier.stats().admitted == 1 && !csf::tagged(tier.record_tagged(9)) && tier.record_csf(9) == nullptr && key_matches(tier, 9),
              "csf: an admitted landing is an expanded record");
        tier.end_round();
        // A refill compresses its record before it lands; a D2H demotion (mark_expanded) leaves its record expanded.
        auto& host = tier.controller();
        for (const std::uint32_t k : {12U, 13U}) {
            const std::vector<std::uint32_t> key{k};
            tier.record_uses(1, key, 1000.0);
        }
        std::uint32_t refilled = 0, copied = 0;
        check(host.vram_evicted(12, true, refilled) == expert_cache::VramEviction::kDemote, "csf: a demotion slot for 12");
        const std::uint32_t id = tier.refill(12, refilled);
        while (tier.refill_state(id) == 0) { std::this_thread::yield(); }
        check(tier.refill_state(id) == 1, "csf: the refill lands");
        tier.refill_release(id);
        host.demotion_completed(12, refilled, false);
        check(csf::tagged(tier.record_tagged(12)) && key_matches(tier, 12), "csf: the refilled record is compressed in its slot");
        check(host.vram_evicted(13, true, copied) == expert_cache::VramEviction::kDemote, "csf: a demotion slot for 13");
        tier.mark_expanded(copied);
        std::memcpy(const_cast<std::uint8_t*>(tier.slot_bytes(copied)), want[13].data(), bytes); // the frame's copy
        host.demotion_completed(13, copied, false);
        check(!csf::tagged(tier.record_tagged(13)) && key_matches(tier, 13), "csf: a copied demotion is an expanded record");
        host.check();
    }
    // An EngineHost's slots keep their records' formats: a CSF tier keeps the compressed ones as compressed, a tier
    // without CSF keeps only the expanded ones.
    auto pool = std::make_shared<infernix::runtime::expert_offload::HostSlotPool>([](void* p) {
        ::operator delete(p, std::align_val_t{4096});
    });
    const auto pooled = [&](bool compressed) {
        auto o          = csf_options(compressed);
        o.pool          = pool;
        o.pool_identity = "csf";
        return o;
    };
    {
        HostExpertTier tier(store, pooled(true));
        check(tier.prefill(ranked) == 8 && tier.stats().kept == 0, "csf pool: the first tier reads its records");
    }
    {
        HostExpertTier tier(store, pooled(true));
        bool all = true;
        check(tier.prefill(ranked) == 8 && tier.stats().kept == 8 && tier.stats().prefill_reads == 0,
              "csf pool: a CSF tier keeps all 8 records");
        for (std::uint32_t k = 0; k < 8; ++k) {
            all = all && key_matches(tier, k) && csf::tagged(tier.record_tagged(k)) == (k != kRandomScales);
        }
        check(all && tier.csf_slots() == 7, "csf pool: the kept records keep their formats");
    }
    {
        HostExpertTier tier(store, pooled(false));
        bool all = true;
        check(tier.prefill(ranked) == 8 && tier.stats().kept == 1 && tier.stats().prefill_reads == 7,
              "csf pool: a tier without CSF keeps only the expanded record");
        for (std::uint32_t k = 0; k < 8; ++k) { all = all && std::memcmp(tier.record(k), want[k].data(), bytes) == 0; }
        check(all && tier.record_tagged(0) == tier.record(0), "csf pool: its records are expanded");
    }
    pool.reset();
    if (fetch) {
        // A fetch of a compressed resident lands an expanded copy in a ring slot (the device stages expanded records).
        infernix::ops::offloaded_moe::FetchChannel channel;
        auto o  = csf_options(true);
        o.fetch = &channel;
        HostExpertTier tier(store, o);
        (void)tier.prefill(ranked);
        tier.begin_round(32);
        DeviceSide device{channel.channel(0)};
        device.publish(0, {1, kRandomScales, 10});
        std::uint32_t status = 0;
        const std::uint8_t* a = device.wait(0, status);
        const std::uint8_t* b = device.wait(1, status);
        const std::uint8_t* c = device.wait(2, status);
        check(status == 0 && a != nullptr && a != tier.record(1) && !csf::tagged(a) && csf_matches(a, want[1], tier.csf_pitch()),
              "csf fetch: a compressed resident lands as an expanded copy");
        check(b == tier.record(kRandomScales) && csf_matches(b, want[kRandomScales], tier.csf_pitch()) &&
                  csf_matches(c, want[10], tier.csf_pitch()),
              "csf fetch: an expanded resident lands in place, an SSD-only record read");
        device.consume(3);
        tier.end_round();
        tier.begin_round(32);
        check(tier.stats().csf_fetch_expansions == 1, "csf fetch: one expansion");
        tier.end_round();
    }
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Files files = make_files();
        test_csf(argc > 1 && std::string_view(argv[1]) == "--fetch");
        std::vector<std::vector<float>> multipliers(kLayers, std::vector<float>(3 * kExperts, 1.0F));
        const ExpertStore store({files.a, files.b}, std::vector<std::uint64_t>(kLayers, kRecord), kExperts, files.records,
                                multipliers);
        check(store.straddling() == 1, "one straddling record");
        test_mixed_strides();
        test_stripe(store);
        {
            HostExpertTier tier(store, options());
            // Startup pre-fill: keys 0..49 fill the 50 resident slots, best first.
            std::vector<std::uint32_t> ranked(kKeys);
            for (std::uint32_t k = 0; k < kKeys; ++k) { ranked[k] = k; }
            check(tier.prefill(ranked) == 50, "pre-fill fills the resident slots");
            bool all = true;
            for (std::uint32_t k = 0; k < 50; ++k) { all = all && matches(tier.record(k), k); }
            check(all, "pre-filled records equal the file bytes");
            // Keys 0..49 follow each other in file A: scatter reads of at most 16 records each.
            check(tier.stats().run_reads == 4 && tier.stats().run_records == 50,
                  "the pre-fill reads neighbouring records as scatter reads of up to 16");
            check(tier.record(50) == nullptr, "a key beyond the pre-fill is SSD-only");
            std::vector<std::uint32_t> dirty;
            tier.take_dirty(dirty);
            check(dirty.size() == 50, "pre-filled keys are dirty");

            // A round: demands from the service thread, a straddler, a resident key and a duplicate.
            tier.begin_round(32);
            std::vector<std::uint32_t> statuses;
            const std::vector<std::uint32_t> keys{55, kStraddle, 70, 3, 55, 90};
            std::vector<bool> valid;
            const auto got = serve(tier, keys, statuses, valid);
            bool landed = true;
            for (std::size_t i = 0; i < keys.size(); ++i) { landed = landed && valid[i]; }
            check(landed, "demanded records (straddling one included) equal the file bytes");
            check(got[3] == tier.record(3), "a resident key is served from its slot");
            check(got[0] == got[4], "a duplicate demand shares the landing");
            // The demanded keys are hot: at the next boundary they replace the coldest residents.
            const std::vector<std::uint32_t> hot{55, kStraddle, 70, 90};
            tier.record_uses(1, hot, 50.0);
            tier.begin_round(32);
            check(tier.stats().admitted == 4, "four landings admitted");
            bool admitted = true;
            for (const auto k : hot) { admitted = admitted && matches(tier.record(k), k); }
            check(admitted, "admitted records hold their bytes in RAM");
            std::uint32_t evicted = 0;
            for (std::uint32_t k = 0; k < 50; ++k) { evicted += tier.record(k) == nullptr ? 1U : 0U; }
            check(evicted == 4, "four residents lost their host copy");
            dirty.clear();
            tier.take_dirty(dirty);
            check(dirty.size() == 8, "the dirty set names the admitted keys and their victims");
            tier.controller().check();

            // More distinct demands than ring slots: finished landings give their slots back.
            std::vector<std::uint32_t> many;
            for (std::uint32_t k = 91; k < 111; ++k) { many.push_back(k); }
            (void)serve(tier, many, statuses, valid, 4);
            bool all_many = true;
            for (std::size_t i = 0; i < many.size(); ++i) { all_many = all_many && valid[i]; }
            check(all_many, "20 demands through an 8-slot ring land correctly");
            tier.begin_round(32);
            tier.controller().check();

            // An unreadable record fails its ticket; the next round still serves.
            const auto bad = serve(tier, {kKeys - 1}, statuses, valid);
            check(bad[0] == nullptr && statuses[0] == EIO, "a read past the file's end reports EIO");
            tier.begin_round(32);
            const auto again = serve(tier, {112}, statuses, valid);
            check(again[0] != nullptr && matches(again[0], 112), "the tier serves after a failed read");
            std::cout << "host expert tier: " << tier.stats().demand_reads << " demand reads, "
                      << tier.stats().demand_failures << " failed, " << tier.stats().admitted << " admitted\n";
            tier.begin_round(32);
        }
        {
            // The warm start's second pass: the frames took keys 0 and 1 (their RAM copies become
            // shadows and are released), and the refill skips keys that still have a copy.
            HostExpertTier tier(store, options());
            std::vector<std::uint32_t> ranked(50);
            for (std::uint32_t k = 0; k < 50; ++k) { ranked[k] = k; }
            check(tier.prefill(ranked) == 50, "warm fill: the first pass fills the resident slots");
            auto& host = tier.controller();
            for (const std::uint32_t k : {0U, 1U}) {
                host.queued(k);
                host.promotion_issued(k);
                host.promotion_completed(k, true);
            }
            check(host.release_shadows() == 2 && tier.record(0) == nullptr && tier.record(1) == nullptr,
                  "warm fill: the seeds' RAM copies are released");
            const std::vector<std::uint32_t> next{5, 6, 50, 51, 52};
            check(tier.prefill(next) == 2, "warm fill: the refill skips keys with a copy and stops when full");
            check(matches(tier.record(50), 50) && matches(tier.record(51), 51) && tier.record(52) == nullptr,
                  "warm fill: the refilled records equal the file bytes");
            host.check();

            // A demotion takes a slot of the demotion list; the next boundary refills the list.
            const std::vector<std::uint32_t> hot{60};
            tier.record_uses(1, hot, 1000.0);
            std::uint32_t target = 0;
            check(host.vram_evicted(60, true, target) == expert_cache::VramEviction::kDemote &&
                      host.demotion_slots() == 3,
                  "a demotion takes a slot of the demotion list");
            tier.begin_round(32);
            check(host.demotion_slots() == 4, "the boundary refills the demotion list");
            tier.end_round();
            host.demotion_completed(60, target, false);
            check(tier.record(60) != nullptr, "the demoted record is in RAM");
            host.check();
        }
        test_unpinned(store);
        test_resize(store);
        test_resize_unpinned(store);
        test_record_fills(store);
        test_slot_pool(store);
        if (argc > 1 && std::string_view(argv[1]) == "--fetch") {
            test_fetch(store);
            test_read_ahead(store);
            test_fetch_unpinned(store);
        }
        test_stream_reads(store);
        {
            // Destruction with reads in flight.
            HostExpertTier tier(store, options());
            tier.begin_round(32);
            std::thread service([&] {
                for (std::uint32_t k = 60; k < 68; ++k) { (void)tier.demand(static_cast<int>(k / kExperts), static_cast<int>(k % kExperts)); }
            });
            service.join();
        }
        std::error_code ec;
        std::filesystem::remove(files.a, ec);
        std::filesystem::remove(files.b, ec);
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "host expert tier: all checks passed\n";
    return 0;
}
