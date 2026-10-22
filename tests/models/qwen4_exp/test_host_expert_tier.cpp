// The SSD tier's host level (design §19.3.7, R10 host part): HostExpertTier over a synthetic two-file
// "artifact" with a record straddling the files, plain aligned memory as slots. Checks against a
// closed-form byte oracle: the startup pre-fill, demand reads from another thread (the CPU
// service's role) including a straddling record, a resident key served without a read, duplicate
// demands sharing a ticket, admission of hot landings at the next boundary (the victims lose their
// host copy, the dirty set names both), ring exhaustion reusing finished slots, a failed read,
// destruction with reads in flight, and unpinned records (a refused pin chunk becoming unpinned
// records, their startup fill, CPU demands served in place and never admitted). CPU and disk only,
// except with --fetch: the tier as the responder
// of a fetch channel (mapped words, so a CUDA context), this thread playing the device's half of the
// protocol (ops/offloaded_sparse_moe/cpu/fetch_request.h): records landed from RAM and read from disk,
// more records than ring slots (slots reused as the "device" consumes them), the heartbeat, a request
// failing on an unreadable record, and admission of fetched landings at the next boundary. An
// expert stripe moves each record's tail to another file; records still land whole.
#include "test_temp_path.h"
#include "runtime/expert_offload/expert_store.h"
#include "runtime/expert_offload/host_expert_tier.h"
#include "ops/offloaded_sparse_moe/cpu/fetch_channel.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
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
          "mixed strides: slots of the largest pitch, each layer its own, no scale tail");
    HostExpertTier::Options o = options();
    o.slots = 24; // ring 8 + prefetch 2 + demotion 4 + 10 resident slots
    HostExpertTier tier(store, o);
    // Layer 1's keys 20..29 follow each other in the file: one scatter run of half-size records.
    std::vector<std::uint32_t> ranked;
    for (std::uint32_t k = kExpertsHere; k < kExpertsHere + 10; ++k) { ranked.push_back(k); }
    check(tier.prefill(ranked) == 10, "mixed strides: the pre-fill fills the resident slots");
    bool all = true;
    for (const auto k : ranked) { all = all && holds(tier.record(k), k); }
    check(all && tier.stats().run_reads == 1 && tier.stats().run_records == 10,
          "mixed strides: a run of short records lands in full-size slots");
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

} // namespace

int main(int argc, char** argv) {
    try {
        const Files files = make_files();
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
