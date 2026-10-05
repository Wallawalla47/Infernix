// The SSD tier's host level (design §19.3.7, R10 host part): HostExpertTier over a synthetic two-file
// "artifact" with a record straddling the files, plain aligned memory as slots. Checks against a
// closed-form byte oracle: the startup pre-fill, demand reads from another thread (the CPU
// service's role) including a straddling record, a resident key served without a read, duplicate
// demands sharing a ticket, admission of hot landings at the next boundary (the victims lose their
// host copy, the dirty set names both), ring exhaustion reusing finished slots, a failed read, and
// destruction with reads in flight. CPU and disk only.
#include "models/qwen4_exp/expert_store.h"
#include "models/qwen4_exp/program/host_expert_tier.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using ninfer::models::qwen4_exp::ExpertStore;
using ninfer::models::qwen4_exp::HostExpertTier;

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
    const auto dir = std::filesystem::temp_directory_path();
    f.a = dir / "ninfer_host_tier_a.bin";
    f.b = dir / "ninfer_host_tier_b.bin";
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

} // namespace

int main() {
    try {
        const Files files = make_files();
        std::vector<std::vector<float>> multipliers(kLayers, std::vector<float>(3 * kExperts, 1.0F));
        const ExpertStore store({files.a, files.b}, kRecord, kLayers, kExperts, files.records, multipliers);
        check(store.straddling() == 1, "one straddling record");
        {
            HostExpertTier tier(store, options());
            // Startup pre-fill: keys 0..49 fill the 50 resident slots, best first.
            std::vector<std::uint32_t> ranked(kKeys);
            for (std::uint32_t k = 0; k < kKeys; ++k) { ranked[k] = k; }
            check(tier.prefill(ranked) == 50, "pre-fill fills the resident slots");
            bool all = true;
            for (std::uint32_t k = 0; k < 50; ++k) { all = all && matches(tier.record(k), k); }
            check(all, "pre-filled records equal the file bytes");
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
