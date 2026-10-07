// Core DirectReadQueue (design §19.3.7 RT3b): unbuffered reads of a synthetic file against a
// closed-form byte oracle, with out-of-order completion, demand-before-prefetch issue order,
// prefetch sub-reads, transient-error backoff, retry of a hard error, timeouts, short reads,
// delayed completions, cancel_all and destruction with reads in flight. CPU and disk only.
#include "core/direct_read_queue.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <new>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using infernix::DirectReadQueue;
using Clock = std::chrono::steady_clock;

constexpr std::size_t kBlock     = DirectReadQueue::kAlignment;
constexpr std::size_t kFileBytes = 64 * 1024 * 1024 + 1000; // the last block is partial

std::uint8_t oracle(std::uint64_t at) {
    return static_cast<std::uint8_t>((at * 2654435761ULL >> 13) ^ (at >> 20) ^ 0x5a);
}

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::cerr << "FAIL: " << what << '\n';
    }
}

struct Aligned {
    std::byte* p = nullptr;
    std::size_t bytes = 0;
    explicit Aligned(std::size_t n) : p(static_cast<std::byte*>(::operator new(n, std::align_val_t{kBlock}))), bytes(n) {}
    ~Aligned() { ::operator delete(p, std::align_val_t{kBlock}); }
    Aligned(const Aligned&)            = delete;
    Aligned& operator=(const Aligned&) = delete;
};

bool matches(const std::byte* data, std::uint64_t offset, std::size_t bytes) {
    for (std::size_t i = 0; i < bytes; ++i) {
        if (static_cast<std::uint8_t>(data[i]) != oracle(offset + i)) { return false; }
    }
    return true;
}

// Polls until `count` completions arrived (or 20 s), in arrival order.
std::vector<DirectReadQueue::Completion> drain(DirectReadQueue& queue, std::size_t count) {
    std::vector<DirectReadQueue::Completion> out;
    const auto stop = Clock::now() + std::chrono::seconds(20);
    while (out.size() < count && Clock::now() < stop) { queue.poll(out, std::chrono::milliseconds(50)); }
    return out;
}

std::filesystem::path make_file() {
    const auto path = std::filesystem::temp_directory_path() / "infernix_direct_read_queue_test.bin";
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    std::vector<char> chunk(1 << 20);
    for (std::uint64_t at = 0; at < kFileBytes; at += chunk.size()) {
        const std::size_t n = static_cast<std::size_t>(std::min<std::uint64_t>(chunk.size(), kFileBytes - at));
        for (std::size_t i = 0; i < n; ++i) { chunk[i] = static_cast<char>(oracle(at + i)); }
        f.write(chunk.data(), static_cast<std::streamsize>(n));
    }
    if (!f) { throw std::runtime_error("cannot write the test file"); }
    return path;
}

// Many reads of mixed sizes in random order: each completes once, Ok, with the file's bytes.
void test_reads(const std::filesystem::path& path) {
    DirectReadQueue queue;
    const auto file = queue.open(path);
    check(queue.file_bytes(file) == kFileBytes, "file size");
    std::mt19937_64 rng(7);
    struct Planned { std::uint64_t offset; std::size_t bytes; std::size_t at; DirectReadQueue::Priority priority; };
    std::vector<Planned> plan;
    std::size_t total = 0;
    for (int i = 0; i < 96; ++i) {
        const std::size_t blocks = 1 + rng() % 600; // up to 2.4 MB, a record is 2.76 MB
        const std::uint64_t last = kFileBytes / kBlock - blocks;
        plan.push_back({(rng() % last) * kBlock, blocks * kBlock, total,
                        i % 3 == 0 ? DirectReadQueue::Priority::Prefetch : DirectReadQueue::Priority::Demand});
        total += blocks * kBlock;
    }
    Aligned buffer(total);
    for (std::size_t i = 0; i < plan.size(); ++i) {
        queue.submit({file, plan[i].offset, plan[i].bytes, buffer.p + plan[i].at, plan[i].priority, i});
    }
    check(queue.pending() == plan.size(), "pending after submit");
    const auto done = drain(queue, plan.size());
    check(done.size() == plan.size(), "every read completes");
    std::vector<int> seen(plan.size(), 0);
    bool in_order = true;
    for (std::size_t k = 0; k < done.size(); ++k) {
        const auto tag = done[k].tag;
        check(tag < plan.size() && ++seen[tag] == 1, "each read completes once");
        check(done[k].status == DirectReadQueue::Status::Ok && done[k].attempts == 1, "read " + std::to_string(tag) + " Ok");
        in_order = in_order && tag == k;
    }
    for (std::size_t i = 0; i < plan.size(); ++i) {
        check(matches(buffer.p + plan[i].at, plan[i].offset, plan[i].bytes), "bytes of read " + std::to_string(i));
    }
    check(queue.pending() == 0, "nothing pending after the reads");
    std::cout << "reads: " << done.size() << " completions, " << (in_order ? "in submit order" : "out of order")
              << ", " << queue.stats().issued << " device reads\n";
}

// With one read in flight overall, a demand read submitted after prefetch reads is issued first,
// and a prefetch read of 1 MiB is issued as four 256 KiB sub-reads.
void test_priority_and_split(const std::filesystem::path& path) {
    DirectReadQueue::Options o;
    o.demand_depth = 1;
    o.prefetch_depth = 1;
    o.total_depth = 1;
    DirectReadQueue queue(o);
    const auto file = queue.open(path);
    Aligned buffer(5 * (1 << 20));
    for (std::uint64_t i = 0; i < 4; ++i) {
        queue.submit({file, i << 20, 1 << 20, buffer.p + (i << 20), DirectReadQueue::Priority::Prefetch, 10 + i});
    }
    queue.submit({file, 8 << 20, 1 << 20, buffer.p + (4 << 20), DirectReadQueue::Priority::Demand, 1});
    const auto done = drain(queue, 5);
    check(done.size() == 5 && done[0].tag == 1, "the demand read completes first");
    check(queue.stats().issued == 1 + 4 * 4, "prefetch reads are issued as 256 KiB sub-reads");
    for (std::uint64_t i = 0; i < 4; ++i) { check(matches(buffer.p + (i << 20), i << 20, 1 << 20), "prefetch bytes"); }
    check(matches(buffer.p + (4 << 20), 8 << 20, 1 << 20), "demand bytes");
}

// Faults: transient backoff, a hard error retried once and failing twice, a timeout retried once
// and twice, a short read at the end of the file, and a delayed completion.
void test_faults(const std::filesystem::path& path) {
    Aligned buffer(1 << 20);
    const auto one = [&](DirectReadQueue& queue, std::uint32_t file, std::uint64_t offset, std::size_t bytes) {
        queue.submit({file, offset, bytes, buffer.p, DirectReadQueue::Priority::Demand, 42});
        const auto done = drain(queue, 1);
        if (done.size() != 1) { throw std::runtime_error("a faulted read did not complete"); }
        return done[0];
    };
    {
        DirectReadQueue queue;
        const auto file = queue.open(path);
        queue.set_faults({.fail_read = 1, .fail_count = 3, .transient = true});
        const auto c = one(queue, file, 0, 64 * 1024);
        check(c.status == DirectReadQueue::Status::Ok && c.attempts == 4, "transient errors are retried");
        check(c.latency_ns >= 7'000'000, "transient retries back off 1 + 2 + 4 ms");
        check(queue.stats().transient_retries == 3 && matches(buffer.p, 0, 64 * 1024), "transient retry bytes");
    }
    {
        DirectReadQueue::Options o;
        o.transient_bound = std::chrono::milliseconds(20);
        DirectReadQueue queue(o);
        const auto file = queue.open(path);
        queue.set_faults({.fail_read = 1, .fail_count = 1000, .transient = true});
        const auto c = one(queue, file, 0, 64 * 1024);
        check(c.status == DirectReadQueue::Status::Failed && c.error, "transient errors past the bound fail");
    }
    {
        DirectReadQueue queue;
        const auto file = queue.open(path);
        queue.set_faults({.fail_read = 1, .fail_count = 1});
        auto c = one(queue, file, 1 << 20, 64 * 1024);
        check(c.status == DirectReadQueue::Status::Ok && c.attempts == 2 && matches(buffer.p, 1 << 20, 64 * 1024),
              "a hard error is retried once");
        queue.set_faults({.fail_read = queue.stats().issued + 1, .fail_count = 2});
        c = one(queue, file, 0, 64 * 1024);
        check(c.status == DirectReadQueue::Status::Failed && c.attempts == 2 && c.error, "a second hard error fails");
    }
    {
        DirectReadQueue::Options o;
        o.timeout = std::chrono::milliseconds(30);
        DirectReadQueue queue(o);
        const auto file = queue.open(path);
        queue.set_faults({.stall_read = 1});
        auto c = one(queue, file, 0, 64 * 1024);
        check(c.status == DirectReadQueue::Status::Ok && c.attempts == 2 && c.latency_ns >= 30'000'000,
              "a read past its timeout is cancelled and retried once");
        queue.set_faults({.stall_read = queue.stats().issued + 1, .stall_count = 2});
        c = one(queue, file, 0, 64 * 1024);
        check(c.status == DirectReadQueue::Status::TimedOut && c.attempts == 2, "a second timeout fails the read");
        check(queue.stats().timeouts == 3, "timeouts counted");
    }
    {
        DirectReadQueue queue;
        const auto file = queue.open(path);
        const std::uint64_t last = (kFileBytes / kBlock) * kBlock; // the partial block
        const auto c = one(queue, file, last, kBlock);
        check(c.status == DirectReadQueue::Status::Failed && c.attempts == 1, "a short read at the end fails at once");
        const auto past = one(queue, file, last + 64 * kBlock, kBlock);
        check(past.status == DirectReadQueue::Status::Failed, "a read past the end fails");
    }
    {
        DirectReadQueue queue;
        const auto file = queue.open(path);
        queue.set_faults({.delay = std::chrono::milliseconds(25)});
        const auto c = one(queue, file, 0, 64 * 1024);
        check(c.status == DirectReadQueue::Status::Ok && c.latency_ns >= 25'000'000, "a delayed completion is held");
    }
    try {
        DirectReadQueue queue;
        const auto file = queue.open(path);
        queue.submit({file, 100, kBlock, buffer.p, DirectReadQueue::Priority::Demand, 0});
        check(false, "a misaligned offset is rejected");
    } catch (const std::invalid_argument&) {}
}

// cancel_all with reads queued and in flight: every request completes once, Cancelled or Ok, and
// Ok ones hold their bytes; a queue destroyed with reads in flight waits for them.
void test_cancel(const std::filesystem::path& path) {
    Aligned buffer(64 * (1 << 20));
    {
        DirectReadQueue queue;
        const auto file = queue.open(path);
        queue.set_faults({.stall_read = 1, .stall_count = 4});
        for (std::uint64_t i = 0; i < 64; ++i) {
            queue.submit({file, i << 20, 1 << 20, buffer.p + (i << 20),
                          i % 2 ? DirectReadQueue::Priority::Prefetch : DirectReadQueue::Priority::Demand, i});
        }
        std::vector<DirectReadQueue::Completion> early;
        queue.poll(early, std::chrono::microseconds(0));
        queue.cancel_all();
        auto done = early;
        queue.poll(done, std::chrono::microseconds(0));
        std::map<std::uint64_t, int> seen;
        std::size_t cancelled = 0;
        for (const auto& c : done) {
            ++seen[c.tag];
            if (c.status == DirectReadQueue::Status::Cancelled) {
                ++cancelled;
            } else {
                check(c.status == DirectReadQueue::Status::Ok && matches(buffer.p + (c.tag << 20), c.tag << 20, 1 << 20),
                      "a read finished before the cancel holds its bytes");
            }
        }
        check(done.size() == 64 && seen.size() == 64, "cancel_all completes every request once");
        check(cancelled > 0 && queue.pending() == 0, "cancel_all cancels queued reads");
        // The queue serves new reads after a cancel.
        queue.set_faults({});
        queue.submit({file, 0, 1 << 20, buffer.p, DirectReadQueue::Priority::Demand, 99});
        const auto again = drain(queue, 1);
        check(again.size() == 1 && again[0].status == DirectReadQueue::Status::Ok && matches(buffer.p, 0, 1 << 20),
              "reads after cancel_all");
    }
    {
        DirectReadQueue queue;
        const auto file = queue.open(path);
        for (std::uint64_t i = 0; i < 32; ++i) {
            queue.submit({file, i << 20, 1 << 20, buffer.p + (i << 20), DirectReadQueue::Priority::Demand, i});
        }
        std::vector<DirectReadQueue::Completion> some;
        queue.poll(some, std::chrono::microseconds(0));
    } // destroyed with reads in flight
}

} // namespace

int main() {
    try {
        const auto path = make_file();
        test_reads(path);
        test_priority_and_split(path);
        test_faults(path);
        test_cancel(path);
        std::error_code ec;
        std::filesystem::remove(path, ec);
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "direct read queue: all checks passed\n";
    return 0;
}
