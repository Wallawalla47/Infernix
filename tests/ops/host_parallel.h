#pragma once

#include <algorithm>
#include <cstdint>
#include <exception>
#include <thread>
#include <vector>

namespace infernix::test {

inline std::int64_t host_thread_count() {
    return std::max<std::int64_t>(1, std::thread::hardware_concurrency());
}

// Calls `body(begin, end)` over contiguous ranges that partition [0, count), on up to `threads`
// host threads (inline when one suffices), and rethrows the first worker exception. Each index
// must be evaluated independently of the partition and write only its own outputs, so the result
// is identical for every thread count.
template <typename Body>
void parallel_ranges(std::int64_t count, std::int64_t threads, const Body& body) {
    threads = std::min(threads, count);
    if (threads <= 1) {
        if (count > 0) body(std::int64_t{0}, count);
        return;
    }
    std::vector<std::thread> workers;
    std::vector<std::exception_ptr> errors(static_cast<std::size_t>(threads));
    workers.reserve(static_cast<std::size_t>(threads));
    for (std::int64_t t = 0; t < threads; ++t) {
        workers.emplace_back([&, t] {
            try {
                body(count * t / threads, count * (t + 1) / threads);
            } catch (...) { errors[static_cast<std::size_t>(t)] = std::current_exception(); }
        });
    }
    for (std::thread& worker : workers) { worker.join(); }
    for (const std::exception_ptr& error : errors) {
        if (error) { std::rethrow_exception(error); }
    }
}

// Rows of fixture or oracle work below this many are cheaper inline than on spawned threads.
inline std::int64_t threads_for_rows(std::int64_t rows, std::int64_t minimum_rows = 256) {
    return rows < minimum_rows ? 1 : host_thread_count();
}

} // namespace infernix::test
