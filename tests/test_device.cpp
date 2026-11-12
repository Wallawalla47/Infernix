#include "core/device.h"

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <thread>
#include <vector>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace {

int fail(const char* message) {
    std::cerr << message << '\n';
    return 1;
}

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

int expect_throws_device(int device_id) {
    try {
        infernix::DeviceContext invalid(device_id);
    } catch (const std::runtime_error&) { return 0; }
    std::cerr << "DeviceContext(" << device_id << ") did not throw\n";
    return 1;
}

int check_context(const infernix::DeviceContext& ctx, const char* label) {
    int failures = 0;
    if (ctx.stream == nullptr) {
        std::cerr << label << " compute stream is null\n";
        ++failures;
    }
    if (ctx.transfer_stream == nullptr) {
        std::cerr << label << " load stream is null\n";
        ++failures;
    }
    if (ctx.compute_capability() <= 0) {
        std::cerr << label << " compute capability is not positive\n";
        ++failures;
    }
    if (ctx.total_vram() == 0) {
        std::cerr << label << " total_vram is zero\n";
        ++failures;
    }
    return failures;
}

void CUDART_CB hold_stream(void* flag) {
    while (!static_cast<std::atomic<bool>*>(flag)->load()) { std::this_thread::yield(); }
}

// Fills `words` words of `p` with `seed` + index.
void fill_words(std::uint32_t* p, std::size_t words, std::uint32_t seed) {
    for (std::size_t i = 0; i < words; ++i) { p[i] = seed + static_cast<std::uint32_t>(i); }
}

std::size_t first_mismatch(const std::uint32_t* p, std::size_t words, std::uint32_t seed) {
    for (std::size_t i = 0; i < words; ++i) {
        if (p[i] != seed + static_cast<std::uint32_t>(i)) { return i; }
    }
    return words;
}

// upload_pinned copies its source as it is at the call. A caller that rewrites its buffer right after the call, while
// its stream has not reached the copy yet (held here by a host function, as a host running a boundary ahead of the
// device leaves it), still delivers the bytes it passed (ZL-01: the expert residency's frame table, re-staged for the
// next boundary before the queued upload ran, gave kernels a later table). Also the staging ring's wrap and the
// pageable copy of an upload too large for it.
int check_upload_snapshot(const infernix::DeviceContext& ctx) {
    int failures = 0;
    // Held: one frame table (48 layers x 512 experts x 4 bytes), rewritten before the stream runs the upload.
    constexpr std::size_t kTable = 48 * 512 * 4;
    // Past half of the 32 MiB staging ring: the pageable copy instead.
    constexpr std::size_t kLarge = 24ULL << 20;
    std::uint32_t* src  = nullptr;
    std::uint32_t* dst  = nullptr;
    std::uint32_t* back = nullptr;
    CUDA_CHECK(cudaMallocHost(&src, kLarge));
    CUDA_CHECK(cudaMalloc(&dst, kLarge));
    CUDA_CHECK(cudaMallocHost(&back, kLarge));
    const auto held_upload = [&](std::size_t bytes, std::uint32_t seed, const char* what) {
        const std::size_t words = bytes / 4;
        fill_words(src, words, seed);
        std::atomic<bool> release{false};
        CUDA_CHECK(cudaLaunchHostFunc(ctx.stream, hold_stream, &release));
        infernix::upload_pinned(dst, src, bytes, ctx.stream);
        fill_words(src, words, seed ^ 0xFFFF0000U); // the next boundary's bytes
        release.store(true);
        CUDA_CHECK(cudaMemcpyAsync(back, dst, bytes, cudaMemcpyDeviceToHost, ctx.stream));
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
        if (const std::size_t at = first_mismatch(back, words, seed); at != words) {
            std::cerr << "upload_pinned (" << what << ") delivered word " << at << " = 0x" << std::hex << back[at]
                      << std::dec << ", rewritten after the call, not the bytes it was given\n";
            ++failures;
        }
    };
    held_upload(kTable, 0xA5000000U, "a frame table behind a held stream");
    // Wrap: uploads of varying sizes until the ring has been reused, each source rewritten right after its call, every
    // destination read back after each 48.
    constexpr std::size_t kRegion = 512 * 1024;
    constexpr int kUploads        = 96; // 96 x up to 512 KiB: half again the 32 MiB ring
    std::vector<std::uint32_t> seeds;
    for (int u = 0; u < kUploads; ++u) {
        const std::size_t bytes = kRegion - static_cast<std::size_t>(u % 7) * 4096 - 16;
        const auto seed         = static_cast<std::uint32_t>(0x10000000U + static_cast<std::uint32_t>(u) * 0x00100000U);
        auto* at                = src + (static_cast<std::size_t>(u) % 2) * (kRegion / 4);
        fill_words(at, bytes / 4, seed);
        infernix::upload_pinned(dst + static_cast<std::size_t>(u % 48) * (kRegion / 4), at, bytes, ctx.stream);
        fill_words(at, bytes / 4, ~seed);
        seeds.push_back(seed);
        if (u % 48 == 47) { // every destination region written once since the last check
            CUDA_CHECK(cudaMemcpyAsync(back, dst, 48 * kRegion, cudaMemcpyDeviceToHost, ctx.stream));
            CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
            for (int r = 0; r < 48; ++r) {
                const int v             = u - 47 + r;
                const std::size_t bytes_v = kRegion - static_cast<std::size_t>(v % 7) * 4096 - 16;
                const std::uint32_t* got  = back + static_cast<std::size_t>(r) * (kRegion / 4);
                if (const std::size_t at_word = first_mismatch(got, bytes_v / 4, seeds[static_cast<std::size_t>(v)]);
                    at_word != bytes_v / 4) {
                    std::cerr << "upload_pinned upload " << v << " (staging ring reuse) differs at word " << at_word << '\n';
                    ++failures;
                    break;
                }
            }
        }
    }
    // Larger than half the ring: copied from pageable memory, which cudaMemcpyAsync stages before it returns, so a
    // rewrite after the call cannot reach it either. (Not behind a held stream: whether the driver's staging of a
    // pageable copy waits for the stream is the driver's choice.)
    {
        const std::size_t words = kLarge / 4;
        fill_words(src, words, 0x3C000000U);
        infernix::upload_pinned(dst, src, kLarge, ctx.stream);
        fill_words(src, words, 0xC3FF0000U);
        CUDA_CHECK(cudaMemcpyAsync(back, dst, kLarge, cudaMemcpyDeviceToHost, ctx.stream));
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
        if (const std::size_t at = first_mismatch(back, words, 0x3C000000U); at != words) {
            std::cerr << "upload_pinned (24 MiB, the pageable copy) delivered word " << at << " = 0x" << std::hex << back[at]
                      << std::dec << ", not the bytes it was given\n";
            ++failures;
        }
    }
    CUDA_CHECK(cudaFreeHost(back));
    CUDA_CHECK(cudaFree(dst));
    CUDA_CHECK(cudaFreeHost(src));
    return failures;
}

// upload_pinned_when: the copy waits for the published word, reads the data written after the
// launch (not the poison before it), counts its waits, and a graph replay with a new expected
// word rejects the word an earlier round left in place.
int check_pinned_gate(const infernix::DeviceContext& ctx) {
    int failures = 0;
    constexpr std::size_t kMaxBytes = 327680;
    std::byte* src             = nullptr;
    std::uint32_t* ready       = nullptr;
    std::uint32_t* expected_h  = nullptr;
    void* dst                  = nullptr;
    std::uint32_t* expected    = nullptr;
    std::uint64_t* stats       = nullptr;
    CUDA_CHECK(cudaMallocHost(&src, kMaxBytes));
    CUDA_CHECK(cudaMallocHost(&ready, 64));
    CUDA_CHECK(cudaMallocHost(&expected_h, 64));
    CUDA_CHECK(cudaMalloc(&dst, kMaxBytes));
    CUDA_CHECK(cudaMalloc(&expected, sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMalloc(&stats, 2 * sizeof(std::uint64_t)));
    std::uint32_t word = 0;
    infernix::publish_pinned_word(ready, 0);
    // Every fill below is issued on ctx.stream: a plain cudaMemset runs on the legacy stream, which does
    // not order against the non-blocking compute stream, so a gate could read stats or dst before it.
    std::vector<std::byte> host(kMaxBytes);
    const auto fill = [&](std::size_t bytes, std::uint32_t round) {
        for (std::size_t i = 0; i < bytes; ++i) { host[i] = static_cast<std::byte>((i * 131 + round * 7 + 3) & 0xff); }
    };
    // Poison, launch, wait until the gate waits, write the data, publish; returns whether the copy
    // holds the data. The host waits for the stream to pass a marker just before the gate, then
    // 200 ms: a fixed sleep after the launch alone let a busy GPU (other processes' kernels) start
    // the gate after the publish, so it recorded no wait or a short one; under a parallel CTest run
    // the gate was seen starting about 19 ms after the marker.
    std::atomic<bool> reached{false};
    const auto produce = [&](std::size_t bytes, std::uint32_t round, const std::function<void()>& launch) {
        std::memset(src, 0xa5, bytes);
        *expected_h = ++word;
        CUDA_CHECK(cudaMemcpyAsync(expected, expected_h, sizeof(std::uint32_t), cudaMemcpyHostToDevice, ctx.stream));
        reached = false;
        CUDA_CHECK(cudaLaunchHostFunc(
            ctx.stream, [](void* flag) { static_cast<std::atomic<bool>*>(flag)->store(true); }, &reached));
        launch();
        ctx.flush();
        const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (!reached.load() && std::chrono::steady_clock::now() < until) { std::this_thread::yield(); }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        fill(bytes, round);
        std::memcpy(src, host.data(), bytes);
        infernix::publish_pinned_word(ready, word);
        ctx.synchronize();
        std::vector<std::byte> out(bytes);
        CUDA_CHECK(cudaMemcpy(out.data(), dst, bytes, cudaMemcpyDeviceToHost));
        return std::memcmp(out.data(), host.data(), bytes) == 0;
    };
    const auto read_stats = [&](std::uint64_t (&out)[2]) {
        CUDA_CHECK(cudaMemcpy(out, stats, sizeof(out), cudaMemcpyDeviceToHost));
    };
    std::uint32_t round = 0;
    for (const std::size_t bytes : {std::size_t{2560}, std::size_t{12800}, kMaxBytes}) {
        CUDA_CHECK(cudaMemsetAsync(stats, 0, 2 * sizeof(std::uint64_t), ctx.stream));
        if (!produce(bytes, ++round, [&] {
                infernix::upload_pinned_when(dst, src, bytes, ready, expected, stats, nullptr, {}, ctx.stream);
            })) {
            ++failures;
            std::cerr << "gated copy of " << bytes << " bytes does not hold the published data\n";
        }
        std::uint64_t waited[2] = {};
        read_stats(waited);
        if (waited[1] != 1 || waited[0] < 1000000) {
            ++failures;
            std::cerr << "gated copy of " << bytes << " bytes recorded " << waited[1] << " waits, " << waited[0]
                      << " ns (expected 1 wait of at least 1 ms)\n";
        }
    }
    // A published word passes at once and records no wait.
    {
        CUDA_CHECK(cudaMemsetAsync(stats, 0, 2 * sizeof(std::uint64_t), ctx.stream));
        *expected_h = ++word;
        CUDA_CHECK(cudaMemcpyAsync(expected, expected_h, sizeof(std::uint32_t), cudaMemcpyHostToDevice, ctx.stream));
        fill(2560, ++round);
        std::memcpy(src, host.data(), 2560);
        infernix::publish_pinned_word(ready, word);
        infernix::upload_pinned_when(dst, src, 2560, ready, expected, stats, nullptr, {}, ctx.stream);
        ctx.synchronize();
        std::uint64_t waited[2] = {};
        read_stats(waited);
        if (waited[1] != 0) {
            ++failures;
            std::cerr << "a gate whose word was published before the launch recorded a wait\n";
        }
    }
    // Graph replays: each round stages a new expected word; the old word stays published until
    // the host writes the round's data, so a gate that accepted a stale word would copy poison.
    {
        constexpr std::size_t bytes = 12800;
        cudaGraph_t graph       = nullptr;
        cudaGraphExec_t exec    = nullptr;
        CUDA_CHECK(cudaStreamBeginCapture(ctx.stream, cudaStreamCaptureModeThreadLocal));
        infernix::upload_pinned_when(dst, src, bytes, ready, expected, stats, nullptr, {}, ctx.stream);
        CUDA_CHECK(cudaStreamEndCapture(ctx.stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
        CUDA_CHECK(cudaMemsetAsync(stats, 0, 2 * sizeof(std::uint64_t), ctx.stream));
        for (int replay = 0; replay < 3; ++replay) {
            if (!produce(bytes, ++round, [&] { CUDA_CHECK(cudaGraphLaunch(exec, ctx.stream)); })) {
                ++failures;
                std::cerr << "graph replay " << replay << " copied a stale or partial round\n";
            }
        }
        std::uint64_t waited[2] = {};
        read_stats(waited);
        if (waited[1] != 3) {
            ++failures;
            std::cerr << "graph replays recorded " << waited[1] << " waits (expected 3)\n";
        }
        CUDA_CHECK(cudaGraphExecDestroy(exec));
        CUDA_CHECK(cudaGraphDestroy(graph));
    }
    try {
        infernix::upload_pinned_when(dst, src, 2561, ready, expected, nullptr, nullptr, {}, ctx.stream);
        ++failures;
        std::cerr << "a gated copy of a size that is not a 16-byte multiple did not throw\n";
    } catch (const std::invalid_argument&) {}
    // A word never published: past its bound (50 ms here) the gate stores its fault, skips the copy and
    // records no wait, and the context stays usable: the next gate of the stream copies its round.
    {
        constexpr std::size_t bytes        = 12800;
        constexpr std::uint32_t kFault     = 0x0001FF10;
        constexpr std::uint64_t kBoundNs   = 50ULL * 1000 * 1000;
        std::uint32_t* const fault         = expected_h + 1; // mapped pinned, beside the staged word
        *reinterpret_cast<volatile std::uint32_t*>(fault) = 0;
        CUDA_CHECK(cudaMemsetAsync(stats, 0, 2 * sizeof(std::uint64_t), ctx.stream));
        CUDA_CHECK(cudaMemsetAsync(dst, 0x5a, bytes, ctx.stream));
        *expected_h = ++word; // the host never publishes this round's word
        CUDA_CHECK(cudaMemcpyAsync(expected, expected_h, sizeof(std::uint32_t), cudaMemcpyHostToDevice, ctx.stream));
        std::memset(src, 0xa5, bytes);
        const auto start = std::chrono::steady_clock::now();
        infernix::upload_pinned_when(dst, src, bytes, ready, expected, stats, nullptr,
                                     infernix::DeviceFault{.word = fault, .value = kFault}, ctx.stream, kBoundNs);
        const cudaError_t status = cudaStreamSynchronize(ctx.stream);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        std::vector<std::byte> out(bytes);
        CUDA_CHECK(cudaMemcpy(out.data(), dst, bytes, cudaMemcpyDeviceToHost));
        std::uint64_t waited[2] = {};
        read_stats(waited);
        if (status != cudaSuccess || *reinterpret_cast<volatile std::uint32_t*>(fault) != kFault ||
            out != std::vector<std::byte>(bytes, std::byte{0x5a}) || waited[1] != 0 || ms < 40.0) {
            ++failures;
            std::cerr << "an unpublished gate past its bound: " << cudaGetErrorString(status) << " after " << ms
                      << " ms, fault word " << std::hex << *reinterpret_cast<volatile std::uint32_t*>(fault) << std::dec
                      << ", " << waited[1] << " waits (expected the fault, the copy skipped and no wait)\n";
        }
        if (!produce(bytes, ++round, [&] {
                infernix::upload_pinned_when(dst, src, bytes, ready, expected, stats, nullptr,
                                             infernix::DeviceFault{.word = fault, .value = kFault}, ctx.stream, kBoundNs * 100);
            })) {
            ++failures;
            std::cerr << "the gate after a timed-out one did not copy its round\n";
        }
    }
    CUDA_CHECK(cudaFree(stats));
    CUDA_CHECK(cudaFree(expected));
    CUDA_CHECK(cudaFree(dst));
    CUDA_CHECK(cudaFreeHost(expected_h));
    CUDA_CHECK(cudaFreeHost(ready));
    CUDA_CHECK(cudaFreeHost(src));
    return failures;
}

// Out of device memory is recoverable (fix sweep FS-01): CUDA_CHECK throws DeviceMemoryExhausted for a real
// cudaErrorMemoryAllocation and leaves the device usable, and the test injection fires once, on the arming
// thread only.
int check_memory_exhausted(const infernix::DeviceContext& ctx) {
    int failures         = 0;
    const auto expect    = [&](bool ok, const char* what) {
        if (!ok) {
            ++failures;
            std::cerr << "out of device memory: " << what << '\n';
        }
    };
    const auto device_works = [&] {
        int* value = nullptr;
        int host   = 0;
        CUDA_CHECK(cudaMalloc(&value, sizeof(int)));
        CUDA_CHECK(cudaMemsetAsync(value, 0x5a, sizeof(int), ctx.stream));
        CUDA_CHECK(cudaMemcpyAsync(&host, value, sizeof(int), cudaMemcpyDeviceToHost, ctx.stream));
        ctx.synchronize();
        CUDA_CHECK(cudaFree(value));
        return host == 0x5a5a5a5a;
    };
    void* huge = nullptr;
    bool thrown = false;
    try {
        CUDA_CHECK(cudaMalloc(&huge, std::size_t{1} << 50U));
    } catch (const infernix::DeviceMemoryExhausted& error) {
        thrown = std::string_view(error.what()).find("out of device memory") != std::string_view::npos;
    }
    expect(thrown, "a 1 PiB cudaMalloc did not throw DeviceMemoryExhausted naming the shortage");
    expect(cudaGetLastError() == cudaSuccess, "the thread's last error was not cleared");
    expect(infernix::settle_after_memory_exhausted(), "the device did not settle after the shortage");
    expect(device_works(), "the device did not allocate, set and copy after the shortage");

    const std::uint64_t injected = infernix::testing::memory_exhausted_injections();
    int* word = nullptr;
    CUDA_CHECK(cudaMalloc(&word, sizeof(int)));
    infernix::testing::arm_memory_exhausted_here(3);
    std::atomic<bool> other_thrown{false};
    std::thread other([&] {
        try {
            CUDA_CHECK(cudaSetDevice(ctx.device));
            for (int i = 0; i < 4; ++i) { CUDA_CHECK(cudaGetLastError()); }
        } catch (const infernix::DeviceMemoryExhausted&) { other_thrown = true; }
    });
    other.join();
    expect(!other_thrown.load(), "the injection fired on another thread");
    int fired_at = 0;
    for (int i = 1; i <= 4 && fired_at == 0; ++i) {
        try {
            CUDA_CHECK(cudaMemsetAsync(word, 0, sizeof(int), ctx.stream));
        } catch (const infernix::DeviceMemoryExhausted& error) {
            fired_at = std::string_view(error.what()).find("injected") != std::string_view::npos ? i : -i;
        }
    }
    expect(fired_at == 3, "the injection did not fire, named, at the third check after arming");
    expect(infernix::testing::memory_exhausted_injections() == injected + 1, "the injection was not counted once");
    bool again = false;
    try {
        for (int i = 0; i < 8; ++i) { CUDA_CHECK(cudaMemsetAsync(word, 0, sizeof(int), ctx.stream)); }
    } catch (const infernix::DeviceMemoryExhausted&) { again = true; }
    expect(!again, "the injection fired a second time");
    infernix::testing::arm_memory_exhausted_here(1);
    infernix::testing::arm_memory_exhausted_here(0);
    bool disarmed = true;
    try {
        CUDA_CHECK(cudaMemsetAsync(word, 0, sizeof(int), ctx.stream));
    } catch (const infernix::DeviceMemoryExhausted&) { disarmed = false; }
    expect(disarmed, "arming 0 did not disarm the injection");
    ctx.synchronize();
    CUDA_CHECK(cudaFree(word));
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--invalid-sync") {
        try {
            infernix::DeviceContext ctx(0);
        } catch (const std::invalid_argument& error) {
            return std::string_view(error.what()).find("INFERNIX_CUDA_SYNC") != std::string_view::npos
                       ? 0
                       : fail("invalid sync setting has no configuration diagnostic");
        }
        return fail("invalid sync setting did not fail before CUDA initialization");
    }
    const unsigned int expected_flags =
        argc == 2 ? static_cast<unsigned int>(std::stoul(argv[1])) : cudaDeviceScheduleSpin;
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }
    if (count == 0) {
        std::cout << "SKIP: no CUDA devices\n";
        return 77;
    }

    int failures = 0;

    infernix::DeviceContext ctx(0);
    unsigned int actual_flags = 0;
    CUDA_CHECK(cudaGetDeviceFlags(&actual_flags));
    if ((actual_flags & cudaDeviceScheduleMask) != expected_flags) {
        return fail("CUDA did not apply the requested synchronization schedule");
    }
    if (ctx.device != 0) {
        ++failures;
        std::cerr << "ctx.device expected 0, got " << ctx.device << '\n';
    }
    failures += check_context(ctx, "ctx");
    failures += check_pinned_gate(ctx);
    failures += check_memory_exhausted(ctx);
    failures += check_upload_snapshot(ctx);
    int* device_value = nullptr;
    int* host_value   = nullptr;
    CUDA_CHECK(cudaMalloc(&device_value, sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&host_value, sizeof(int)));
    *host_value = 0;
    CUDA_CHECK(cudaMemsetAsync(device_value, 0x5a, sizeof(int), ctx.stream));
    CUDA_CHECK(
        cudaMemcpyAsync(host_value, device_value, sizeof(int), cudaMemcpyDeviceToHost, ctx.stream));
    ctx.synchronize();
    const bool transfer_complete = *host_value == 0x5a5a5a5a;
    CUDA_CHECK(cudaFreeHost(host_value));
    CUDA_CHECK(cudaFree(device_value));
    if (!transfer_complete) {
        return fail("stream synchronization returned before transfer completed");
    }

    const cudaStream_t original_stream = ctx.stream;
    infernix::DeviceContext moved(std::move(ctx));
    if (ctx.stream != nullptr || ctx.transfer_stream != nullptr) {
        ++failures;
        std::cerr << "move construction did not null source streams\n";
    }
    if (moved.stream != original_stream) {
        ++failures;
        std::cerr << "move construction did not transfer compute stream\n";
    }
    failures += check_context(moved, "moved");

    failures += expect_throws_device(count);

    infernix::CudaEventTimer timer(moved);
    timer.start();
    moved.synchronize();
    const float elapsed_ms = timer.stop_ms();
    if (elapsed_ms < 0.0f) {
        ++failures;
        std::cerr << "timer elapsed time was negative\n";
    }

    // try_elapsed_ms never waits: empty while the stop is queued behind host work, a value after.
    infernix::CudaEventTimer pending(moved);
    pending.start();
    CUDA_CHECK(cudaLaunchHostFunc(
        moved.stream,
        [](void*) { std::this_thread::sleep_for(std::chrono::milliseconds(200)); }, nullptr));
    pending.record_stop();
    if (pending.try_elapsed_ms()) {
        ++failures;
        std::cerr << "try_elapsed_ms returned a time before the stop completed\n";
    }
    moved.synchronize();
    const auto settled = pending.try_elapsed_ms();
    if (!settled || *settled < 100.0f) {
        ++failures;
        std::cerr << "try_elapsed_ms did not report the completed stop\n";
    }

    return failures == 0 ? 0 : fail("device test failed");
}
