#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer {

void cuda_check(cudaError_t err, const char* expr, const char* file, int line);

#define CUDA_CHECK(expr) ::ninfer::cuda_check((expr), #expr, __FILE__, __LINE__)

// Non-owning execution facts passed to Ops whose launch policy depends on physical device
// capacity. DeviceContext remains the owner and authoritative source of both values.
struct DeviceExecutionView {
    cudaStream_t stream               = nullptr;
    std::int32_t multiprocessor_count = 0;

    [[nodiscard]] constexpr DeviceExecutionView
    on_stream(cudaStream_t target_stream) const noexcept {
        return {target_stream, multiprocessor_count};
    }
};

struct DeviceContext {
    int device                   = 0;
    cudaStream_t stream          = nullptr;
    cudaStream_t transfer_stream = nullptr;
    cudaDeviceProp props{};

    explicit DeviceContext(int device_id = 0);
    ~DeviceContext();

    DeviceContext(const DeviceContext&)            = delete;
    DeviceContext& operator=(const DeviceContext&) = delete;
    DeviceContext(DeviceContext&& other) noexcept;
    DeviceContext& operator=(DeviceContext&& other) noexcept;

    void bind_to_current_thread() const;
    void bind_to_current_thread_noexcept() const noexcept;
    int compute_capability() const noexcept;
    int multiprocessor_count() const noexcept;
    DeviceExecutionView execution_view() const noexcept;
    std::size_t total_vram() const noexcept;
    // Device memory currently free on this device (cudaMemGetInfo).
    std::size_t free_bytes() const;
    const char* sync_mode() const;
    // Waits for the compute stream only. Work on transfer_stream is ordered separately: callers
    // that read what a transfer writes wait on its event or synchronize that stream themselves.
    void synchronize() const;
    // Submits queued work without waiting for it. On a batched driver model (WDDM) a launch can
    // otherwise sit in the command buffer until the next blocking call or launch.
    void flush() const;
};

// Copies `bytes` from pinned host memory to device memory in stream order, read by the SMs
// through the host mapping (pinned allocations are device-accessible under UVA) rather than by a
// copy engine. Small per-round inputs then never queue behind bulk host-to-device transfers of
// other streams (for example expert promotions), which share the copy engine in FIFO order.
void upload_pinned(void* device_dst, const void* pinned_src, std::size_t bytes, cudaStream_t stream);

// upload_pinned behind a host gate: the copy waits in stream order until the pinned word
// `pinned_ready` equals the device word `device_expected` (staged earlier in stream order), so the
// host may write `pinned_src` after the launch and then release it with publish_pinned_word. The
// host's values must strictly increase, so a word left from an earlier round never matches.
// `bytes` and both buffers are 16-byte multiples/aligned. When `wait_stats` (device U64) is set,
// a wait that found the word unpublished adds its nanoseconds and 1 to the two words at
// `wait_stats + 2 * *wait_row` (row 0 when `wait_row` is null). A wait longer than
// kPinnedGateTimeoutNs traps (the context is lost): it bounds a producer bug, not a slow read.
inline constexpr std::uint64_t kPinnedGateTimeoutNs = 120ULL * 1000 * 1000 * 1000;
void upload_pinned_when(void* device_dst, const void* pinned_src, std::size_t bytes,
                        const std::uint32_t* pinned_ready, const std::uint32_t* device_expected,
                        std::uint64_t* wait_stats, const std::int32_t* wait_row, cudaStream_t stream);

// Releases the gates waiting for `value`: orders every earlier store to pinned memory (including
// non-temporal ones) before the word's store.
void publish_pinned_word(std::uint32_t* pinned_ready, std::uint32_t value) noexcept;

class CudaEventTimer {
public:
    explicit CudaEventTimer(const DeviceContext& ctx);
    CudaEventTimer(const DeviceContext& ctx, cudaStream_t stream);
    ~CudaEventTimer();

    CudaEventTimer(const CudaEventTimer&)            = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;
    CudaEventTimer(CudaEventTimer&& other) noexcept;
    CudaEventTimer& operator=(CudaEventTimer&& other) noexcept;

    void start();
    void record_stop();
    [[nodiscard]] float elapsed_ms() const;
    float stop_ms();

private:
    cudaStream_t stream_ = nullptr;
    cudaEvent_t start_   = nullptr;
    cudaEvent_t stop_    = nullptr;
};

// Reusable non-timing event for worker-driven asynchronous control transactions. The owning
// component records it after enqueueing one transfer batch and polls it from later boundaries.
class CudaCompletionEvent {
public:
    explicit CudaCompletionEvent(const DeviceContext& ctx);
    ~CudaCompletionEvent();

    CudaCompletionEvent(const CudaCompletionEvent&)            = delete;
    CudaCompletionEvent& operator=(const CudaCompletionEvent&) = delete;
    CudaCompletionEvent(CudaCompletionEvent&& other) noexcept;
    CudaCompletionEvent& operator=(CudaCompletionEvent&& other) noexcept;

    void record(cudaStream_t stream);
    void wait(cudaStream_t stream) const;
    [[nodiscard]] bool ready() const;
    void synchronize() const;

private:
    int device_        = 0;
    cudaEvent_t event_ = nullptr;
};

} // namespace ninfer
