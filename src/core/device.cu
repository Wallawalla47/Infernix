#include "core/device.h"

#include "core/cuda_diagnostics.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace infernix {
namespace {

struct SyncSchedule {
    std::string_view name;
    unsigned int flags;
};

constexpr SyncSchedule kSyncSchedules[] = {
    {"spin", cudaDeviceScheduleSpin},
    {"blocking", cudaDeviceScheduleBlockingSync},
    {"yield", cudaDeviceScheduleYield},
    {"auto", cudaDeviceScheduleAuto},
};

unsigned int sync_schedule_from_environment() {
    const char* value = std::getenv("INFERNIX_CUDA_SYNC");
    if (value == nullptr) { return cudaDeviceScheduleSpin; }
    for (const auto& schedule : kSyncSchedules) {
        if (schedule.name == value) { return schedule.flags; }
    }
    throw std::invalid_argument("INFERNIX_CUDA_SYNC must be spin, blocking, yield, or auto");
}

std::string cuda_error_message(const char* prefix, cudaError_t err) {
    return std::string(prefix) + ": " + cudaGetErrorName(err) + ": " + cudaGetErrorString(err);
}

std::mutex& cuda_diagnostic_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::shared_ptr<const DiagnosticObserver>& cuda_diagnostic_observer() {
    static std::shared_ptr<const DiagnosticObserver> observer;
    return observer;
}

void publish_cuda_diagnostic(DiagnosticLevel level, const char* text) noexcept {
    std::shared_ptr<const DiagnosticObserver> observer;
    {
        std::lock_guard lock(cuda_diagnostic_mutex());
        observer = cuda_diagnostic_observer();
    }
    if (!observer || !observer->callback) { return; }
    try {
        observer->callback(Diagnostic{.level = level, .message = text});
    } catch (...) {}
}

void destroy_stream(cudaStream_t& stream) noexcept {
    if (stream != nullptr) {
        report_cuda_cleanup_failure("cudaStreamDestroy", cudaStreamDestroy(stream));
        stream = nullptr;
    }
}

void destroy_event(cudaEvent_t& event) noexcept {
    if (event != nullptr) {
        report_cuda_cleanup_failure("cudaEventDestroy", cudaEventDestroy(event));
        event = nullptr;
    }
}

} // namespace

DiagnosticObserver set_cuda_diagnostic_observer(DiagnosticObserver observer) {
    std::shared_ptr<const DiagnosticObserver> installed;
    if (observer.callback) {
        installed = std::make_shared<const DiagnosticObserver>(std::move(observer));
    }
    std::shared_ptr<const DiagnosticObserver> previous;
    {
        std::lock_guard lock(cuda_diagnostic_mutex());
        previous = std::exchange(cuda_diagnostic_observer(), installed);
    }
    return previous ? *previous : DiagnosticObserver{};
}

void report_cuda_check_failure(const char* expr, const char* file, int line,
                               cudaError_t err) noexcept {
    const char* name        = cudaGetErrorName(err);
    const char* description = cudaGetErrorString(err);
    std::fprintf(stderr, "%s:%d: CUDA_CHECK(%s) failed: %s: %s\n", file, line, expr, name,
                 description);
    char record[2048];
    std::snprintf(record, sizeof(record), "%s:%d: CUDA_CHECK(%s) failed: %s: %s", file, line, expr,
                  name, description);
    publish_cuda_diagnostic(DiagnosticLevel::Error, record);
}

void report_cuda_cleanup_failure(const char* op, cudaError_t err) noexcept {
    if (err == cudaSuccess) { return; }
    const char* name        = cudaGetErrorName(err);
    const char* description = cudaGetErrorString(err);
    std::fprintf(stderr, "CUDA cleanup failed during %s: %s: %s\n", op, name, description);
    char record[1024];
    std::snprintf(record, sizeof(record), "CUDA cleanup failed during %s: %s: %s", op, name,
                  description);
    publish_cuda_diagnostic(DiagnosticLevel::Warning, record);
}

namespace {

// testing::arm_memory_exhausted_here: this thread's countdown (0: disarmed).
thread_local std::uint64_t t_memory_fault_countdown = 0;
std::atomic<std::uint64_t> g_memory_fault_injections{0};

[[noreturn]] void throw_memory_exhausted(const char* expr, const char* file, int line, bool injected) {
    char what[640];
    std::snprintf(what, sizeof(what), "out of device memory (%s:%d: %s%s)", file, line, expr,
                  injected ? ", injected by a test" : "");
    throw DeviceMemoryExhausted(what);
}

} // namespace

void cuda_check(cudaError_t err, const char* expr, const char* file, int line) {
    if (err == cudaSuccess) {
        if (t_memory_fault_countdown == 0 || --t_memory_fault_countdown != 0) { return; }
        g_memory_fault_injections.fetch_add(1, std::memory_order_relaxed);
        throw_memory_exhausted(expr, file, line, true);
    }
    report_cuda_check_failure(expr, file, line, err);
    if (err == cudaErrorMemoryAllocation) {
        // Not sticky: the thread's last error is cleared so that the next launch check does not read it.
        (void)cudaGetLastError();
        throw_memory_exhausted(expr, file, line, false);
    }
    std::abort();
}

bool settle_after_memory_exhausted() noexcept {
    // Two passes: work enqueued before the failure may itself report the shortage once.
    for (int pass = 0; pass < 2; ++pass) {
        const cudaError_t err = cudaDeviceSynchronize();
        if (err == cudaSuccess) {
            (void)cudaGetLastError();
            return true;
        }
        report_cuda_check_failure("cudaDeviceSynchronize() after an out-of-memory error", __FILE__, __LINE__, err);
        if (err != cudaErrorMemoryAllocation) { return false; }
        (void)cudaGetLastError();
    }
    return false;
}

namespace testing {
void arm_memory_exhausted_here(std::uint64_t after) noexcept { t_memory_fault_countdown = after; }
std::uint64_t memory_exhausted_injections() noexcept {
    return g_memory_fault_injections.load(std::memory_order_relaxed);
}
} // namespace testing

// Allocates upload_pinned's staging ring (defined below, with the ring).
void prepare_upload_staging();

DeviceContext::DeviceContext(int device_id) : device(device_id) {
    const unsigned int sync_flags = sync_schedule_from_environment();
    int count       = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaGetDeviceCount failed", err));
    }
    if (count <= 0) { throw std::runtime_error("no CUDA devices available"); }
    if (device_id < 0 || device_id >= count) { throw std::runtime_error("invalid CUDA device id"); }

    bind_to_current_thread();

    err = cudaSetDeviceFlags(sync_flags);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaSetDeviceFlags failed", err));
    }

    err = cudaGetDeviceProperties(&props, device_id);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaGetDeviceProperties failed", err));
    }

    cudaStream_t compute = nullptr;
    cudaStream_t load    = nullptr;
    err                  = cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            cuda_error_message("cudaStreamCreateWithFlags(stream) failed", err));
    }

    err = cudaStreamCreateWithFlags(&load, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        destroy_stream(compute);
        throw std::runtime_error(
            cuda_error_message("cudaStreamCreateWithFlags(transfer_stream) failed", err));
    }

    stream          = compute;
    transfer_stream = load;
    prepare_upload_staging(); // while the device is idle (upload_pinned's staging ring)
}

DeviceContext::~DeviceContext() {
    if (stream != nullptr || transfer_stream != nullptr) { bind_to_current_thread_noexcept(); }
    destroy_stream(transfer_stream);
    destroy_stream(stream);
}

DeviceContext::DeviceContext(DeviceContext&& other) noexcept
    : device(other.device), stream(other.stream), transfer_stream(other.transfer_stream),
      props(other.props) {
    other.stream          = nullptr;
    other.transfer_stream = nullptr;
}

DeviceContext& DeviceContext::operator=(DeviceContext&& other) noexcept {
    if (this == &other) { return *this; }

    if (stream != nullptr || transfer_stream != nullptr) { bind_to_current_thread_noexcept(); }
    destroy_stream(transfer_stream);
    destroy_stream(stream);

    device          = other.device;
    props           = other.props;
    stream          = other.stream;
    transfer_stream = other.transfer_stream;

    other.stream          = nullptr;
    other.transfer_stream = nullptr;
    return *this;
}

void DeviceContext::bind_to_current_thread() const {
    const cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaSetDevice failed", err));
    }
}

void DeviceContext::bind_to_current_thread_noexcept() const noexcept {
    report_cuda_cleanup_failure("cudaSetDevice", cudaSetDevice(device));
}

int DeviceContext::compute_capability() const noexcept { return props.major * 10 + props.minor; }

int DeviceContext::multiprocessor_count() const noexcept { return props.multiProcessorCount; }

DeviceExecutionView DeviceContext::execution_view() const noexcept {
    return {.stream = stream, .multiprocessor_count = multiprocessor_count()};
}

std::size_t DeviceContext::total_vram() const noexcept { return props.totalGlobalMem; }

const char* DeviceContext::sync_mode() const {
    bind_to_current_thread();
    unsigned int flags    = 0;
    const cudaError_t err = cudaGetDeviceFlags(&flags);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaGetDeviceFlags failed", err));
    }
    for (const auto& schedule : kSyncSchedules) {
        if ((flags & cudaDeviceScheduleMask) == schedule.flags) { return schedule.name.data(); }
    }
    throw std::runtime_error("unknown CUDA synchronization schedule");
}

void DeviceContext::synchronize() const { CUDA_CHECK(cudaStreamSynchronize(stream)); }

std::size_t DeviceContext::free_bytes() const {
    std::size_t free  = 0;
    std::size_t total = 0;
    CUDA_CHECK(cudaMemGetInfo(&free, &total));
    return free;
}

void DeviceContext::flush() const {
    const cudaError_t status = cudaStreamQuery(stream);
    if (status == cudaErrorNotReady) {
        // Pending work is the expected answer; do not leave it as the thread's last error.
        (void)cudaGetLastError();
        return;
    }
    CUDA_CHECK(status);
}

CudaEventTimer::CudaEventTimer(const DeviceContext& ctx) : CudaEventTimer(ctx, ctx.stream) {}

CudaEventTimer::CudaEventTimer(const DeviceContext& ctx, cudaStream_t stream) : stream_(stream) {
    if (stream == nullptr) { throw std::invalid_argument("CUDA timer stream is null"); }
    ctx.bind_to_current_thread();

    cudaEvent_t start = nullptr;
    cudaEvent_t stop  = nullptr;
    cudaError_t err   = cudaEventCreate(&start);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaEventCreate(start) failed", err));
    }

    err = cudaEventCreate(&stop);
    if (err != cudaSuccess) {
        destroy_event(start);
        throw std::runtime_error(cuda_error_message("cudaEventCreate(stop) failed", err));
    }

    start_ = start;
    stop_  = stop;
}

CudaEventTimer::~CudaEventTimer() {
    destroy_event(stop_);
    destroy_event(start_);
}

CudaEventTimer::CudaEventTimer(CudaEventTimer&& other) noexcept
    : stream_(other.stream_), start_(other.start_), stop_(other.stop_) {
    other.stream_ = nullptr;
    other.start_  = nullptr;
    other.stop_   = nullptr;
}

CudaEventTimer& CudaEventTimer::operator=(CudaEventTimer&& other) noexcept {
    if (this == &other) { return *this; }

    destroy_event(stop_);
    destroy_event(start_);

    stream_ = other.stream_;
    start_  = other.start_;
    stop_   = other.stop_;

    other.stream_ = nullptr;
    other.start_  = nullptr;
    other.stop_   = nullptr;
    return *this;
}

void CudaEventTimer::start() { CUDA_CHECK(cudaEventRecord(start_, stream_)); }

void CudaEventTimer::record_stop() { CUDA_CHECK(cudaEventRecord(stop_, stream_)); }

float CudaEventTimer::elapsed_ms() const {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
}

std::optional<float> CudaEventTimer::try_elapsed_ms() const {
    const cudaError_t status = cudaEventQuery(stop_);
    if (status == cudaErrorNotReady) { return std::nullopt; }
    CUDA_CHECK(status);
    return elapsed_ms();
}

float CudaEventTimer::stop_ms() {
    record_stop();
    CUDA_CHECK(cudaEventSynchronize(stop_));
    return elapsed_ms();
}

CudaCompletionEvent::CudaCompletionEvent(const DeviceContext& ctx) : device_(ctx.device) {
    ctx.bind_to_current_thread();
    const cudaError_t err = cudaEventCreateWithFlags(&event_, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaEventCreateWithFlags failed", err));
    }
}

CudaCompletionEvent::~CudaCompletionEvent() { destroy_event(event_); }

CudaCompletionEvent::CudaCompletionEvent(CudaCompletionEvent&& other) noexcept
    : device_(other.device_), event_(std::exchange(other.event_, nullptr)) {}

CudaCompletionEvent& CudaCompletionEvent::operator=(CudaCompletionEvent&& other) noexcept {
    if (this == &other) { return *this; }
    destroy_event(event_);
    device_ = other.device_;
    event_  = std::exchange(other.event_, nullptr);
    return *this;
}

void CudaCompletionEvent::record(cudaStream_t stream) {
    if (event_ == nullptr || stream == nullptr) {
        throw std::logic_error("CUDA completion event is not recordable");
    }
    CUDA_CHECK(cudaEventRecord(event_, stream));
}

void CudaCompletionEvent::wait(cudaStream_t stream) const {
    if (event_ == nullptr || stream == nullptr) {
        throw std::logic_error("CUDA completion event is not waitable");
    }
    CUDA_CHECK(cudaStreamWaitEvent(stream, event_, 0));
}

void CudaCompletionEvent::record_external(cudaStream_t stream) {
    if (event_ == nullptr || stream == nullptr) {
        throw std::logic_error("CUDA completion event is not recordable");
    }
    CUDA_CHECK(cudaEventRecordWithFlags(event_, stream, cudaEventRecordExternal));
}

bool CudaCompletionEvent::ready() const {
    if (event_ == nullptr) { throw std::logic_error("CUDA completion event is empty"); }
    const cudaError_t status = cudaEventQuery(event_);
    if (status == cudaSuccess) { return true; }
    if (status == cudaErrorNotReady) { return false; }
    CUDA_CHECK(status);
    return false;
}

void CudaCompletionEvent::synchronize() const {
    if (event_ == nullptr) { throw std::logic_error("CUDA completion event is empty"); }
    CUDA_CHECK(cudaEventSynchronize(event_));
}


namespace {

// .cv loads: host bytes at these addresses were rewritten since an earlier upload read them (the staging ring wraps,
// and a captured node rereads its source at each replay), so no cache line of an earlier read may serve this one.
__global__ void upload_pinned_kernel(void* __restrict__ dst, const void* __restrict__ src, std::size_t bytes) {
    const std::size_t thread = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    const bool vectors = (reinterpret_cast<std::uintptr_t>(dst) | reinterpret_cast<std::uintptr_t>(src)) % 16 == 0;
    const std::size_t head = vectors ? bytes / 16 : 0;
    for (std::size_t i = thread; i < head; i += stride) {
        static_cast<uint4*>(dst)[i] = __ldcv(static_cast<const uint4*>(src) + i);
    }
    for (std::size_t i = head * 16 + thread; i < bytes; i += stride) {
        static_cast<unsigned char*>(dst)[i] = __ldcv(static_cast<const unsigned char*>(src) + i);
    }
}

// upload_pinned's staging (device.h): each eager upload's bytes are copied here at the call and the kernel reads the
// copy, so a caller may rewrite its buffer as soon as upload_pinned returns, whether or not the stream has reached the
// upload. A FIFO ring of pinned memory, allocated once by the first DeviceContext (allocating pinned memory waits for
// the device, which work spinning on a host word this thread has yet to publish never lets go of); a span is reused
// only once the event recorded after its kernel has completed, which it normally has long since: the host waits only
// when it runs a whole ring of uploads ahead of the device. An upload larger than half the ring, or one made before
// any DeviceContext, is copied from pageable memory instead (cudaMemcpyAsync stages a pageable source before it
// returns), on the copy engine.
class UploadStaging {
public:
    ~UploadStaging() {
        // Process exit: the device may already be gone, so nothing is checked.
        for (const Span& span : live_) { (void)cudaEventDestroy(span.done); }
        for (cudaEvent_t event : spare_) { (void)cudaEventDestroy(event); }
        if (ring_ != nullptr) { (void)cudaFreeHost(ring_); }
    }

    // Allocates the ring (DeviceContext's constructor: the device is idle then).
    void prepare() {
        const std::lock_guard<std::mutex> lock(mutex_);
        if (ring_ != nullptr) { return; }
        void* ring = nullptr;
        CUDA_CHECK(cudaHostAlloc(&ring, kRing, cudaHostAllocPortable | cudaHostAllocMapped));
        ring_ = static_cast<std::byte*>(ring);
    }

    // Copies `bytes` of `src` into the ring, launches the copy to `dst` on `stream` from there and records its event.
    void upload(void* dst, const void* src, std::size_t bytes, cudaStream_t stream, unsigned int blocks, unsigned int threads) {
        const std::lock_guard<std::mutex> lock(mutex_);
        const std::size_t span = (bytes + kAlignment - 1) / kAlignment * kAlignment;
        if (ring_ == nullptr || span > kRing / 2) {
            const std::vector<std::byte> copy(static_cast<const std::byte*>(src), static_cast<const std::byte*>(src) + bytes);
            CUDA_CHECK(cudaMemcpyAsync(dst, copy.data(), bytes, cudaMemcpyHostToDevice, stream));
            return;
        }
        // Relaxed: the event queries and waits are not stream work, so an upload made while this thread captures a graph
        // on another stream neither lands in it nor invalidates it (as tma_descriptor_staging.cuh).
        cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
        CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&mode));
        try {
            reserve(span);
        } catch (...) {
            (void)cudaThreadExchangeStreamCaptureMode(&mode);
            throw;
        }
        CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&mode));
        std::byte* staged = ring_ + head_;
        std::memcpy(staged, src, bytes);
        upload_pinned_kernel<<<blocks, threads, 0, stream>>>(dst, staged, bytes);
        CUDA_CHECK(cudaGetLastError());
        cudaEvent_t done = take_event();
        CUDA_CHECK(cudaEventRecord(done, stream));
        live_.push_back(Span{head_, head_ + span, done});
        head_ += span;
    }

private:
    static constexpr std::size_t kAlignment = 256;
    static constexpr std::size_t kRing      = kUploadStagingRingBytes;
    struct Span {
        std::size_t begin = 0, end = 0;
        cudaEvent_t done = nullptr;
    };

    cudaEvent_t take_event() {
        if (!spare_.empty()) {
            cudaEvent_t event = spare_.back();
            spare_.pop_back();
            return event;
        }
        cudaEvent_t event = nullptr;
        CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        return event;
    }
    void retire_front(bool wait) {
        if (wait) { CUDA_CHECK(cudaEventSynchronize(live_.front().done)); }
        spare_.push_back(live_.front().done);
        live_.pop_front();
    }
    // Makes [head_, head_ + span) free (span <= kRing / 2): wraps to the ring's start when the span does not fit before
    // its end, and waits for the oldest uploads whose spans it overlaps.
    void reserve(std::size_t span) {
        while (!live_.empty()) { // spans whose uploads have run
            const cudaError_t status = cudaEventQuery(live_.front().done);
            if (status == cudaErrorNotReady) { break; }
            CUDA_CHECK(status);
            retire_front(false);
        }
        if (head_ + span > kRing) { head_ = 0; }
        const auto overlaps = [&](const Span& s) { return s.begin < head_ + span && head_ < s.end; };
        while (std::any_of(live_.begin(), live_.end(), overlaps)) { retire_front(true); }
    }

    std::mutex mutex_;
    std::byte* ring_  = nullptr;
    std::size_t head_ = 0;
    std::deque<Span> live_;
    std::vector<cudaEvent_t> spare_;
};

UploadStaging& upload_staging() {
    static UploadStaging staging;
    return staging;
}

} // namespace

void prepare_upload_staging() { upload_staging().prepare(); }

namespace {

constexpr std::size_t kGateCtaBytes = 32 * 1024;
constexpr int kGateMaxCtas         = 16;
constexpr int kGateThreads         = 256;

// Each CTA's thread 0 polls the host word, so no CTA depends on another being resident. .cv loads:
// the source is rewritten for every round, so no cache line of an earlier round may be used.
__global__ void __launch_bounds__(kGateThreads)
    upload_pinned_when_kernel(uint4* __restrict__ dst, const uint4* __restrict__ src, std::size_t vectors,
                              const std::uint32_t* ready, const std::uint32_t* expected, std::uint64_t* stats,
                              const std::int32_t* row, DeviceFault fault, std::uint64_t timeout_ns) {
    __shared__ bool timed_out;
    if (threadIdx.x == 0) {
        timed_out                = false;
        const std::uint32_t want = *expected;
        const auto* word         = reinterpret_cast<const volatile std::uint32_t*>(ready);
        if (*word != want) {
            std::uint64_t start, now;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
            do {
                __nanosleep(200);
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
                if (now - start > timeout_ns) {
                    if (fault.word != nullptr) { *reinterpret_cast<volatile std::uint32_t*>(fault.word) = fault.value; }
                    timed_out = true;
                    break;
                }
            } while (*word != want);
            if (!timed_out && stats != nullptr && blockIdx.x == 0) {
                std::uint64_t* at = stats + 2 * (row != nullptr ? *row : 0);
                atomicAdd(reinterpret_cast<unsigned long long*>(at), static_cast<unsigned long long>(now - start));
                atomicAdd(reinterpret_cast<unsigned long long*>(at + 1), 1ULL);
            }
        }
        __threadfence_system();
    }
    __syncthreads();
    if (timed_out) { return; }
    constexpr int kBatch     = 8;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t base = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; base < vectors;
         base += kBatch * stride) {
        uint4 loaded[kBatch];
#pragma unroll
        for (int b = 0; b < kBatch; ++b) {
            const std::size_t i = base + b * stride;
            if (i < vectors) { loaded[b] = __ldcv(src + i); }
        }
#pragma unroll
        for (int b = 0; b < kBatch; ++b) {
            const std::size_t i = base + b * stride;
            if (i < vectors) { dst[i] = loaded[b]; }
        }
    }
}

} // namespace

void upload_pinned_when(void* device_dst, const void* pinned_src, std::size_t bytes, const std::uint32_t* pinned_ready,
                        const std::uint32_t* device_expected, std::uint64_t* wait_stats, const std::int32_t* wait_row,
                        DeviceFault fault, cudaStream_t stream, std::uint64_t timeout_ns) {
    if (bytes == 0) { return; }
    if (bytes % 16 != 0 || (reinterpret_cast<std::uintptr_t>(device_dst) | reinterpret_cast<std::uintptr_t>(pinned_src)) % 16 != 0) {
        throw std::invalid_argument("upload_pinned_when: bytes and buffers must be 16-byte multiples");
    }
    const auto ctas = static_cast<unsigned int>(
        std::min<std::size_t>((bytes + kGateCtaBytes - 1) / kGateCtaBytes, kGateMaxCtas));
    upload_pinned_when_kernel<<<ctas, kGateThreads, 0, stream>>>(
        static_cast<uint4*>(device_dst), static_cast<const uint4*>(pinned_src), bytes / 16, pinned_ready,
        device_expected, wait_stats, wait_row, fault, timeout_ns);
    CUDA_CHECK(cudaGetLastError());
}

void publish_pinned_word(std::uint32_t* pinned_ready, std::uint32_t value) noexcept {
#if defined(_M_X64) || defined(__x86_64__)
    _mm_sfence();
#endif
    std::atomic_thread_fence(std::memory_order_release);
    *static_cast<volatile std::uint32_t*>(pinned_ready) = value;
}

void upload_pinned(void* device_dst, const void* pinned_src, std::size_t bytes, cudaStream_t stream) {
    if (bytes == 0) { return; }
    static const bool unified = [] {
        int device = 0, value = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&value, cudaDevAttrUnifiedAddressing, device));
        return value != 0;
    }();
    if (!unified) { throw std::runtime_error("upload_pinned needs unified addressing"); }
    constexpr int kThreads = 256;
    // One 16-byte load per thread: every load of a call is in flight at once.
    const std::size_t blocks = std::min<std::size_t>((bytes / 16 + kThreads) / kThreads + 1, 64);
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
    if (capture == cudaStreamCaptureStatusNone) {
        upload_staging().upload(device_dst, pinned_src, bytes, stream, static_cast<unsigned int>(blocks), kThreads);
    } else {
        // A captured node reads the caller's buffer at each replay (device.h).
        upload_pinned_kernel<<<static_cast<unsigned int>(blocks), kThreads, 0, stream>>>(device_dst, pinned_src, bytes);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace infernix
