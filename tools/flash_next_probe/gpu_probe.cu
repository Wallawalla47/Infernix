// Flash-Next M0 GPU and PCIe probe (docs/maintainer/qwen3_8-flash-next-design.md §8, §9, §14.2
// Stage 1, §19 M0). Measures the machine facts the decode pipeline is budgeted on:
//
//   inventory      device attributes the design depends on (copy engines, mapped memory, L2, SMs)
//   copy           H2D and D2H from pinned memory versus size, single-copy latency and streaming rate
//   duplex         H2D throughput with and without a concurrent D2H stream
//   register       cudaHostRegister time per GiB, 4 KiB pages versus transparent huge pages
//   contention     host DRAM read bandwidth with H2D DMA idle and saturated, and the DMA rate meanwhile
//   pdl            per-kernel cost of a 507-kernel chain in a CUDA Graph, plain versus PDL-chained
//   mailbox        GPU -> host -> GPU round trip through mapped memory (CPU-served expert handshake)
//   submit         host enqueue of k expert-record copies -> device-written completion flag visible
//
// Build (from the repository root; links only the CUDA runtime):
//   nvcc -O3 -std=c++17 -arch=sm_120a tools/flash_next_probe/gpu_probe.cu -o build/flash_next_gpu_probe
//   ./build/flash_next_gpu_probe [section ...]        (default: every section)
//
// Run on a quiescent machine. Also record, under load, the negotiated link:
//   nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current,clocks.mem --format=csv

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <sys/mman.h>

#define CUDA_CHECK(expr)                                                                           \
    do {                                                                                           \
        const cudaError_t error__ = (expr);                                                        \
        if (error__ != cudaSuccess) {                                                              \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,                  \
                         cudaGetErrorString(error__));                                             \
            std::exit(EXIT_FAILURE);                                                               \
        }                                                                                          \
    } while (false)

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t kRecordBytes = 2764800; // one nvfp4_expert_rg16_v1 record
constexpr int kRepetitions         = 5;

double seconds_since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

__device__ __forceinline__ std::uint64_t global_ns() {
    std::uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// ------------------------------------------------------------------------------------ inventory

void inventory() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, device));
    auto attr = [&](cudaDeviceAttr a) {
        int v = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&v, a, device));
        return v;
    };
    int driver = 0, runtime = 0;
    CUDA_CHECK(cudaDriverGetVersion(&driver));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    std::printf("== inventory\n");
    std::printf("device            %s (sm_%d%d), PCI %04x:%02x:%02x\n", p.name, p.major, p.minor, p.pciDomainID,
                p.pciBusID, p.pciDeviceID);
    std::printf("driver / runtime  %d / %d\n", driver, runtime);
    std::printf("SMs               %d, L2 %d KiB, persisting L2 max %d KiB\n", p.multiProcessorCount, p.l2CacheSize >> 10,
                p.persistingL2CacheMaxSize >> 10);
    std::printf("global memory     %.0f MiB, bus %d bits, memory clock %d kHz (attribute)\n",
                static_cast<double>(p.totalGlobalMem) / (1 << 20), attr(cudaDevAttrGlobalMemoryBusWidth),
                attr(cudaDevAttrMemoryClockRate));
    std::printf("asyncEngineCount  %d   (design §8.6 assumes H2D shares one engine on GeForce)\n",
                attr(cudaDevAttrAsyncEngineCount));
    std::printf("mapped host mem   canMap %d, hostRegister %d, readOnlyRegister %d, pageableViaHostTables %d\n",
                attr(cudaDevAttrCanMapHostMemory), attr(cudaDevAttrHostRegisterSupported),
                attr(cudaDevAttrHostRegisterReadOnlySupported),
                attr(cudaDevAttrPageableMemoryAccessUsesHostPageTables));
    std::printf("shared mem/SM     %d KiB, max per block opt-in %d KiB\n", attr(cudaDevAttrMaxSharedMemoryPerMultiprocessor) >> 10,
                attr(cudaDevAttrMaxSharedMemoryPerBlockOptin) >> 10);
    std::printf("stream priorities ");
    int lo = 0, hi = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
    std::printf("[%d, %d]\n", lo, hi);
}

// ------------------------------------------------------------------------------------ copies

void copy_sizes() {
    std::printf("\n== copy: pinned host <-> device versus size (median of %d)\n", kRepetitions);
    const std::size_t max_bytes = std::size_t{64} << 20;
    void *host = nullptr, *dev = nullptr;
    CUDA_CHECK(cudaHostAlloc(&host, max_bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaMalloc(&dev, max_bytes));
    std::memset(host, 1, max_bytes);
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    std::printf("%12s %10s %14s %14s %14s\n", "bytes", "dir", "1 copy us", "stream GB/s", "copies/batch");
    std::vector<std::size_t> sizes{64 << 10, 256 << 10, 1 << 20, kRecordBytes, 4 << 20, 16 << 20};
    for (std::size_t bytes : sizes) {
        for (const bool h2d : {true, false}) {
            const cudaMemcpyKind kind = h2d ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
            void* dst                 = h2d ? dev : host;
            const void* src           = h2d ? host : dev;
            std::vector<double> single, stream;
            const int batch = static_cast<int>(std::max<std::size_t>(4, (std::size_t{256} << 20) / bytes));
            for (int r = 0; r < kRepetitions; ++r) {
                CUDA_CHECK(cudaEventRecord(a, s));
                CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, kind, s));
                CUDA_CHECK(cudaEventRecord(b, s));
                CUDA_CHECK(cudaEventSynchronize(b));
                float ms = 0;
                CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
                single.push_back(ms * 1e3);
                const std::size_t slots = max_bytes / bytes;
                CUDA_CHECK(cudaEventRecord(a, s));
                for (int i = 0; i < batch; ++i) {
                    const std::size_t off = (static_cast<std::size_t>(i) % slots) * bytes;
                    CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(dst) + off, static_cast<const char*>(src) + off, bytes,
                                               kind, s));
                }
                CUDA_CHECK(cudaEventRecord(b, s));
                CUDA_CHECK(cudaEventSynchronize(b));
                CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
                stream.push_back(static_cast<double>(bytes) * batch / (ms * 1e-3) / 1e9);
            }
            std::printf("%12zu %10s %14.1f %14.2f %14d\n", bytes, h2d ? "H2D" : "D2H", median(single), median(stream), batch);
        }
    }
    CUDA_CHECK(cudaEventDestroy(a));
    CUDA_CHECK(cudaEventDestroy(b));
    CUDA_CHECK(cudaStreamDestroy(s));
    CUDA_CHECK(cudaFree(dev));
    CUDA_CHECK(cudaFreeHost(host));
}

// Streams `count` copies of `bytes` on `s` and returns the GB/s seen by events on that stream.
double timed_stream(cudaStream_t s, void* dst, const void* src, std::size_t bytes, int count, cudaMemcpyKind kind) {
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    CUDA_CHECK(cudaEventRecord(a, s));
    for (int i = 0; i < count; ++i) { CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, kind, s)); }
    CUDA_CHECK(cudaEventRecord(b, s));
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    CUDA_CHECK(cudaEventDestroy(a));
    CUDA_CHECK(cudaEventDestroy(b));
    return static_cast<double>(bytes) * count / (ms * 1e-3) / 1e9;
}

void duplex() {
    std::printf("\n== duplex: H2D with and without concurrent D2H (16 MiB copies)\n");
    const std::size_t bytes = std::size_t{16} << 20;
    void *h_up = nullptr, *h_down = nullptr, *d_up = nullptr, *d_down = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h_up, bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_down, bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaMalloc(&d_up, bytes));
    CUDA_CHECK(cudaMalloc(&d_down, bytes));
    cudaStream_t up, down;
    CUDA_CHECK(cudaStreamCreateWithFlags(&up, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&down, cudaStreamNonBlocking));
    const double alone = timed_stream(up, d_up, h_up, bytes, 64, cudaMemcpyHostToDevice);
    double with_down   = 0;
    std::thread t([&] { (void)timed_stream(down, h_down, d_down, bytes, 96, cudaMemcpyDeviceToHost); });
    with_down = timed_stream(up, d_up, h_up, bytes, 64, cudaMemcpyHostToDevice);
    t.join();
    std::printf("H2D alone %.2f GB/s, H2D with concurrent D2H %.2f GB/s\n", alone, with_down);
    CUDA_CHECK(cudaStreamDestroy(up));
    CUDA_CHECK(cudaStreamDestroy(down));
    CUDA_CHECK(cudaFree(d_up));
    CUDA_CHECK(cudaFree(d_down));
    CUDA_CHECK(cudaFreeHost(h_up));
    CUDA_CHECK(cudaFreeHost(h_down));
}

// ------------------------------------------------------------------------------------ registration

void registration() {
    std::printf("\n== register: cudaHostRegister of an mmap'd, populated region (4 GiB)\n");
    const std::size_t bytes = std::size_t{4} << 30;
    for (const bool thp : {false, true}) {
        void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) {
            std::perror("mmap");
            return;
        }
        madvise(p, bytes, thp ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
        std::memset(p, 1, bytes);
        const auto t0 = Clock::now();
        CUDA_CHECK(cudaHostRegister(p, bytes, cudaHostRegisterDefault));
        const double reg = seconds_since(t0);
        const auto t1    = Clock::now();
        CUDA_CHECK(cudaHostUnregister(p));
        const double unreg = seconds_since(t1);
        std::printf("%-22s register %.3f s/GiB, unregister %.3f s/GiB (70 GB arena: %.0f s)\n",
                    thp ? "THP (madvise)" : "4 KiB pages", reg / 4, unreg / 4, reg / 4 * 65.2);
        munmap(p, bytes);
    }
}

// ------------------------------------------------------------------------------------ contention

void contention() {
    std::printf("\n== contention: host DRAM read bandwidth while H2D DMA runs\n");
    const int threads        = static_cast<int>(std::max(1U, std::thread::hardware_concurrency() / 2));
    const std::size_t arena  = std::size_t{4} << 30;
    auto* cpu_buf            = static_cast<std::uint64_t*>(mmap(nullptr, arena, PROT_READ | PROT_WRITE,
                                                                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    madvise(cpu_buf, arena, MADV_HUGEPAGE);
    std::memset(cpu_buf, 1, arena);
    const std::size_t dma_bytes = std::size_t{1} << 30;
    void *h = nullptr, *d = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h, dma_bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaMalloc(&d, dma_bytes));
    std::memset(h, 1, dma_bytes);
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

    auto cpu_read = [&](double seconds) {
        std::atomic<bool> stop{false};
        std::vector<std::uint64_t> words(static_cast<std::size_t>(threads) * 8);
        std::vector<std::thread> pool;
        const std::size_t n = arena / 8;
        const auto t0       = Clock::now();
        for (int t = 0; t < threads; ++t) {
            pool.emplace_back([&, t] {
                const std::size_t b = n * t / threads, e = n * (t + 1) / threads;
                std::uint64_t acc = 0, count = 0;
                while (!stop.load(std::memory_order_relaxed)) {
                    for (std::size_t i = b; i < e; ++i) { acc += cpu_buf[i]; }
                    count += e - b;
                }
                words[static_cast<std::size_t>(t) * 8] = count + (acc == 42);
            });
        }
        std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
        stop.store(true);
        for (auto& th : pool) { th.join(); }
        std::uint64_t total = 0;
        for (int t = 0; t < threads; ++t) { total += words[static_cast<std::size_t>(t) * 8]; }
        return static_cast<double>(total) * 8 / seconds_since(t0) / 1e9;
    };

    const double cpu_alone = cpu_read(1.0);
    const double dma_alone = timed_stream(s, d, h, dma_bytes, 4, cudaMemcpyHostToDevice);
    std::atomic<bool> dma_stop{false};
    double dma_during = 0;
    std::thread dma([&] {
        double sum = 0;
        int n      = 0;
        while (!dma_stop.load()) {
            sum += timed_stream(s, d, h, dma_bytes, 1, cudaMemcpyHostToDevice);
            ++n;
        }
        dma_during = n ? sum / n : 0;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    const double cpu_during = cpu_read(1.5);
    dma_stop.store(true);
    dma.join();
    std::printf("%d CPU threads: alone %.1f GB/s, during H2D %.1f GB/s; H2D alone %.2f GB/s, during CPU reads %.2f GB/s\n",
                threads, cpu_alone, cpu_during, dma_alone, dma_during);
    CUDA_CHECK(cudaStreamDestroy(s));
    CUDA_CHECK(cudaFree(d));
    CUDA_CHECK(cudaFreeHost(h));
    munmap(cpu_buf, arena);
}

// ------------------------------------------------------------------------------------ PDL chains

__global__ void tiny_kernel(float* x, int pdl) {
    if (pdl) { cudaGridDependencySynchronize(); }
    x[blockIdx.x * blockDim.x + threadIdx.x] += 1.0F;
    if (pdl) { cudaTriggerProgrammaticLaunchCompletion(); }
}

void pdl_chain() {
    std::printf("\n== pdl: per-kernel time of a 507-kernel chain replayed as a CUDA Graph\n");
    constexpr int kKernels = 507;
    float* x               = nullptr;
    CUDA_CHECK(cudaMalloc(&x, 170 * 128 * sizeof(float)));
    CUDA_CHECK(cudaMemset(x, 0, 170 * 128 * sizeof(float)));
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    for (const int blocks : {1, 170}) {
        for (const int pdl : {0, 1}) {
            cudaGraph_t graph;
            CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
            for (int i = 0; i < kKernels; ++i) {
                cudaLaunchConfig_t cfg{};
                cfg.gridDim  = dim3(blocks);
                cfg.blockDim = dim3(128);
                cfg.stream   = s;
                cudaLaunchAttribute attr[1];
                attr[0].id                                         = cudaLaunchAttributeProgrammaticStreamSerialization;
                attr[0].val.programmaticStreamSerializationAllowed = 1;
                cfg.attrs                                          = attr;
                cfg.numAttrs                                       = pdl ? 1 : 0;
                CUDA_CHECK(cudaLaunchKernelEx(&cfg, tiny_kernel, x, pdl));
            }
            CUDA_CHECK(cudaStreamEndCapture(s, &graph));
            cudaGraphExec_t exec;
            CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
            CUDA_CHECK(cudaGraphLaunch(exec, s)); // warm-up
            CUDA_CHECK(cudaStreamSynchronize(s));
            cudaEvent_t a, b;
            CUDA_CHECK(cudaEventCreate(&a));
            CUDA_CHECK(cudaEventCreate(&b));
            std::vector<double> per;
            for (int r = 0; r < kRepetitions * 4; ++r) {
                CUDA_CHECK(cudaEventRecord(a, s));
                CUDA_CHECK(cudaGraphLaunch(exec, s));
                CUDA_CHECK(cudaEventRecord(b, s));
                CUDA_CHECK(cudaEventSynchronize(b));
                float ms = 0;
                CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
                per.push_back(ms * 1e3 / kKernels);
            }
            std::printf("%3d blocks, %-5s %.2f us per kernel, %.0f us per 507-kernel token\n", blocks,
                        pdl ? "PDL" : "plain", median(per), median(per) * kKernels);
            CUDA_CHECK(cudaEventDestroy(a));
            CUDA_CHECK(cudaEventDestroy(b));
            CUDA_CHECK(cudaGraphExecDestroy(exec));
            CUDA_CHECK(cudaGraphDestroy(graph));
        }
    }
    CUDA_CHECK(cudaStreamDestroy(s));
    CUDA_CHECK(cudaFree(x));
}

// ------------------------------------------------------------------------------------ mailbox

// One device thread publishes request i and spins until the host answers i (design §8.2).
__global__ void mailbox_kernel(volatile std::uint32_t* request, volatile std::uint32_t* reply, int rounds,
                               std::uint64_t* elapsed_ns) {
    const std::uint64_t t0 = global_ns();
    for (int i = 1; i <= rounds; ++i) {
        *request = static_cast<std::uint32_t>(i);
        __threadfence_system();
        while (*reply != static_cast<std::uint32_t>(i)) {}
    }
    *elapsed_ns = global_ns() - t0;
}

void mailbox() {
    std::printf("\n== mailbox: GPU -> host -> GPU round trip through mapped pinned memory\n");
    constexpr int kRounds = 20000;
    std::uint32_t* flags  = nullptr; // [0] request, [32] reply: separate 128-byte lines
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&flags), 4096, cudaHostAllocMapped));
    std::memset(flags, 0, 4096);
    std::uint32_t* dflags = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&dflags), flags, 0));
    std::uint64_t* elapsed = nullptr;
    CUDA_CHECK(cudaMalloc(&elapsed, sizeof(std::uint64_t)));
    std::thread host([&] {
        volatile std::uint32_t* req = flags;
        volatile std::uint32_t* rep = flags + 32;
        for (std::uint32_t seen = 0; seen < kRounds;) {
            const std::uint32_t r = *req;
            if (r != seen) {
                seen = r;
                std::atomic_thread_fence(std::memory_order_seq_cst);
                *rep = r;
            }
        }
    });
    mailbox_kernel<<<1, 1>>>(dflags, dflags + 32, kRounds, elapsed);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    host.join();
    std::uint64_t ns = 0;
    CUDA_CHECK(cudaMemcpy(&ns, elapsed, sizeof ns, cudaMemcpyDeviceToHost));
    std::printf("%d round trips: %.2f us each (device globaltimer)\n", kRounds, static_cast<double>(ns) / kRounds / 1e3);
    CUDA_CHECK(cudaFree(elapsed));
    CUDA_CHECK(cudaFreeHost(flags));
}

// ------------------------------------------------------------------------------------ submission

__global__ void flag_kernel(volatile std::uint32_t* flag, std::uint32_t value) {
    *flag = value;
    __threadfence_system();
}

void submission() {
    std::printf("\n== submit: host enqueue of k expert-record copies -> completion flag visible on the host\n");
    const int max_k = 16;
    void *h = nullptr, *d = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h, kRecordBytes * max_k, cudaHostAllocDefault));
    CUDA_CHECK(cudaMalloc(&d, kRecordBytes * max_k));
    std::uint32_t* flag = nullptr;
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&flag), 4096, cudaHostAllocMapped));
    std::uint32_t* dflag = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&dflag), flag, 0));
    *flag = 0;
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    std::uint32_t seq = 0;
    std::printf("%4s %14s %14s %16s\n", "k", "enqueue us", "visible us", "us per record");
    for (int k : {1, 2, 4, 8, 16}) {
        std::vector<double> enq, vis;
        for (int r = 0; r < kRepetitions * 4; ++r) {
            ++seq;
            const auto t0 = Clock::now();
            for (int i = 0; i < k; ++i) {
                CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(d) + i * kRecordBytes, static_cast<char*>(h) + i * kRecordBytes,
                                           kRecordBytes, cudaMemcpyHostToDevice, s));
            }
            flag_kernel<<<1, 1, 0, s>>>(dflag, seq);
            const double e = seconds_since(t0);
            while (*reinterpret_cast<volatile std::uint32_t*>(flag) != seq) {}
            vis.push_back(seconds_since(t0) * 1e6);
            enq.push_back(e * 1e6);
        }
        std::printf("%4d %14.1f %14.1f %16.1f\n", k, median(enq), median(vis), median(vis) / k);
    }
    CUDA_CHECK(cudaStreamSynchronize(s));
    CUDA_CHECK(cudaStreamDestroy(s));
    CUDA_CHECK(cudaFreeHost(flag));
    CUDA_CHECK(cudaFree(d));
    CUDA_CHECK(cudaFreeHost(h));
}

} // namespace

int main(int argc, char** argv) {
    const std::vector<std::pair<const char*, void (*)()>> sections{
        {"inventory", inventory}, {"copy", copy_sizes},   {"duplex", duplex}, {"register", registration},
        {"contention", contention}, {"pdl", pdl_chain}, {"mailbox", mailbox}, {"submit", submission},
    };
    std::vector<std::string> wanted(argv + 1, argv + argc);
    for (const auto& w : wanted) {
        if (std::none_of(sections.begin(), sections.end(), [&](const auto& s) { return w == s.first; })) {
            std::fprintf(stderr, "unknown section %s; sections:", w.c_str());
            for (const auto& s : sections) { std::fprintf(stderr, " %s", s.first); }
            std::fprintf(stderr, "\n");
            return 2;
        }
    }
    CUDA_CHECK(cudaSetDevice(0));
    for (const auto& [name, fn] : sections) {
        if (wanted.empty() || std::find(wanted.begin(), wanted.end(), name) != wanted.end()) { fn(); }
    }
    return 0;
}
