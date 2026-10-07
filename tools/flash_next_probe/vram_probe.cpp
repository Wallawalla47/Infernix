// VRAM facts for the memory track (design §19.3.7, measurements RM0a and the read half of RM0c;
// memory-tiers.md §3). Run alone on an otherwise idle GPU:
//
//   infernix_vram_probe [--parity-gib N] [--chunks N] [--only malloc,vmm,query,unmap,parity]
//
// It prints, for the CUDA device 0:
//   1. a snapshot of every VRAM source: cudaMemGetInfo; on Windows DXGI LOCAL/NON_LOCAL budget and
//      usage of the LUID-matched adapter, its outputs (displays), D3DKMT per-process segment-group
//      usage and Demoted bytes, and the local memory segments' commit/resident bytes; NVML (loaded
//      with LoadLibrary/dlopen) memory, DisplayMode and DisplayActive; the OS budget slack against
//      device free (Q2) and the display state by the §3.1 rule;
//   2. whether a 1 GiB cudaMalloc shows up in DXGI CurrentUsage and D3DKMT Usage (RM0a);
//   3. VMM: N x 64 MiB cuMemCreate + cuMemMap + cuMemSetAccess timed, the snapshot after mapping,
//      then cuMemUnmap + cuMemRelease timed and whether device free returns at once (RM0a);
//   4. the cost of each query (cudaMemGetInfo, DXGI, D3DKMT, NVML) from a second thread, alone and
//      while the first thread launches a stream of short kernels, and how much the queries slow that
//      launch loop;
//   5. whether cuMemUnmap/cuMemRelease on another thread stall kernel launches, and whether an
//      unmap waits for an unrelated kernel already running (implicit device synchronization);
//   6. read parity: a streaming read and a gather of random 2,764,800-byte records over N GiB of
//      cudaMalloc memory against the same over N GiB of 64 MiB VMM chunks (an Op-level proxy for
//      RM0c; RM0c itself is a tg512 A/B with the R2 toggle).
//
// Nothing here changes driver or system settings.

#include <cuda.h>
#include <cuda_runtime.h>

#ifdef _WIN32
#include <windows.h>
#include <winternl.h>
#include <d3dkmthk.h>
#include <dxgi1_6.h>
#include <psapi.h>
#else
#include <dlfcn.h>
#endif

#include <nvml.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

cudaError_t probe_spin(long long cycles, cudaStream_t stream);
cudaError_t probe_stream_read(const void* data, std::size_t bytes, unsigned* sink, cudaStream_t stream);
cudaError_t probe_gather_read(const void* base, std::size_t stride, const unsigned* records, unsigned count,
                              std::size_t record_bytes, unsigned slices, unsigned* sink, cudaStream_t stream);

namespace {

using Clock = std::chrono::steady_clock;
constexpr std::size_t kMiB = std::size_t{1} << 20;
constexpr std::size_t kChunk = 64 * kMiB;
constexpr std::size_t kRecord = 2'764'800;

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e)); }
}
void check(CUresult e, const char* what) {
    if (e != CUDA_SUCCESS) {
        const char* text = nullptr;
        cuGetErrorString(e, &text);
        throw std::runtime_error(std::string(what) + ": " + (text ? text : "?"));
    }
}

double us_since(Clock::time_point t) { return std::chrono::duration<double, std::micro>(Clock::now() - t).count(); }
long long mib(long long bytes) { return bytes / static_cast<long long>(kMiB); }

struct Stats {
    double p50 = 0, p90 = 0, p99 = 0, max = 0, mean = 0;
    std::size_t n = 0;
};
Stats stats(std::vector<double> v) {
    Stats s;
    if (v.empty()) { return s; }
    std::sort(v.begin(), v.end());
    const auto at = [&](double q) { return v[std::min(v.size() - 1, static_cast<std::size_t>(q * v.size()))]; };
    s.p50 = at(0.5);
    s.p90 = at(0.9);
    s.p99 = at(0.99);
    s.max = v.back();
    double sum = 0;
    for (double x : v) { sum += x; }
    s.mean = sum / v.size();
    s.n    = v.size();
    return s;
}
void print_stats(const char* label, const Stats& s, const char* unit = "us") {
    std::printf("  %-44s n %6zu  p50 %9.1f  p90 %9.1f  p99 %9.1f  max %9.1f  mean %9.1f %s\n", label, s.n, s.p50,
                s.p90, s.p99, s.max, s.mean, unit);
}

// ------------------------------------------------------------------ NVML through the loader
struct Nvml {
    using InitFn       = nvmlReturn_t (*)();
    using HandleFn     = nvmlReturn_t (*)(const char*, nvmlDevice_t*);
    using MemoryFn     = nvmlReturn_t (*)(nvmlDevice_t, nvmlMemory_v2_t*);
    using EnableFn     = nvmlReturn_t (*)(nvmlDevice_t, nvmlEnableState_t*);
    using DriverFn     = nvmlReturn_t (*)(nvmlDevice_t, nvmlDriverModel_t*, nvmlDriverModel_t*);
    void* library      = nullptr;
    nvmlDevice_t device = nullptr;
    MemoryFn memory     = nullptr;
    EnableFn display_mode = nullptr, display_active = nullptr;
    DriverFn driver_model = nullptr;

    void* symbol(const char* name) const {
#ifdef _WIN32
        return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(library), name));
#else
        return dlsym(library, name);
#endif
    }
    bool open(const char* pci_bus_id) {
#ifdef _WIN32
        library = LoadLibraryA("nvml.dll");
#else
        library = dlopen("libnvidia-ml.so.1", RTLD_NOW);
#endif
        if (library == nullptr) { return false; }
        const auto init   = reinterpret_cast<InitFn>(symbol("nvmlInit_v2"));
        const auto handle = reinterpret_cast<HandleFn>(symbol("nvmlDeviceGetHandleByPciBusId_v2"));
        memory            = reinterpret_cast<MemoryFn>(symbol("nvmlDeviceGetMemoryInfo_v2"));
        display_mode      = reinterpret_cast<EnableFn>(symbol("nvmlDeviceGetDisplayMode"));
        display_active    = reinterpret_cast<EnableFn>(symbol("nvmlDeviceGetDisplayActive"));
        driver_model      = reinterpret_cast<DriverFn>(symbol("nvmlDeviceGetDriverModel_v2"));
        return init != nullptr && handle != nullptr && init() == NVML_SUCCESS &&
               handle(pci_bus_id, &device) == NVML_SUCCESS;
    }
};

// ------------------------------------------------------------------ Windows sources
#ifdef _WIN32
struct Dxgi {
    IDXGIFactory4* factory = nullptr;
    IDXGIAdapter3* adapter = nullptr;
    HANDLE event           = nullptr;
    DWORD cookie           = 0;
    bool open(LUID luid) {
        if (FAILED(CreateDXGIFactory2(0, __uuidof(IDXGIFactory4), reinterpret_cast<void**>(&factory)))) { return false; }
        if (FAILED(factory->EnumAdapterByLuid(luid, __uuidof(IDXGIAdapter3), reinterpret_cast<void**>(&adapter)))) {
            return false;
        }
        event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (event != nullptr && FAILED(adapter->RegisterVideoMemoryBudgetChangeNotificationEvent(event, &cookie))) {
            CloseHandle(event);
            event = nullptr;
        }
        return true;
    }
    int outputs() const {
        int n = 0;
        IDXGIOutput* output = nullptr;
        while (adapter->EnumOutputs(static_cast<UINT>(n), &output) != DXGI_ERROR_NOT_FOUND) {
            if (output != nullptr) { output->Release(); }
            output = nullptr;
            ++n;
        }
        return n;
    }
    DXGI_QUERY_VIDEO_MEMORY_INFO query(DXGI_MEMORY_SEGMENT_GROUP group) const {
        DXGI_QUERY_VIDEO_MEMORY_INFO info{};
        adapter->QueryVideoMemoryInfo(0, group, &info);
        return info;
    }
};

struct Kmt {
    LUID luid{};
    ULONG segments = 0;
    bool ok        = false;
    bool open(LUID l) {
        luid = l;
        D3DKMT_QUERYSTATISTICS q{};
        q.Type        = D3DKMT_QUERYSTATISTICS_ADAPTER;
        q.AdapterLuid = luid;
        if (D3DKMTQueryStatistics(&q) < 0) { return false; }
        segments = q.QueryResult.AdapterInformation.NbSegments;
        ok       = true;
        return true;
    }
    // Per-process LOCAL segment group: budget, requested, usage, demoted (summed over classes).
    bool process_group(std::uint64_t& budget, std::uint64_t& requested, std::uint64_t& usage,
                       std::uint64_t& demoted) const {
        D3DKMT_QUERYSTATISTICS q{};
        q.Type                     = D3DKMT_QUERYSTATISTICS_PROCESS_SEGMENT_GROUP;
        q.AdapterLuid              = luid;
        q.hProcess                 = GetCurrentProcess();
        q.QueryProcessSegmentGroup = D3DKMT_MEMORY_SEGMENT_GROUP_LOCAL;
        if (D3DKMTQueryStatistics(&q) < 0) { return false; }
        const auto& g = q.QueryResult.ProcessSegmentGroupInformation;
        budget        = g.Budget;
        requested     = g.Requested;
        usage         = g.Usage;
        demoted       = 0;
        for (auto d : g.Demoted) { demoted += d; }
        return true;
    }
    // Device-wide local memory segments: commit limit, committed and resident bytes.
    bool device_segments(std::uint64_t& limit, std::uint64_t& committed, std::uint64_t& resident) const {
        limit = committed = resident = 0;
        bool any                     = false;
        for (ULONG s = 0; s < segments; ++s) {
            D3DKMT_QUERYSTATISTICS q{};
            q.Type                   = D3DKMT_QUERYSTATISTICS_SEGMENT;
            q.AdapterLuid            = luid;
            q.QuerySegment.SegmentId = s;
            if (D3DKMTQueryStatistics(&q) < 0) { continue; }
            const auto& seg = q.QueryResult.SegmentInformation;
            if (seg.Aperture != 0) { continue; }
            if (seg.SegmentProperties.SystemMemory != 0) { continue; }
            limit += seg.CommitLimit;
            committed += seg.BytesCommitted;
            resident += seg.BytesResident;
            any = true;
        }
        return any;
    }
};
#endif

// ------------------------------------------------------------------ snapshot
struct Probe;
struct Snapshot {
    std::size_t cuda_free = 0, cuda_total = 0;
    bool dxgi = false;
    std::uint64_t local_budget = 0, local_usage = 0, local_avail_res = 0, nonlocal_budget = 0, nonlocal_usage = 0;
    int outputs = -1;
    bool nvml = false;
    std::uint64_t nvml_total = 0, nvml_reserved = 0, nvml_free = 0, nvml_used = 0;
    int display_mode = -1, display_active = -1, driver_model = -1;
    bool kmt = false;
    std::uint64_t kmt_budget = 0, kmt_requested = 0, kmt_usage = 0, kmt_demoted = 0;
    bool kmt_seg = false;
    std::uint64_t seg_limit = 0, seg_committed = 0, seg_resident = 0;
    std::uint64_t private_usage = 0;
};

struct Probe {
    int device = 0;
    cudaDeviceProp prop{};
    Nvml nvml;
    bool has_nvml = false;
#ifdef _WIN32
    Dxgi dxgi;
    bool has_dxgi = false;
    Kmt kmt;
#endif

    void open() {
        check(cudaSetDevice(device), "cudaSetDevice");
        check(cudaFree(nullptr), "context");
        check(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
        char bus[32];
        std::snprintf(bus, sizeof(bus), "%08X:%02X:%02X.0", prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
        has_nvml = nvml.open(bus);
#ifdef _WIN32
        LUID luid{};
        std::memcpy(&luid, prop.luid, sizeof(luid));
        has_dxgi = dxgi.open(luid);
        kmt.open(luid);
#endif
    }

    Snapshot snapshot() const {
        Snapshot s;
        check(cudaMemGetInfo(&s.cuda_free, &s.cuda_total), "cudaMemGetInfo");
#ifdef _WIN32
        if (has_dxgi) {
            s.dxgi                = true;
            const auto local      = dxgi.query(DXGI_MEMORY_SEGMENT_GROUP_LOCAL);
            const auto nonlocal   = dxgi.query(DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL);
            s.local_budget        = local.Budget;
            s.local_usage         = local.CurrentUsage;
            s.local_avail_res     = local.AvailableForReservation;
            s.nonlocal_budget     = nonlocal.Budget;
            s.nonlocal_usage      = nonlocal.CurrentUsage;
            s.outputs             = dxgi.outputs();
        }
        if (kmt.ok) {
            s.kmt     = kmt.process_group(s.kmt_budget, s.kmt_requested, s.kmt_usage, s.kmt_demoted);
            s.kmt_seg = kmt.device_segments(s.seg_limit, s.seg_committed, s.seg_resident);
        }
        PROCESS_MEMORY_COUNTERS_EX pmc{};
        if (GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof(pmc))) {
            s.private_usage = pmc.PrivateUsage;
        }
#endif
        if (has_nvml) {
            nvmlMemory_v2_t m{};
            m.version = nvmlMemory_v2;
            if (nvml.memory != nullptr && nvml.memory(nvml.device, &m) == NVML_SUCCESS) {
                s.nvml          = true;
                s.nvml_total    = m.total;
                s.nvml_reserved = m.reserved;
                s.nvml_free     = m.free;
                s.nvml_used     = m.used;
            }
            nvmlEnableState_t state{};
            if (nvml.display_mode != nullptr && nvml.display_mode(nvml.device, &state) == NVML_SUCCESS) {
                s.display_mode = state == NVML_FEATURE_ENABLED ? 1 : 0;
            }
            if (nvml.display_active != nullptr && nvml.display_active(nvml.device, &state) == NVML_SUCCESS) {
                s.display_active = state == NVML_FEATURE_ENABLED ? 1 : 0;
            }
            nvmlDriverModel_t current{}, pending{};
            if (nvml.driver_model != nullptr && nvml.driver_model(nvml.device, &current, &pending) == NVML_SUCCESS) {
                s.driver_model = static_cast<int>(current);
            }
        }
        return s;
    }
};

void print_snapshot(const char* label, const Snapshot& s) {
    std::printf("[%s]\n", label);
    std::printf("  cudaMemGetInfo        free %8lld MiB  total %8lld MiB  used %8lld MiB\n",
                mib(static_cast<long long>(s.cuda_free)), mib(static_cast<long long>(s.cuda_total)),
                mib(static_cast<long long>(s.cuda_total - s.cuda_free)));
    if (s.dxgi) {
        std::printf("  DXGI LOCAL            budget %8lld MiB  usage %8lld MiB  available-for-reservation %lld MiB"
                    "  (budget - usage %lld MiB)\n",
                    mib(s.local_budget), mib(s.local_usage), mib(s.local_avail_res),
                    mib(static_cast<long long>(s.local_budget) - static_cast<long long>(s.local_usage)));
        std::printf("  DXGI NON_LOCAL        budget %8lld MiB  usage %8lld MiB\n", mib(s.nonlocal_budget),
                    mib(s.nonlocal_usage));
        std::printf("  DXGI outputs          %d\n", s.outputs);
    } else {
        std::printf("  DXGI                  unavailable\n");
    }
    if (s.kmt) {
        std::printf("  D3DKMT process LOCAL  budget %8lld MiB  requested %8lld MiB  usage %8lld MiB  demoted %lld MiB\n",
                    mib(s.kmt_budget), mib(s.kmt_requested), mib(s.kmt_usage), mib(s.kmt_demoted));
    }
    if (s.kmt_seg) {
        std::printf("  D3DKMT local segments limit %8lld MiB  committed %8lld MiB  resident %8lld MiB  "
                    "(limit - resident %lld MiB)\n",
                    mib(s.seg_limit), mib(s.seg_committed), mib(s.seg_resident),
                    mib(static_cast<long long>(s.seg_limit) - static_cast<long long>(s.seg_resident)));
    }
    if (s.nvml) {
        std::printf("  NVML                  total %8lld MiB  reserved %8lld MiB  free %8lld MiB  used %8lld MiB\n",
                    mib(s.nvml_total), mib(s.nvml_reserved), mib(s.nvml_free), mib(s.nvml_used));
    }
    std::printf("  NVML display          mode %d  active %d  driver model %d (0 WDDM, 1 WDM/TCC, 2 MCDM)\n",
                s.display_mode, s.display_active, s.driver_model);
    std::printf("  process PrivateUsage  %lld MiB\n", mib(s.private_usage));
    if (s.dxgi) {
        const long long slack = static_cast<long long>(s.local_budget) - static_cast<long long>(s.local_usage);
        std::printf("  OS budget slack B = budget - usage %lld MiB; device free F %lld MiB; B - F %+lld MiB\n",
                    mib(slack), mib(static_cast<long long>(s.cuda_free)),
                    mib(slack - static_cast<long long>(s.cuda_free)));
    }
    // memory-tiers.md §3.1: attached if outputs > 0 or NVML DisplayActive; headless if no outputs
    // and NVML unavailable or inactive; otherwise unknown (treated as attached on Windows).
    const char* display = (s.outputs > 0 || s.display_active == 1)    ? "attached"
                          : (s.outputs == 0 && s.display_active != 1) ? "headless"
                                                                      : "unknown";
    std::printf("  display state (§3.1 rule): %s\n", display);
}

void print_delta(const char* label, const Snapshot& a, const Snapshot& b, long long expected) {
    std::printf("  %-34s expected %+7lld MiB | cuda free %+7lld | DXGI LOCAL usage %+7lld | NON_LOCAL usage %+7lld | "
                "D3DKMT usage %+7lld | segments resident %+7lld | NVML used %+7lld | private %+7lld MiB\n",
                label, mib(expected), mib(static_cast<long long>(b.cuda_free) - static_cast<long long>(a.cuda_free)),
                mib(static_cast<long long>(b.local_usage) - static_cast<long long>(a.local_usage)),
                mib(static_cast<long long>(b.nonlocal_usage) - static_cast<long long>(a.nonlocal_usage)),
                mib(static_cast<long long>(b.kmt_usage) - static_cast<long long>(a.kmt_usage)),
                mib(static_cast<long long>(b.seg_resident) - static_cast<long long>(a.seg_resident)),
                mib(static_cast<long long>(b.nvml_used) - static_cast<long long>(a.nvml_used)),
                mib(static_cast<long long>(b.private_usage) - static_cast<long long>(a.private_usage)));
}

// ------------------------------------------------------------------ VMM chunks
struct VmmChunks {
    CUdeviceptr base = 0;
    std::size_t reserved = 0;
    std::vector<CUmemGenericAllocationHandle> handles;
    CUmemAllocationProp prop{};
    CUmemAccessDesc access{};

    explicit VmmChunks(std::size_t chunks, int device) {
        prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id   = device;
        access.location    = prop.location;
        access.flags       = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        reserved           = chunks * kChunk;
        check(cuMemAddressReserve(&base, reserved, kChunk, 0, 0), "cuMemAddressReserve");
    }
    ~VmmChunks() {
        while (!handles.empty()) { unmap_top(); }
        if (base != 0) { cuMemAddressFree(base, reserved); }
    }
    // Returns {create, map, set-access} microseconds.
    std::array<double, 3> map_top() {
        const CUdeviceptr at = base + handles.size() * kChunk;
        CUmemGenericAllocationHandle h{};
        auto t = Clock::now();
        check(cuMemCreate(&h, kChunk, &prop, 0), "cuMemCreate");
        const double create = us_since(t);
        t                   = Clock::now();
        check(cuMemMap(at, kChunk, 0, h, 0), "cuMemMap");
        const double map = us_since(t);
        t                = Clock::now();
        check(cuMemSetAccess(at, kChunk, &access, 1), "cuMemSetAccess");
        const double set = us_since(t);
        handles.push_back(h);
        return {create, map, set};
    }
    // Returns {unmap, release} microseconds.
    std::array<double, 2> unmap_top() {
        const CUdeviceptr at = base + (handles.size() - 1) * kChunk;
        auto t               = Clock::now();
        check(cuMemUnmap(at, kChunk), "cuMemUnmap");
        const double unmap = us_since(t);
        t                  = Clock::now();
        check(cuMemRelease(handles.back()), "cuMemRelease");
        const double release = us_since(t);
        handles.pop_back();
        return {unmap, release};
    }
    [[nodiscard]] void* data() const { return reinterpret_cast<void*>(base); }
    [[nodiscard]] std::size_t mapped() const { return handles.size() * kChunk; }
};

// ------------------------------------------------------------------ sections
void section_malloc(const Probe& probe) {
    std::printf("\n== RM0a: does a CUDA allocation show up in each source? (1 GiB cudaMalloc + memset)\n");
    const Snapshot a = probe.snapshot();
    void* p          = nullptr;
    const std::size_t bytes = std::size_t{1} << 30;
    check(cudaMalloc(&p, bytes), "cudaMalloc");
    check(cudaMemset(p, 1, bytes), "cudaMemset");
    check(cudaDeviceSynchronize(), "sync");
    const Snapshot b = probe.snapshot();
    print_delta("after cudaMalloc 1 GiB", a, b, static_cast<long long>(bytes));
    check(cudaFree(p), "cudaFree");
    check(cudaDeviceSynchronize(), "sync");
    const Snapshot c = probe.snapshot();
    print_delta("after cudaFree", b, c, -static_cast<long long>(bytes));
    const long long usage = static_cast<long long>(b.local_usage) - static_cast<long long>(a.local_usage);
    std::printf("  => DXGI LOCAL CurrentUsage %s CUDA allocations (delta %lld MiB of 1024)\n",
                !b.dxgi ? "unavailable for" : (usage > 900LL * static_cast<long long>(kMiB) ? "COUNTS" : "does NOT count"),
                mib(usage));
}

void section_vmm(const Probe& probe, std::size_t chunks) {
    std::printf("\n== RM0a: VMM create/map/access and unmap/release, %zu x 64 MiB\n", chunks);
    int supported = 0;
    check(cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, probe.device),
          "VMM attribute");
    VmmChunks vmm(chunks, probe.device);
    std::size_t minimum = 0, recommended = 0;
    check(cuMemGetAllocationGranularity(&minimum, &vmm.prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM), "granularity");
    check(cuMemGetAllocationGranularity(&recommended, &vmm.prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED), "granularity");
    std::printf("  VMM supported %d; granularity minimum %zu KiB, recommended %zu KiB\n", supported, minimum >> 10,
                recommended >> 10);
    const Snapshot a = probe.snapshot();
    std::vector<double> create, map, set, unmap, release;
    for (std::size_t i = 0; i < chunks; ++i) {
        const auto t = vmm.map_top();
        create.push_back(t[0]);
        map.push_back(t[1]);
        set.push_back(t[2]);
    }
    check(cudaMemset(vmm.data(), 2, vmm.mapped()), "memset VMM");
    check(cudaDeviceSynchronize(), "sync");
    const Snapshot b = probe.snapshot();
    print_stats("cuMemCreate 64 MiB", stats(create));
    print_stats("cuMemMap 64 MiB", stats(map));
    print_stats("cuMemSetAccess 64 MiB", stats(set));
    print_delta("after mapping + memset", a, b, static_cast<long long>(vmm.mapped()));
    const std::size_t mapped = vmm.mapped();
    while (vmm.mapped() != 0) {
        const auto t = vmm.unmap_top();
        unmap.push_back(t[0]);
        release.push_back(t[1]);
    }
    const Snapshot c = probe.snapshot();
    print_stats("cuMemUnmap 64 MiB", stats(unmap));
    print_stats("cuMemRelease 64 MiB", stats(release));
    print_delta("right after unmap + release", b, c, -static_cast<long long>(mapped));
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    print_delta("50 ms later", b, probe.snapshot(), -static_cast<long long>(mapped));
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    print_delta("1 s later", b, probe.snapshot(), -static_cast<long long>(mapped));
    const long long returned = static_cast<long long>(c.cuda_free) - static_cast<long long>(b.cuda_free);
    std::printf("  => device free %s at release (%lld of %lld MiB back immediately)\n",
                returned >= static_cast<long long>(mapped) - 32LL * static_cast<long long>(kMiB) ? "RETURNS" : "does NOT return",
                mib(returned), mib(static_cast<long long>(mapped)));
}

// Launches `count` spin kernels of `cycles` on `stream` and returns the host wall time in us of
// launching them and waiting for all; `launches` collects each launch call's host latency.
double launch_loop(cudaStream_t stream, int count, long long cycles, std::vector<double>* launches) {
    const auto start = Clock::now();
    for (int i = 0; i < count; ++i) {
        const auto t = Clock::now();
        check(probe_spin(cycles, stream), "spin launch");
        if (launches != nullptr) { launches->push_back(us_since(t)); }
    }
    check(cudaStreamSynchronize(stream), "sync");
    return us_since(start);
}

void section_query_cost(const Probe& probe) {
    std::printf("\n== RM0a: query cost from a second thread, idle and during a kernel loop\n");
    cudaStream_t stream = nullptr;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    const long long cycles = 20'000; // ~10 us at ~2 GHz
    const int count        = 20'000;
    launch_loop(stream, 2'000, cycles, nullptr); // warm-up
    struct Query {
        const char* name;
        std::function<void()> call;
    };
    std::vector<Query> queries;
    queries.push_back({"cudaMemGetInfo", [] {
                           std::size_t f = 0, t = 0;
                           check(cudaMemGetInfo(&f, &t), "cudaMemGetInfo");
                       }});
#ifdef _WIN32
    if (probe.has_dxgi) {
        queries.push_back({"DXGI QueryVideoMemoryInfo LOCAL", [&] { (void)probe.dxgi.query(DXGI_MEMORY_SEGMENT_GROUP_LOCAL); }});
        queries.push_back({"DXGI IsCurrent + EnumOutputs", [&] {
                               (void)probe.dxgi.factory->IsCurrent();
                               (void)probe.dxgi.outputs();
                           }});
    }
    if (probe.kmt.ok) {
        queries.push_back({"D3DKMT process segment group", [&] {
                               std::uint64_t a, b, c, d;
                               (void)probe.kmt.process_group(a, b, c, d);
                           }});
        queries.push_back({"D3DKMT local segments (device free)", [&] {
                               std::uint64_t a, b, c;
                               (void)probe.kmt.device_segments(a, b, c);
                           }});
    }
#endif
    if (probe.has_nvml && probe.nvml.memory != nullptr) {
        queries.push_back({"NVML memory v2", [&] {
                               nvmlMemory_v2_t m{};
                               m.version = nvmlMemory_v2;
                               (void)probe.nvml.memory(probe.nvml.device, &m);
                           }});
    }
    const double base = launch_loop(stream, count, cycles, nullptr);
    std::printf("  launch loop alone: %d spin kernels in %.1f ms (%.2f us per kernel)\n", count, base / 1000,
                base / count);
    for (const auto& q : queries) {
        std::vector<double> idle;
        for (int i = 0; i < 2'000; ++i) {
            const auto t = Clock::now();
            q.call();
            idle.push_back(us_since(t));
        }
        std::atomic<bool> stop{false};
        std::vector<double> busy;
        std::thread worker([&] {
            check(cudaSetDevice(probe.device), "cudaSetDevice");
            while (!stop.load(std::memory_order_relaxed)) {
                const auto t = Clock::now();
                q.call();
                busy.push_back(us_since(t));
                std::this_thread::sleep_for(std::chrono::microseconds(200));
            }
        });
        const double loaded = launch_loop(stream, count, cycles, nullptr);
        stop = true;
        worker.join();
        std::printf("  %s\n", q.name);
        print_stats("    alone", stats(idle));
        print_stats("    during the launch loop (every ~0.2 ms)", stats(busy));
        std::printf("    launch loop with these queries: %.1f ms (%+.2f %% against alone)\n", loaded / 1000,
                    100.0 * (loaded - base) / base);
    }
#ifdef _WIN32
    const Snapshot s = probe.snapshot();
    if (s.kmt_seg) {
        std::printf("  device free: cudaMemGetInfo %lld MiB; D3DKMT segment limit - resident %lld MiB; NVML free %lld MiB\n",
                    mib(static_cast<long long>(s.cuda_free)),
                    mib(static_cast<long long>(s.seg_limit) - static_cast<long long>(s.seg_resident)),
                    mib(static_cast<long long>(s.nvml_free)));
    }
#endif
    check(cudaStreamDestroy(stream), "stream destroy");
}

void section_unmap_stall(const Probe& probe) {
    std::printf("\n== RM0a: do VMM map/unmap/release on another thread stall kernel launches?\n");
    cudaStream_t stream = nullptr;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    const long long cycles = 20'000;
    const int count        = 20'000;
    launch_loop(stream, 2'000, cycles, nullptr);
    std::vector<double> alone_launches;
    const double alone = launch_loop(stream, count, cycles, &alone_launches);
    std::atomic<bool> stop{false};
    std::vector<double> unmaps, releases, maps;
    std::thread worker([&] {
        check(cudaSetDevice(probe.device), "cudaSetDevice");
        VmmChunks vmm(4, probe.device);
        while (!stop.load(std::memory_order_relaxed)) {
            for (int i = 0; i < 4; ++i) {
                const auto t = vmm.map_top();
                maps.push_back(t[0] + t[1] + t[2]);
            }
            for (int i = 0; i < 4; ++i) {
                const auto t = vmm.unmap_top();
                unmaps.push_back(t[0]);
                releases.push_back(t[1]);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    });
    std::vector<double> busy_launches;
    const double busy = launch_loop(stream, count, cycles, &busy_launches);
    stop = true;
    worker.join();
    print_stats("launch call, alone", stats(alone_launches));
    print_stats("launch call, with map/unmap cycles", stats(busy_launches));
    print_stats("create+map+access 64 MiB during the loop", stats(maps));
    print_stats("cuMemUnmap 64 MiB during the loop", stats(unmaps));
    print_stats("cuMemRelease 64 MiB during the loop", stats(releases));
    std::printf("  launch loop: alone %.1f ms, with VMM cycles %.1f ms (%+.2f %%)\n", alone / 1000, busy / 1000,
                100.0 * (busy - alone) / alone);

    // Does an unmap wait for an unrelated kernel that is already running?
    VmmChunks vmm(1, probe.device);
    vmm.map_top();
    // Calibrate the spin: cycles per millisecond from a timed 50-million-cycle kernel.
    cudaEvent_t e0, e1;
    check(cudaEventCreate(&e0), "event");
    check(cudaEventCreate(&e1), "event");
    check(cudaEventRecord(e0, stream), "record");
    check(probe_spin(50'000'000, stream), "calibration spin");
    check(cudaEventRecord(e1, stream), "record");
    check(cudaEventSynchronize(e1), "sync");
    float calibration_ms = 0;
    check(cudaEventElapsedTime(&calibration_ms, e0, e1), "elapsed");
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    const long long long_cycles = static_cast<long long>(50'000'000.0 / calibration_ms * 200.0); // ~200 ms
    check(probe_spin(long_cycles, stream), "long spin");
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    const auto t     = Clock::now();
    const auto times = vmm.unmap_top();
    const double total = us_since(t);
    const auto w       = Clock::now();
    check(cudaStreamSynchronize(stream), "sync");
    std::printf("  unmap + release while an unrelated ~200 ms kernel runs: unmap %.1f us, release %.1f us (total %.1f "
                "ms); the kernel then needed %.1f ms more\n",
                times[0], times[1], total / 1000, us_since(w) / 1000);
    std::printf("  => unmap %s for running kernels\n", total > 100'000 ? "WAITS (implicit synchronization)" : "does not wait");
    check(cudaStreamDestroy(stream), "stream destroy");
}

double read_gbs(const void* data, std::size_t bytes, unsigned* sink, cudaStream_t stream, int reps) {
    check(probe_stream_read(data, bytes, sink, stream), "stream read");
    check(cudaStreamSynchronize(stream), "sync");
    cudaEvent_t a, b;
    check(cudaEventCreate(&a), "event");
    check(cudaEventCreate(&b), "event");
    check(cudaEventRecord(a, stream), "record");
    for (int i = 0; i < reps; ++i) { check(probe_stream_read(data, bytes, sink, stream), "stream read"); }
    check(cudaEventRecord(b, stream), "record");
    check(cudaEventSynchronize(b), "sync");
    float ms = 0;
    check(cudaEventElapsedTime(&ms, a, b), "elapsed");
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return static_cast<double>(bytes) * reps / (ms * 1e6);
}

double gather_gbs(const void* data, std::size_t bytes, const unsigned* records, unsigned count, unsigned* sink,
                  cudaStream_t stream, int reps) {
    check(probe_gather_read(data, kRecord, records, count, kRecord, 16, sink, stream), "gather");
    check(cudaStreamSynchronize(stream), "sync");
    cudaEvent_t a, b;
    check(cudaEventCreate(&a), "event");
    check(cudaEventCreate(&b), "event");
    check(cudaEventRecord(a, stream), "record");
    for (int i = 0; i < reps; ++i) {
        check(probe_gather_read(data, kRecord, records, count, kRecord, 16, sink, stream), "gather");
    }
    check(cudaEventRecord(b, stream), "record");
    check(cudaEventSynchronize(b), "sync");
    float ms = 0;
    check(cudaEventElapsedTime(&ms, a, b), "elapsed");
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    (void)bytes;
    return static_cast<double>(kRecord) * count * reps / (ms * 1e6);
}

void section_parity(const Probe& probe, std::size_t gib) {
    std::printf("\n== RM0c proxy: reads from cudaMalloc against 64 MiB VMM chunks, %zu GiB each\n", gib);
    const std::size_t bytes  = gib << 30;
    const std::size_t chunks = bytes / kChunk;
    cudaStream_t stream      = nullptr;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    unsigned* sink = nullptr;
    check(cudaMalloc(&sink, 4), "sink");
    const unsigned frames = static_cast<unsigned>(bytes / kRecord);
    std::mt19937 rng(2026);
    std::vector<unsigned> picks(4096);
    for (auto& f : picks) { f = rng() % frames; }
    unsigned* records = nullptr;
    check(cudaMalloc(&records, picks.size() * 4), "records");
    check(cudaMemcpy(records, picks.data(), picks.size() * 4, cudaMemcpyHostToDevice), "records");
    void* plain = nullptr;
    check(cudaMalloc(&plain, bytes), "cudaMalloc parity");
    check(cudaMemset(plain, 3, bytes), "memset");
    VmmChunks vmm(chunks, probe.device);
    for (std::size_t i = 0; i < chunks; ++i) { vmm.map_top(); }
    check(cudaMemset(vmm.data(), 3, bytes), "memset");
    check(cudaDeviceSynchronize(), "sync");
    // ABBA, three rounds.
    std::vector<double> stream_plain, stream_vmm, gather_plain, gather_vmm;
    for (int round = 0; round < 3; ++round) {
        stream_plain.push_back(read_gbs(plain, bytes, sink, stream, 5));
        stream_vmm.push_back(read_gbs(vmm.data(), bytes, sink, stream, 5));
        stream_vmm.push_back(read_gbs(vmm.data(), bytes, sink, stream, 5));
        stream_plain.push_back(read_gbs(plain, bytes, sink, stream, 5));
        gather_plain.push_back(gather_gbs(plain, bytes, records, 4096, sink, stream, 10));
        gather_vmm.push_back(gather_gbs(vmm.data(), bytes, records, 4096, sink, stream, 10));
        gather_vmm.push_back(gather_gbs(vmm.data(), bytes, records, 4096, sink, stream, 10));
        gather_plain.push_back(gather_gbs(plain, bytes, records, 4096, sink, stream, 10));
    }
    const auto line = [](const char* label, std::vector<double> a, std::vector<double> b) {
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        const double ma = a[a.size() / 2], mb = b[b.size() / 2];
        std::printf("  %-38s cudaMalloc median %7.1f GB/s (%.1f-%.1f)  VMM median %7.1f GB/s (%.1f-%.1f)  VMM %+.2f %%\n",
                    label, ma, a.front(), a.back(), mb, b.front(), b.back(), 100.0 * (mb - ma) / ma);
    };
    line("streaming read", stream_plain, stream_vmm);
    line("gather of 4,096 random 2.76 MB records", gather_plain, gather_vmm);
    check(cudaFree(plain), "free");
    check(cudaFree(records), "free");
    check(cudaFree(sink), "free");
    check(cudaStreamDestroy(stream), "stream destroy");
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::size_t parity_gib = 8, chunks = 16;
        std::string only;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--parity-gib" && i + 1 < argc) {
                parity_gib = std::stoul(argv[++i]);
            } else if (arg == "--chunks" && i + 1 < argc) {
                chunks = std::stoul(argv[++i]);
            } else if (arg == "--only" && i + 1 < argc) {
                only = "," + std::string(argv[++i]) + ",";
            } else {
                std::fprintf(stderr,
                             "usage: infernix_vram_probe [--parity-gib N] [--chunks N] "
                             "[--only malloc,vmm,query,unmap,parity]\n");
                return 2;
            }
        }
        const auto wanted = [&](const char* name) {
            return only.empty() || only.find("," + std::string(name) + ",") != std::string::npos;
        };
        check(cuInit(0), "cuInit");
        Probe probe;
        probe.open();
        std::printf("device %d: %s, %zu MiB, PCI %04x:%02x:%02x, NVML %s, DXGI %s\n", probe.device, probe.prop.name,
                    probe.prop.totalGlobalMem >> 20, probe.prop.pciDomainID, probe.prop.pciBusID,
                    probe.prop.pciDeviceID, probe.has_nvml ? "loaded" : "unavailable",
#ifdef _WIN32
                    probe.has_dxgi ? "adapter matched by LUID" : "unavailable"
#else
                    "not on this platform"
#endif
        );
        print_snapshot("baseline after context creation", probe.snapshot());
        if (wanted("malloc")) { section_malloc(probe); }
        if (wanted("vmm")) { section_vmm(probe, chunks); }
        if (wanted("query")) { section_query_cost(probe); }
        if (wanted("unmap")) { section_unmap_stall(probe); }
        if (wanted("parity")) { section_parity(probe, parity_gib); }
#ifdef _WIN32
        if (probe.dxgi.event != nullptr) {
            std::printf("\nDXGI budget-change event signalled during the run: %s\n",
                        WaitForSingleObject(probe.dxgi.event, 0) == WAIT_OBJECT_0 ? "yes" : "no");
        }
#endif
        print_snapshot("final", probe.snapshot());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
