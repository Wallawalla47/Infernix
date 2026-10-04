// Expert-record reads from the artifact in place, and pinned host allocation time, for the memory
// track's SSD tier (design §19.3.7; measurements RM0e and RM0g; memory-tiers.md §4.4-§4.10). Run
// alone on an otherwise idle machine; it only reads the artifact and never writes a file.
//
//   ninfer_expert_read_probe ARTIFACT.ninfer [--seconds 5] [--depths 1,2,4,8] [--quarter-depths 1,2,4]
//       [--only reads,quarters,load,subread,straddlers,pin,pinlimit] [--pin-chunks 10]
//       [--pin-limit-gib 0] [--seed 2026]
//
// The routed-expert records (nvfp4_expert_rg16_v1, 2,764,800 B each, 4 KiB aligned) are located
// through the artifact directory and read unbuffered (Windows: FILE_FLAG_NO_BUFFERING, overlapped,
// one I/O completion port, deny-write sharing as the tier will open them; POSIX: O_DIRECT with one
// thread per I/O in flight) into pinned, mapped host slots (cudaHostAlloc Portable | Mapped), as
// the tier will. Records are drawn uniformly at random over all banks; a record that straddles two
// part files is read as two pieces. Sections:
//   reads       whole-record reads at each queue depth (records in flight): latency per record
//               (issue to completion) p50/p90/p99/max/mean, GB/s and records/s
//   quarters    each record as four parallel page-aligned quarter reads (169, 169, 169, 168 pages):
//               does splitting lower the single-read latency (§4.6.4)?
//   load        H2D bandwidth alone, then whole reads at QD 1 and 4 and quarters at QD 1 while a
//               second thread streams H2D copies from pinned memory (the decode promotions and
//               staging), with the H2D rate achieved meanwhile
//   subread     QD 1 demand reads alone and with one 256 KiB prefetch sub-read always in flight
//               (§4.9), and the sub-reads' own latency
//   straddlers  the records split across part files read through their two pieces, compared with a
//               buffered read of the same logical range
//   pin         RM0g: cudaHostAlloc(Portable | Mapped) time of N chunks of 512 slots (1.32 GiB)
//               against one block of the same total, chunks / block / chunks, with touch and free
//               times; skipped unless available RAM covers the block plus 8 GiB
//   pinlimit    (only when --pin-limit-gib > 0) pins 1.32 GiB chunks until a failure, the cap, or
//               available RAM below 4 GiB, and reports the total (the WSL pinned-memory limit)
// Default: every section except pinlimit.

#include "artifact/reader.h"
#include "artifact/schema.h"

#include <cuda_runtime.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/resource.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <variant>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
constexpr std::uint64_t kRecord      = 2'764'800; // one routed expert, 675 x 4 KiB
constexpr std::uint64_t kPage        = 4096;
constexpr std::uint64_t kSubRead     = 256 * 1024;
constexpr std::uint64_t kSlotsChunk  = 512;       // host-tier pinned chunk: 512 slots = 675 x 2 MiB
constexpr std::uint64_t kChunkBytes  = kSlotsChunk * kRecord;
constexpr std::uint64_t kGiB         = std::uint64_t{1} << 30;
constexpr int kMaxPieces             = 8;         // 4 quarters x 2 file pieces

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e)); }
}

double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

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
    s.mean = sum / static_cast<double>(v.size());
    s.n    = v.size();
    return s;
}

std::uint64_t available_ram() {
#ifdef _WIN32
    MEMORYSTATUSEX m{};
    m.dwLength = sizeof(m);
    return GlobalMemoryStatusEx(&m) ? m.ullAvailPhys : 0;
#else
    std::ifstream in("/proc/meminfo");
    std::string key;
    std::uint64_t value = 0;
    std::string unit;
    while (in >> key >> value >> unit) {
        if (key == "MemAvailable:") { return value * 1024; }
    }
    return 0;
#endif
}

// ------------------------------------------------------------------ the artifact's records
struct Piece {
    std::size_t file      = 0;
    std::uint64_t offset  = 0; // file offset
    std::uint64_t at      = 0; // offset inside the destination
    std::uint64_t bytes   = 0;
};

class Records {
public:
    explicit Records(const std::filesystem::path& artifact) : reader_(artifact) {
        const auto& directory = reader_.directory();
        for (std::size_t i = 0; i < directory.files.size(); ++i) {
            paths_.push_back(i == 0 ? artifact : artifact.parent_path() / *directory.files[i].path);
        }
        struct Bank {
            std::uint64_t offset = 0, experts = 0;
        };
        std::vector<Bank> banks;
        for (const auto& object : directory.objects) {
            const auto* tensor = std::get_if<ninfer::artifact::TensorObject>(&object);
            if (tensor == nullptr || tensor->layout != "nvfp4_expert_rg16_v1" || tensor->shape.empty()) { continue; }
            const std::uint64_t experts = tensor->shape[0];
            if (tensor->bytes < experts * kRecord || tensor->bytes - experts * kRecord >= kRecord ||
                tensor->offset % kPage != 0) {
                throw std::runtime_error(tensor->id + ": not a bank of 2,764,800-byte page-aligned records");
            }
            banks.push_back({tensor->offset, experts});
        }
        if (banks.empty()) { throw std::runtime_error("no nvfp4_expert_rg16_v1 banks in the artifact"); }
        std::sort(banks.begin(), banks.end(), [](const Bank& a, const Bank& b) { return a.offset < b.offset; });
        for (std::size_t b = 0; b < banks.size(); ++b) {
            for (std::uint64_t e = 0; e < banks[b].experts; ++e) {
                offsets_.push_back(banks[b].offset + e * kRecord);
                if (reader_.segments(offsets_.back(), kRecord).size() > 1) {
                    straddlers_.push_back(offsets_.size() - 1);
                    std::printf("record %zu (bank %zu by offset, expert %llu) straddles two files\n",
                                offsets_.size() - 1, b, static_cast<unsigned long long>(e));
                }
            }
        }
        std::printf("artifact: %zu files, %zu banks, %zu records of %llu B\n", paths_.size(), banks.size(),
                    offsets_.size(), static_cast<unsigned long long>(kRecord));
    }

    [[nodiscard]] const std::vector<std::filesystem::path>& paths() const { return paths_; }
    [[nodiscard]] std::size_t count() const { return offsets_.size(); }
    [[nodiscard]] const std::vector<std::size_t>& straddlers() const { return straddlers_; }
    [[nodiscard]] const ninfer::artifact::Reader& reader() const { return reader_; }

    // The file pieces of bytes [begin, begin + bytes) of record `index`.
    void pieces(std::size_t index, std::uint64_t begin, std::uint64_t bytes, std::vector<Piece>& out) const {
        for (const auto& s : reader_.segments(offsets_[index] + begin, bytes)) {
            if (s.file_offset % kPage != 0 || s.bytes % kPage != 0) {
                throw std::runtime_error("record piece is not page aligned");
            }
            out.push_back({s.file_index, s.file_offset, begin + s.destination_offset, s.bytes});
        }
    }
    [[nodiscard]] std::uint64_t logical(std::size_t index) const { return offsets_[index]; }

private:
    ninfer::artifact::Reader reader_;
    std::vector<std::filesystem::path> paths_;
    std::vector<std::uint64_t> offsets_;
    std::vector<std::size_t> straddlers_;
};

// ------------------------------------------------------------------ unbuffered asynchronous reads
struct Op {
#ifdef _WIN32
    OVERLAPPED overlapped{}; // first member: completions map back to the Op
#endif
    std::size_t file     = 0;
    std::uint64_t offset = 0;
    std::uint32_t bytes  = 0;
    std::byte* dest      = nullptr;
    int owner            = 0; // the measurement's slot
    bool failed          = false;
};

class DirectReads {
public:
    DirectReads(const std::vector<std::filesystem::path>& paths, std::size_t max_in_flight) {
#ifdef _WIN32
        (void)max_in_flight;
        for (std::size_t i = 0; i < paths.size(); ++i) {
            // Deny writers, as the tier's handles will (memory-tiers.md §4.4).
            HANDLE file = CreateFileW(paths[i].c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr,
                                      OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, nullptr);
            if (file == INVALID_HANDLE_VALUE) {
                throw std::runtime_error("cannot open " + paths[i].string() + " unbuffered (error " +
                                         std::to_string(GetLastError()) + ")");
            }
            files_.push_back(file);
            port_ = CreateIoCompletionPort(file, port_, static_cast<ULONG_PTR>(i), 0);
            if (port_ == nullptr) { throw std::runtime_error("CreateIoCompletionPort failed"); }
        }
#else
        for (const auto& path : paths) {
            const int fd = ::open(path.c_str(), O_RDONLY | O_DIRECT);
            if (fd < 0) { throw std::runtime_error("cannot open " + path.string() + " with O_DIRECT"); }
            files_.push_back(fd);
        }
        for (std::size_t i = 0; i < max_in_flight; ++i) {
            workers_.emplace_back([this] { work(); });
        }
#endif
    }
    ~DirectReads() {
#ifdef _WIN32
        for (HANDLE h : files_) { CloseHandle(h); }
        if (port_ != nullptr) { CloseHandle(port_); }
#else
        {
            const std::lock_guard lock(mutex_);
            stop_ = true;
        }
        work_cv_.notify_all();
        for (auto& t : workers_) { t.join(); }
        for (int fd : files_) { ::close(fd); }
#endif
    }
    DirectReads(const DirectReads&)            = delete;
    DirectReads& operator=(const DirectReads&) = delete;

    void submit(Op* op) {
        op->failed = false;
#ifdef _WIN32
        op->overlapped            = {};
        op->overlapped.Offset     = static_cast<DWORD>(op->offset);
        op->overlapped.OffsetHigh = static_cast<DWORD>(op->offset >> 32);
        if (!ReadFile(files_[op->file], op->dest, op->bytes, nullptr, &op->overlapped) &&
            GetLastError() != ERROR_IO_PENDING) {
            throw std::runtime_error("ReadFile failed (error " + std::to_string(GetLastError()) + ")");
        }
#else
        {
            const std::lock_guard lock(mutex_);
            queue_.push_back(op);
        }
        work_cv_.notify_one();
#endif
    }

    // Blocks until at least one read completes and appends the completed ones.
    void wait(std::vector<Op*>& done) {
#ifdef _WIN32
        OVERLAPPED_ENTRY entries[64];
        ULONG n = 0;
        if (!GetQueuedCompletionStatusEx(port_, entries, 64, &n, INFINITE, FALSE)) {
            throw std::runtime_error("GetQueuedCompletionStatusEx failed");
        }
        for (ULONG i = 0; i < n; ++i) {
            auto* op   = reinterpret_cast<Op*>(entries[i].lpOverlapped);
            op->failed = op->overlapped.Internal != 0 || entries[i].dwNumberOfBytesTransferred != op->bytes;
            done.push_back(op);
        }
#else
        std::unique_lock lock(mutex_);
        done_cv_.wait(lock, [this] { return !completed_.empty(); });
        done.insert(done.end(), completed_.begin(), completed_.end());
        completed_.clear();
#endif
    }

private:
#ifdef _WIN32
    std::vector<HANDLE> files_;
    HANDLE port_ = nullptr;
#else
    void work() {
        while (true) {
            Op* op = nullptr;
            {
                std::unique_lock lock(mutex_);
                work_cv_.wait(lock, [this] { return stop_ || !queue_.empty(); });
                if (stop_) { return; }
                op = queue_.front();
                queue_.pop_front();
            }
            const ssize_t got = ::pread(files_[op->file], op->dest, op->bytes, static_cast<off_t>(op->offset));
            op->failed        = got != static_cast<ssize_t>(op->bytes);
            {
                const std::lock_guard lock(mutex_);
                completed_.push_back(op);
            }
            done_cv_.notify_one();
        }
    }
    std::vector<int> files_;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable work_cv_, done_cv_;
    std::deque<Op*> queue_;
    std::vector<Op*> completed_;
    bool stop_ = false;
#endif
};

// ------------------------------------------------------------------ pinned host slots
class Pinned {
public:
    explicit Pinned(std::uint64_t bytes) : bytes_(bytes) {
        check(cudaHostAlloc(&p_, bytes, cudaHostAllocPortable | cudaHostAllocMapped), "cudaHostAlloc");
        std::memset(p_, 0, bytes);
    }
    ~Pinned() {
        if (p_ != nullptr) { (void)cudaFreeHost(p_); }
    }
    Pinned(const Pinned&)            = delete;
    Pinned& operator=(const Pinned&) = delete;
    [[nodiscard]] std::byte* data() const { return static_cast<std::byte*>(p_); }
    [[nodiscard]] std::uint64_t bytes() const { return bytes_; }

private:
    void* p_ = nullptr;
    std::uint64_t bytes_ = 0;
};

// ------------------------------------------------------------------ measurements
struct Probe {
    Records& records;
    DirectReads& io;
    std::mt19937_64 rng;
    double seconds = 5;
};

// A stream of reads, each one request of `pieces` I/Os into its slot of `slots`.
struct Stream {
    enum class Kind { Whole, Quarters, SubRead };
    Kind kind      = Kind::Whole;
    int depth      = 1;
    std::byte* base = nullptr; // depth slots of kRecord bytes
    // per slot
    std::vector<int> pending;
    std::vector<Clock::time_point> start;
    std::vector<Op> ops; // depth * kMaxPieces
    std::vector<double> latency_ms;
    std::uint64_t bytes = 0;
    int in_flight       = 0;
    int id              = 0;
};

void issue(Probe& probe, Stream& s, int slot, std::vector<Piece>& scratch) {
    std::uniform_int_distribution<std::size_t> pick(0, probe.records.count() - 1);
    const std::size_t record = pick(probe.rng);
    scratch.clear();
    std::uint64_t request = kRecord;
    if (s.kind == Stream::Kind::Whole) {
        probe.records.pieces(record, 0, kRecord, scratch);
    } else if (s.kind == Stream::Kind::Quarters) {
        const std::uint64_t pages = kRecord / kPage, quarter = (pages + 3) / 4;
        for (std::uint64_t q = 0; q < 4; ++q) {
            const std::uint64_t first = q * quarter, last = std::min(pages, first + quarter);
            probe.records.pieces(record, first * kPage, (last - first) * kPage, scratch);
        }
    } else {
        const std::uint64_t parts = (kRecord + kSubRead - 1) / kSubRead;
        std::uniform_int_distribution<std::uint64_t> part(0, parts - 1);
        const std::uint64_t begin = part(probe.rng) * kSubRead;
        request                   = std::min(kSubRead, kRecord - begin);
        probe.records.pieces(record, begin, request, scratch);
        for (auto& p : scratch) { p.at -= begin; }
    }
    if (scratch.size() > static_cast<std::size_t>(kMaxPieces)) { throw std::logic_error("too many pieces"); }
    s.pending[static_cast<std::size_t>(slot)] = static_cast<int>(scratch.size());
    s.start[static_cast<std::size_t>(slot)]   = Clock::now();
    s.bytes += request;
    ++s.in_flight;
    for (std::size_t i = 0; i < scratch.size(); ++i) {
        Op& op    = s.ops[static_cast<std::size_t>(slot) * kMaxPieces + i];
        op.file   = scratch[i].file;
        op.offset = scratch[i].offset;
        op.bytes  = static_cast<std::uint32_t>(scratch[i].bytes);
        op.dest   = s.base + static_cast<std::uint64_t>(slot) * kRecord + scratch[i].at;
        op.owner  = s.id * 1024 + slot;
        probe.io.submit(&op);
    }
}

Stream make_stream(Stream::Kind kind, int depth, std::byte* base, int id) {
    Stream s;
    s.kind = kind;
    s.depth = depth;
    s.base = base;
    s.id = id;
    s.pending.assign(static_cast<std::size_t>(depth), 0);
    s.start.assign(static_cast<std::size_t>(depth), Clock::now());
    s.ops.resize(static_cast<std::size_t>(depth) * kMaxPieces);
    return s;
}

// Runs the streams together for probe.seconds; returns the elapsed seconds.
double run(Probe& probe, std::vector<Stream*> streams) {
    std::vector<Piece> scratch;
    const auto begin    = Clock::now();
    const auto deadline = begin + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(probe.seconds));
    for (auto* s : streams) {
        for (int slot = 0; slot < s->depth; ++slot) { issue(probe, *s, slot, scratch); }
    }
    std::vector<Op*> done;
    int failures = 0;
    while (true) {
        int in_flight = 0;
        for (auto* s : streams) { in_flight += s->in_flight; }
        if (in_flight == 0) { break; }
        done.clear();
        probe.io.wait(done);
        const auto now = Clock::now();
        for (Op* op : done) {
            failures += op->failed ? 1 : 0;
            Stream& s      = *streams[static_cast<std::size_t>(op->owner / 1024)];
            const int slot = op->owner % 1024;
            if (--s.pending[static_cast<std::size_t>(slot)] != 0) { continue; }
            s.latency_ms.push_back(std::chrono::duration<double, std::milli>(now - s.start[static_cast<std::size_t>(slot)]).count());
            --s.in_flight;
            if (now < deadline) { issue(probe, s, slot, scratch); }
        }
    }
    if (failures != 0) { throw std::runtime_error(std::to_string(failures) + " reads failed or were short"); }
    return std::chrono::duration<double>(Clock::now() - begin).count();
}

void report(const char* label, const Stream& s, double elapsed) {
    const Stats st = stats(s.latency_ms);
    std::printf("  %-40s n %6zu  %6.2f GB/s  %7.1f req/s   ms p50 %6.3f  p90 %6.3f  p99 %6.3f  max %7.3f  mean %6.3f\n",
                label, st.n, static_cast<double>(s.bytes) / elapsed / 1e9, static_cast<double>(st.n) / elapsed, st.p50,
                st.p90, st.p99, st.max, st.mean);
}

void section_reads(Probe& probe, const Pinned& slots, const std::vector<int>& depths, Stream::Kind kind,
                   const char* name) {
    std::printf("\n== RM0e: %s reads from the artifact, %.1f s per queue depth (records in flight)\n", name,
                probe.seconds);
    for (const int depth : depths) {
        Stream s = make_stream(kind, depth, slots.data(), 0);
        const double elapsed = run(probe, {&s});
        char label[64];
        std::snprintf(label, sizeof(label), "%s QD %d", name, depth);
        report(label, s, elapsed);
    }
}

// Streams H2D copies from pinned memory on a second thread while it lives.
class H2dLoad {
public:
    H2dLoad() {
        worker_ = std::thread([this] {
            try {
                check(cudaSetDevice(0), "cudaSetDevice");
                constexpr std::size_t kCopy = 64ULL << 20;
                void* src = nullptr;
                void* dst = nullptr;
                cudaStream_t stream = nullptr;
                check(cudaHostAlloc(&src, kCopy, cudaHostAllocPortable), "H2D source");
                check(cudaMalloc(&dst, 4 * kCopy), "H2D destination");
                check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
                ready_ = true;
                const auto start = Clock::now();
                while (!stop_.load(std::memory_order_relaxed)) {
                    for (int i = 0; i < 4; ++i) {
                        check(cudaMemcpyAsync(static_cast<std::byte*>(dst) + i * kCopy, src, kCopy,
                                              cudaMemcpyHostToDevice, stream),
                              "H2D copy");
                    }
                    check(cudaStreamSynchronize(stream), "H2D sync");
                    bytes_ += 4 * kCopy;
                }
                seconds_ = std::chrono::duration<double>(Clock::now() - start).count();
                (void)cudaStreamDestroy(stream);
                (void)cudaFree(dst);
                (void)cudaFreeHost(src);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "H2D load: %s\n", e.what());
                ready_ = true;
            }
        });
        while (!ready_.load()) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
    }
    // Stops the copies and returns their GB/s.
    double stop() {
        stop_ = true;
        if (worker_.joinable()) { worker_.join(); }
        return seconds_ > 0 ? static_cast<double>(bytes_.load()) / seconds_ / 1e9 : 0.0;
    }
    ~H2dLoad() { (void)stop(); }
    H2dLoad(const H2dLoad&)            = delete;
    H2dLoad& operator=(const H2dLoad&) = delete;

private:
    std::thread worker_;
    std::atomic<bool> ready_{false}, stop_{false};
    std::atomic<std::uint64_t> bytes_{0};
    double seconds_ = 0;
};

void section_load(Probe& probe, const Pinned& slots) {
    std::printf("\n== RM0e: reads with and without a concurrent H2D stream from pinned memory\n");
    {
        H2dLoad load;
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
        std::printf("  H2D alone: %.2f GB/s\n", load.stop());
    }
    struct Case {
        Stream::Kind kind;
        int depth;
        const char* name;
    };
    for (const Case c : {Case{Stream::Kind::Whole, 1, "whole QD 1"}, Case{Stream::Kind::Whole, 4, "whole QD 4"},
                         Case{Stream::Kind::Quarters, 1, "quarters QD 1"}}) {
        Stream alone = make_stream(c.kind, c.depth, slots.data(), 0);
        const double t_alone = run(probe, {&alone});
        Stream loaded = make_stream(c.kind, c.depth, slots.data(), 0);
        double t_loaded = 0, h2d = 0;
        {
            H2dLoad load;
            t_loaded = run(probe, {&loaded});
            h2d      = load.stop();
        }
        char label[96];
        std::snprintf(label, sizeof(label), "%s, alone", c.name);
        report(label, alone, t_alone);
        std::snprintf(label, sizeof(label), "%s, with H2D (%.1f GB/s)", c.name, h2d);
        report(label, loaded, t_loaded);
    }
}

void section_subread(Probe& probe, const Pinned& slots, const Pinned& sub) {
    std::printf("\n== RM0e: QD 1 demand reads alone and with one 256 KiB prefetch sub-read always in flight\n");
    Stream alone = make_stream(Stream::Kind::Whole, 1, slots.data(), 0);
    const double t_alone = run(probe, {&alone});
    report("demand, alone", alone, t_alone);
    Stream demand = make_stream(Stream::Kind::Whole, 1, slots.data(), 0);
    Stream prefetch = make_stream(Stream::Kind::SubRead, 1, sub.data(), 1);
    const double t_both = run(probe, {&demand, &prefetch});
    report("demand, with a sub-read in flight", demand, t_both);
    report("256 KiB sub-reads beside the demand", prefetch, t_both);
    Stream subs = make_stream(Stream::Kind::SubRead, 1, sub.data(), 0);
    const double t_subs = run(probe, {&subs});
    report("256 KiB sub-reads, alone", subs, t_subs);
}

void section_straddlers(Probe& probe, const Pinned& slots) {
    std::printf("\n== records split across part files: unbuffered pieces against a buffered read\n");
    std::vector<Piece> pieces;
    std::vector<std::byte> expected(kRecord);
    for (const std::size_t record : probe.records.straddlers()) {
        pieces.clear();
        probe.records.pieces(record, 0, kRecord, pieces);
        std::vector<Op> ops(pieces.size());
        std::memset(slots.data(), 0xA5, kRecord);
        for (std::size_t i = 0; i < pieces.size(); ++i) {
            ops[i].file   = pieces[i].file;
            ops[i].offset = pieces[i].offset;
            ops[i].bytes  = static_cast<std::uint32_t>(pieces[i].bytes);
            ops[i].dest   = slots.data() + pieces[i].at;
            probe.io.submit(&ops[i]);
        }
        std::vector<Op*> done;
        while (done.size() < ops.size()) { probe.io.wait(done); }
        bool failed = false;
        for (const Op* op : done) { failed = failed || op->failed; }
        probe.records.reader().read_into(probe.records.logical(record), expected);
        const bool same = !failed && std::memcmp(expected.data(), slots.data(), kRecord) == 0;
        std::printf("  record %zu: %zu pieces (", record, pieces.size());
        for (std::size_t i = 0; i < pieces.size(); ++i) {
            std::printf("%sfile %zu +%llu B", i ? ", " : "", pieces[i].file,
                        static_cast<unsigned long long>(pieces[i].bytes));
        }
        std::printf("): %s\n", same ? "bytes equal" : "MISMATCH");
    }
}

// RM0g; skipped when the machine has too little available RAM to run it.
void section_pin(std::uint64_t chunks) {
    const std::uint64_t total = chunks * kChunkBytes;
    std::printf("\n== RM0g: cudaHostAlloc(Portable | Mapped) of %llu x %.2f GiB chunks against one %.2f GiB block\n",
                static_cast<unsigned long long>(chunks), static_cast<double>(kChunkBytes) / kGiB,
                static_cast<double>(total) / kGiB);
    const std::uint64_t avail = available_ram();
    std::printf("  available RAM %.1f GiB\n", static_cast<double>(avail) / kGiB);
    if (avail < total + 8 * kGiB) {
        std::printf("  skipped: needs %.1f GiB available\n", static_cast<double>(total + 8 * kGiB) / kGiB);
        return;
    }
    const auto touch = [](void* p, std::uint64_t bytes) {
        const auto t = Clock::now();
        std::memset(p, 1, bytes);
        return ms_since(t);
    };
    const auto chunked = [&](const char* label) {
        std::vector<void*> blocks;
        std::vector<double> alloc;
        double touched = 0, freed = 0;
        const std::uint64_t before = available_ram();
        for (std::uint64_t i = 0; i < chunks; ++i) {
            void* p      = nullptr;
            const auto t = Clock::now();
            check(cudaHostAlloc(&p, kChunkBytes, cudaHostAllocPortable | cudaHostAllocMapped), "cudaHostAlloc chunk");
            alloc.push_back(ms_since(t));
            blocks.push_back(p);
        }
        const std::uint64_t after = available_ram();
        for (void* p : blocks) { touched += touch(p, kChunkBytes); }
        for (void* p : blocks) {
            const auto t = Clock::now();
            check(cudaFreeHost(p), "cudaFreeHost");
            freed += ms_since(t);
        }
        double sum = 0;
        for (double a : alloc) { sum += a; }
        const Stats s = stats(alloc);
        std::printf("  %-22s alloc %8.1f ms total (per chunk p50 %.1f, max %.1f ms; %.2f GiB/s); touch %7.1f ms; "
                    "free %7.1f ms; available RAM fell %.1f GiB\n",
                    label, sum, s.p50, s.max, static_cast<double>(total) / kGiB / (sum / 1000), touched, freed,
                    (static_cast<double>(before) - static_cast<double>(after)) / kGiB);
    };
    const auto block = [&](const char* label) {
        void* p                    = nullptr;
        const std::uint64_t before = available_ram();
        const auto t               = Clock::now();
        check(cudaHostAlloc(&p, total, cudaHostAllocPortable | cudaHostAllocMapped), "cudaHostAlloc block");
        const double alloc = ms_since(t);
        const std::uint64_t after = available_ram();
        const double touched      = touch(p, total);
        const auto f              = Clock::now();
        check(cudaFreeHost(p), "cudaFreeHost");
        std::printf("  %-22s alloc %8.1f ms (%.2f GiB/s); touch %7.1f ms; free %7.1f ms; available RAM fell %.1f GiB\n",
                    label, alloc, static_cast<double>(total) / kGiB / (alloc / 1000), touched, ms_since(f),
                    (static_cast<double>(before) - static_cast<double>(after)) / kGiB);
    };
    chunked("chunks (first)");
    block("one block");
    chunked("chunks (again)");
}

void section_pinlimit(std::uint64_t cap_gib) {
    std::printf("\n== pinned-memory limit: 1.32 GiB chunks until failure, %llu GiB, or < 4 GiB available\n",
                static_cast<unsigned long long>(cap_gib));
#ifndef _WIN32
    rlimit limit{};
    if (getrlimit(RLIMIT_MEMLOCK, &limit) == 0) {
        std::printf("  RLIMIT_MEMLOCK soft %llu, hard %llu\n", static_cast<unsigned long long>(limit.rlim_cur),
                    static_cast<unsigned long long>(limit.rlim_max));
    }
#endif
    std::vector<void*> blocks;
    std::string stop = "cap reached";
    while (static_cast<std::uint64_t>(blocks.size() + 1) * kChunkBytes <= cap_gib * kGiB) {
        if (available_ram() < 4 * kGiB + kChunkBytes) {
            stop = "available RAM below 4 GiB";
            break;
        }
        void* p = nullptr;
        const cudaError_t e = cudaHostAlloc(&p, kChunkBytes, cudaHostAllocPortable | cudaHostAllocMapped);
        if (e != cudaSuccess) {
            (void)cudaGetLastError();
            stop = std::string("cudaHostAlloc failed: ") + cudaGetErrorString(e);
            break;
        }
        blocks.push_back(p);
    }
    std::printf("  pinned %zu chunks = %.2f GiB (%s)\n", blocks.size(),
                static_cast<double>(blocks.size() * kChunkBytes) / kGiB, stop.c_str());
    for (void* p : blocks) { (void)cudaFreeHost(p); }
}

std::vector<int> parse_list(const std::string& text) {
    std::vector<int> out;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (!item.empty()) { out.push_back(std::stoi(item)); }
    }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::fprintf(stderr,
                         "usage: ninfer_expert_read_probe ARTIFACT.ninfer [--seconds S] [--depths 1,2,4,8] "
                         "[--quarter-depths 1,2,4] [--only reads,quarters,load,subread,straddlers,pin,pinlimit] "
                         "[--pin-chunks N] [--pin-limit-gib N] [--seed N]\n");
            return 2;
        }
        const std::filesystem::path artifact = argv[1];
        double seconds = 5;
        std::vector<int> depths = {1, 2, 4, 8}, quarter_depths = {1, 2, 4};
        std::set<std::string> only;
        std::uint64_t pin_chunks = 10, pin_limit_gib = 0, seed = 2026;
        for (int i = 2; i < argc; ++i) {
            const std::string arg = argv[i];
            const auto value      = [&]() -> std::string {
                if (i + 1 >= argc) { throw std::invalid_argument(arg + " needs a value"); }
                return argv[++i];
            };
            if (arg == "--seconds") {
                seconds = std::stod(value());
            } else if (arg == "--depths") {
                depths = parse_list(value());
            } else if (arg == "--quarter-depths") {
                quarter_depths = parse_list(value());
            } else if (arg == "--only") {
                std::stringstream stream(value());
                std::string item;
                while (std::getline(stream, item, ',')) { only.insert(item); }
            } else if (arg == "--pin-chunks") {
                pin_chunks = std::stoull(value());
            } else if (arg == "--pin-limit-gib") {
                pin_limit_gib = std::stoull(value());
            } else if (arg == "--seed") {
                seed = std::stoull(value());
            } else {
                throw std::invalid_argument("unknown argument " + arg);
            }
        }
        for (const int d : depths) {
            if (d < 1 || d > 64) { throw std::invalid_argument("queue depths must be in [1,64]"); }
        }
        for (const int d : quarter_depths) {
            if (d < 1 || d > 16) { throw std::invalid_argument("quarter depths must be in [1,16]"); }
        }
        const auto wanted = [&](const char* name) {
            return only.empty() ? std::string(name) != "pinlimit" : only.count(name) != 0;
        };
        check(cudaSetDevice(0), "cudaSetDevice");
        check(cudaFree(nullptr), "context");

        Records records(artifact);
        int max_depth = 1;
        for (const int d : depths) { max_depth = std::max(max_depth, d); }
        for (const int d : quarter_depths) { max_depth = std::max(max_depth, d); }
        DirectReads io(records.paths(), static_cast<std::size_t>(max_depth * kMaxPieces + kMaxPieces));
        Pinned slots(static_cast<std::uint64_t>(max_depth) * kRecord);
        Pinned sub(kRecord);
        Probe probe{records, io, std::mt19937_64(seed), seconds};
        std::printf("pinned slots: %d x %llu B (cudaHostAlloc Portable | Mapped); available RAM %.1f GiB\n", max_depth,
                    static_cast<unsigned long long>(kRecord), static_cast<double>(available_ram()) / kGiB);

        if (wanted("straddlers")) { section_straddlers(probe, slots); }
        if (wanted("reads")) { section_reads(probe, slots, depths, Stream::Kind::Whole, "whole"); }
        if (wanted("quarters")) { section_reads(probe, slots, quarter_depths, Stream::Kind::Quarters, "quarters"); }
        if (wanted("load")) { section_load(probe, slots); }
        if (wanted("subread")) { section_subread(probe, slots, sub); }
        if (wanted("pin")) { section_pin(pin_chunks); }
        if (wanted("pinlimit")) {
            if (pin_limit_gib == 0) {
                std::printf("\npinlimit needs --pin-limit-gib N\n");
            } else {
                section_pinlimit(pin_limit_gib);
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
