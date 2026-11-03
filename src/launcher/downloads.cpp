#include "launcher/downloads.h"

#include "launcher/sha256.h"

#include <curl/curl.h>

#include <algorithm>
#include <chrono>
#include <random>
#include <stdexcept>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <winioctl.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace infernix::launcher {
namespace {

constexpr std::uint64_t kChunkBytes = 64ull << 20;
constexpr int kAttempts             = 6;

std::string new_id() {
    std::random_device device;
    std::mt19937_64 generator(device() ^ static_cast<std::uint64_t>(now_ms()));
    const std::uint64_t value = generator();
    return hex_string(reinterpret_cast<const std::uint8_t*>(&value), sizeof(value));
}

bool safe_relative(const std::string& path) {
    if (path.empty() || path.front() == '/' || path.front() == '\\' || path.find(':') != std::string::npos) {
        return false;
    }
    std::size_t start = 0;
    while (start <= path.size()) {
        const std::size_t end = path.find_first_of("/\\", start);
        const std::string part = path.substr(start, end == std::string::npos ? std::string::npos : end - start);
        if (part == "..") { return false; }
        if (end == std::string::npos) { break; }
        start = end + 1;
    }
    return true;
}

// A file written at positions by several connections at once.
class PartFile {
public:
    explicit PartFile(const std::filesystem::path& path) {
#ifdef _WIN32
        handle_ = CreateFileW(path.wstring().c_str(), GENERIC_WRITE | GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                              nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (handle_ == INVALID_HANDLE_VALUE) { handle_ = nullptr; }
#else
        fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
#endif
    }
    ~PartFile() {
#ifdef _WIN32
        if (handle_ != nullptr) { CloseHandle(handle_); }
#else
        if (fd_ >= 0) { ::close(fd_); }
#endif
    }
    PartFile(const PartFile&)            = delete;
    PartFile& operator=(const PartFile&) = delete;

    [[nodiscard]] bool ok() const {
#ifdef _WIN32
        return handle_ != nullptr;
#else
        return fd_ >= 0;
#endif
    }

    // Sizes the file. On NTFS it is made sparse first, so writes far into it do not zero-fill
    // the gap before them.
    bool resize(std::uint64_t size) {
#ifdef _WIN32
        DWORD returned = 0;
        DeviceIoControl(handle_, FSCTL_SET_SPARSE, nullptr, 0, nullptr, 0, &returned, nullptr);
        FILE_END_OF_FILE_INFO end{};
        end.EndOfFile.QuadPart = static_cast<LONGLONG>(size);
        return SetFileInformationByHandle(handle_, FileEndOfFileInfo, &end, sizeof(end)) != 0;
#else
        return ::ftruncate(fd_, static_cast<off_t>(size)) == 0;
#endif
    }

    bool write(std::uint64_t offset, const char* data, std::size_t size) {
        while (size > 0) {
#ifdef _WIN32
            OVERLAPPED position{};
            position.Offset     = static_cast<DWORD>(offset);
            position.OffsetHigh = static_cast<DWORD>(offset >> 32);
            DWORD written       = 0;
            const DWORD take    = static_cast<DWORD>(std::min<std::size_t>(size, 1u << 30));
            if (!WriteFile(handle_, data, take, &written, &position) || written == 0) { return false; }
#else
            const ssize_t written = ::pwrite(fd_, data, size, static_cast<off_t>(offset));
            if (written <= 0) { return false; }
#endif
            offset += static_cast<std::uint64_t>(written);
            data += written;
            size -= static_cast<std::size_t>(written);
        }
        return true;
    }

    // What is written so far, on the disk.
    bool flush() {
#ifdef _WIN32
        return FlushFileBuffers(handle_) != 0;
#else
        return ::fsync(fd_) == 0;
#endif
    }

    // Once every range is written the file needs no sparse flag.
    void finish() {
#ifdef _WIN32
        FILE_SET_SPARSE_BUFFER clear{};
        clear.SetSparse = FALSE;
        DWORD returned  = 0;
        DeviceIoControl(handle_, FSCTL_SET_SPARSE, &clear, sizeof(clear), nullptr, 0, &returned, nullptr);
        FlushFileBuffers(handle_);
#else
        ::fsync(fd_);
#endif
    }

private:
#ifdef _WIN32
    HANDLE handle_ = nullptr;
#else
    int fd_ = -1;
#endif
};

struct ChunkTransfer {
    PartFile* file           = nullptr;
    DownloadFile* progress   = nullptr;
    std::uint64_t offset     = 0;
    std::uint64_t expected   = 0;
    std::uint64_t written    = 0;
    bool whole_file          = false;
    bool status_checked      = false;
    bool bad_status          = false;
    const std::atomic<bool>* interrupt = nullptr;
    const std::atomic<bool>* abort     = nullptr;
    CURL* curl               = nullptr;
};

size_t write_chunk(char* data, size_t size, size_t count, void* user) {
    auto* transfer    = static_cast<ChunkTransfer*>(user);
    const size_t bytes = size * count;
    if (!transfer->status_checked) {
        long status = 0;
        curl_easy_getinfo(transfer->curl, CURLINFO_RESPONSE_CODE, &status);
        transfer->status_checked = true;
        // A range answers 206; only a request for the whole file may answer 200.
        transfer->bad_status = !(status == 206 || (status == 200 && transfer->whole_file));
    }
    if (transfer->bad_status || transfer->written + bytes > transfer->expected) { return 0; }
    if (!transfer->file->write(transfer->offset + transfer->written, data, bytes)) { return 0; }
    transfer->written += bytes;
    transfer->progress->done += bytes;
    return bytes;
}

int transfer_progress(void* user, curl_off_t, curl_off_t, curl_off_t, curl_off_t) {
    const auto* transfer = static_cast<const ChunkTransfer*>(user);
    return transfer->interrupt->load() || transfer->abort->load() ? 1 : 0;
}

Json read_meta(const std::filesystem::path& path) {
    const std::optional<std::string> text = read_file(path);
    if (!text) { return Json::object(); }
    try {
        return Json::parse(*text);
    } catch (const Json::exception&) { return Json::object(); }
}

} // namespace

Downloads::Downloads(Hooks hooks) : hooks_(std::move(hooks)) {
    load();
    worker_   = std::thread([this] { run(); });
    reporter_ = std::thread([this] {
        std::uint64_t previous = 0;
        std::string previous_id;
        auto last = std::chrono::steady_clock::now();
        int ticks = 0;
        std::unique_lock lock(mutex_);
        while (!stopping_) {
            cv_.wait_for(lock, std::chrono::seconds(1));
            if (stopping_) { break; }
            DownloadJob* job = running_id_.empty() ? nullptr : find_locked(running_id_);
            if (job == nullptr) {
                previous_id.clear();
                continue;
            }
            std::uint64_t done = 0;
            for (const auto& file : job->files) { done += file->done.load(); }
            const auto now       = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration<double>(now - last).count();
            last                 = now;
            if (previous_id == job->id && elapsed > 0.0 && done >= previous) {
                const double instant = static_cast<double>(done - previous) / elapsed;
                job->speed_bps       = job->speed_bps == 0.0 ? instant : 0.7 * job->speed_bps + 0.3 * instant;
            }
            previous    = done;
            previous_id = job->id;
            if (++ticks % 10 == 0) { save_locked(); }
            lock.unlock();
            notify();
            lock.lock();
        }
    });
}

Downloads::~Downloads() { shutdown(); }

void Downloads::shutdown() {
    {
        std::lock_guard lock(mutex_);
        if (stopping_) { return; }
        stopping_ = true;
    }
    interrupt_ = true;
    cv_.notify_all();
    if (worker_.joinable()) { worker_.join(); }
    if (reporter_.joinable()) { reporter_.join(); }
    std::lock_guard lock(mutex_);
    save_locked();
}

void Downloads::notify() {
    if (hooks_.changed) { hooks_.changed(list()); }
}

DownloadJob* Downloads::find_locked(const std::string& id) {
    for (const auto& job : jobs_) {
        if (job->id == id) { return job.get(); }
    }
    return nullptr;
}

Json Downloads::job_json_locked(const DownloadJob& job) const {
    Json files       = Json::array();
    std::uint64_t total = 0, done = 0;
    for (const auto& file : job.files) {
        const std::uint64_t file_done = std::min(file->done.load(), file->bytes);
        total += file->bytes;
        done += file_done;
        files.push_back({{"path", file->path},
                         {"bytes", file->bytes},
                         {"sha256", file->sha256 ? Json(*file->sha256) : Json(nullptr)},
                         {"done", file_done},
                         {"state", file->state},
                         {"error", file->error ? Json(*file->error) : Json(nullptr)}});
    }
    const bool running = job.state == "running";
    Json eta           = nullptr;
    if (running && job.speed_bps > 1.0 && total > done) {
        eta = static_cast<double>(total - done) / job.speed_bps;
    }
    return {{"id", job.id},
            {"repo", job.repo},
            {"revision", job.revision},
            {"target_dir", path_utf8(job.target_dir)},
            {"files", files},
            {"total", total},
            {"done", done},
            {"speed_bps", running ? job.speed_bps : 0.0},
            {"eta_s", eta},
            {"state", job.state},
            {"error", job.error ? Json(*job.error) : Json(nullptr)},
            {"created_at", job.created_at},
            {"finished_at", job.finished_at ? Json(*job.finished_at) : Json(nullptr)}};
}

Json Downloads::list() const {
    std::lock_guard lock(mutex_);
    Json out = Json::array();
    for (const auto& job : jobs_) { out.push_back(job_json_locked(*job)); }
    return out;
}

bool Downloads::active() const {
    std::lock_guard lock(mutex_);
    return std::any_of(jobs_.begin(), jobs_.end(), [](const auto& job) {
        return job->state == "running" || job->state == "queued" || job->state == "verifying";
    });
}

Json Downloads::create(const Json& request) {
    const std::string repo = json_or<std::string>(request, "repo", "");
    if (!valid_repo_id(repo)) { throw std::invalid_argument("not a Hugging Face repository id: " + repo); }
    const std::string dest = trim(json_or<std::string>(request, "dest_dir", ""));
    if (dest.empty()) { throw std::invalid_argument("choose a destination folder"); }
    const std::filesystem::path dest_dir = utf8_path(dest);
    if (!dest_dir.is_absolute()) { throw std::invalid_argument("the destination must be an absolute path"); }
    if (!request.contains("files") || !request["files"].is_array() || request["files"].empty()) {
        throw std::invalid_argument("select at least one file");
    }
    const HfListing listing = hf_list_files(repo, hooks_.token ? hooks_.token() : std::string{});
    auto job        = std::make_unique<DownloadJob>();
    job->id         = new_id();
    job->repo       = repo;
    job->revision   = listing.revision;
    job->dest_dir   = dest_dir;
    job->target_dir = json_or<bool>(request, "subfolder", true)
                          ? dest_dir / utf8_path(repo.substr(repo.find('/') + 1))
                          : dest_dir;
    job->created_at = now_ms();
    job->replace    = json_or<bool>(request, "replace", false);
    std::uint64_t needed = 0;
    for (const Json& entry : request["files"]) {
        if (!entry.is_string()) { throw std::invalid_argument("file names must be strings"); }
        const std::string path = entry.get<std::string>();
        if (!safe_relative(path)) { throw std::invalid_argument("unsafe file path: " + path); }
        const auto it = std::find_if(listing.files.begin(), listing.files.end(),
                                     [&](const HfFile& file) { return file.path == path; });
        if (it == listing.files.end()) { throw std::invalid_argument(path + " is not in " + repo); }
        auto file    = std::make_unique<DownloadFile>();
        file->path   = path;
        file->bytes    = it->size;
        file->sha256   = it->sha256;
        file->git_blob = it->git_blob;
        // What still has to arrive: not a file already there whole (unless replacing it), less what
        // an interrupted download left.
        const std::filesystem::path final_path = job->target_dir / utf8_path(path);
        std::error_code error;
        const bool present = std::filesystem::file_size(final_path, error) == it->size && !error;
        if (job->replace || !present) {
            // A resumable partial file has its finished chunks on disk already.
            std::filesystem::path meta = final_path;
            meta += ".part.json";
            const Json chunks       = read_meta(meta).value("done", Json::array());
            const std::uint64_t got = std::min<std::uint64_t>(chunks.size() * kChunkBytes, it->size);
            needed += it->size - got;
        }
        job->files.push_back(std::move(file));
    }
    // Room on the destination's drive (its nearest existing folder), with 1 GiB to spare.
    std::filesystem::path probe = dest_dir;
    std::error_code space_error;
    while (!probe.empty() && !std::filesystem::exists(probe, space_error) && probe.has_parent_path() &&
           probe.parent_path() != probe) {
        probe = probe.parent_path();
    }
    const std::filesystem::space_info space = std::filesystem::space(probe, space_error);
    if (!space_error && space.available < needed + (1ull << 30)) {
        throw std::invalid_argument("not enough free space on " + path_utf8(probe.root_path()) + ": the download needs " +
                                    format_bytes(needed) + " (and 1 GiB to spare), " + format_bytes(space.available) +
                                    " is free");
    }
    Json created;
    {
        std::lock_guard lock(mutex_);
        created = job_json_locked(*job);
        jobs_.push_back(std::move(job));
        save_locked();
    }
    cv_.notify_all();
    notify();
    return created;
}

void Downloads::pause(const std::string& id) {
    {
        std::lock_guard lock(mutex_);
        DownloadJob* job = find_locked(id);
        if (job == nullptr || (job->state != "queued" && job->state != "running" && job->state != "verifying")) { return; }
        job->state = "paused";
        if (running_id_ == id) { interrupt_ = true; }
        save_locked();
    }
    cv_.notify_all();
    notify();
}

void Downloads::resume(const std::string& id) {
    {
        std::lock_guard lock(mutex_);
        DownloadJob* job = find_locked(id);
        if (job == nullptr || (job->state != "paused" && job->state != "failed")) { return; }
        job->state = "queued";
        job->error.reset();
        for (const auto& file : job->files) {
            if (file->state == "failed" || file->state == "paused") {
                file->state = "queued";
                file->error.reset();
            }
        }
        save_locked();
    }
    cv_.notify_all();
    notify();
}

void Downloads::cancel(const std::string& id) {
    std::vector<std::filesystem::path> leftovers;
    {
        std::lock_guard lock(mutex_);
        DownloadJob* job = find_locked(id);
        if (job == nullptr || job->state == "done" || job->state == "cancelled") { return; }
        const bool running = running_id_ == id;
        job->state         = "cancelled";
        job->finished_at   = now_ms();
        if (running) {
            interrupt_ = true; // the worker removes the partial files
        } else {
            for (const auto& file : job->files) {
                if (file->state == "done" || file->state == "skipped") { continue; }
                std::filesystem::path part = job->target_dir / utf8_path(file->path);
                part += ".part";
                leftovers.push_back(part);
            }
        }
        save_locked();
    }
    for (const std::filesystem::path& part : leftovers) {
        std::error_code ignored;
        std::filesystem::remove(part, ignored);
        std::filesystem::path meta = part;
        meta += ".json";
        std::filesystem::remove(meta, ignored);
    }
    cv_.notify_all();
    notify();
}

void Downloads::remove(const std::string& id) {
    {
        std::lock_guard lock(mutex_);
        DownloadJob* job = find_locked(id);
        // A cancelled job stays the worker's until it has stopped and removed its partial files.
        if (job == nullptr || running_id_ == id || job->state == "running" || job->state == "verifying" ||
            job->state == "queued" || job->state == "paused") {
            return;
        }
        jobs_.erase(std::remove_if(jobs_.begin(), jobs_.end(), [&](const auto& j) { return j->id == id; }), jobs_.end());
        save_locked();
    }
    notify();
}

void Downloads::run() {
    std::unique_lock lock(mutex_);
    while (!stopping_) {
        DownloadJob* next = nullptr;
        for (const auto& job : jobs_) {
            if (job->state == "queued") {
                next = job.get();
                break;
            }
        }
        if (next == nullptr) {
            cv_.wait(lock);
            continue;
        }
        running_id_     = next->id;
        next->state     = "running";
        next->error.reset();
        next->speed_bps = 0.0;
        interrupt_      = false;
        lock.unlock();
        notify();
        run_job(*next);
        lock.lock();
        running_id_.clear();
        if (next->state == "cancelled") {
            for (const auto& file : next->files) {
                if (file->state == "done" || file->state == "skipped") { continue; }
                std::error_code ignored;
                std::filesystem::path part = next->target_dir / utf8_path(file->path);
                part += ".part";
                std::filesystem::remove(part, ignored);
                std::filesystem::path meta = part;
                meta += ".json";
                std::filesystem::remove(meta, ignored);
            }
        }
        save_locked();
        lock.unlock();
        notify();
        lock.lock();
    }
}

void Downloads::run_job(DownloadJob& job) {
    const auto fail = [&](DownloadFile* file, const std::string& message) {
        {
            std::lock_guard lock(mutex_);
            if (file != nullptr) {
                file->state = "failed";
                file->error = message;
            }
            job.state = "failed";
            job.error = message;
        }
        if (hooks_.failed) { hooks_.failed(job.repo, message); }
    };
    std::error_code error;
    std::filesystem::create_directories(job.target_dir, error);
    if (error) {
        fail(nullptr, "cannot create " + path_utf8(job.target_dir) + ": " + error.message());
        return;
    }
    for (const auto& file_ptr : job.files) {
        DownloadFile& file = *file_ptr;
        if (file.state == "done" || file.state == "skipped") { continue; }
        if (interrupt_) { break; }
        const std::filesystem::path final_path = job.target_dir / utf8_path(file.path);
        std::filesystem::create_directories(final_path.parent_path(), error);
        std::error_code size_error;
        if (!job.replace && std::filesystem::is_regular_file(final_path, size_error) &&
            std::filesystem::file_size(final_path, size_error) == file.bytes) {
            std::lock_guard lock(mutex_);
            file.done  = file.bytes;
            file.state = "skipped";
            continue;
        }
        {
            std::lock_guard lock(mutex_);
            file.state = "downloading";
            file.error.reset();
        }
        notify();
        const bool complete = download_file(job, file, final_path);
        if (interrupt_) {
            std::lock_guard lock(mutex_);
            file.state = "paused";
            break;
        }
        if (!complete) {
            fail(&file, file.error.value_or("download failed"));
            return;
        }
        std::filesystem::path part = final_path;
        part += ".part";
        if ((file.sha256 || file.git_blob) && (!hooks_.verify || hooks_.verify())) {
            {
                std::lock_guard lock(mutex_);
                file.state = "verifying";
                job.state  = "verifying";
            }
            notify();
            const std::optional<std::string> digest =
                file.sha256 ? sha256_file(part, nullptr, interrupt_) : git_blob_id_file(part, interrupt_);
            const std::string& expected = file.sha256 ? *file.sha256 : *file.git_blob;
            if (interrupt_) {
                std::lock_guard lock(mutex_);
                file.state = "paused";
                break;
            }
            {
                std::lock_guard lock(mutex_);
                job.state = "running";
            }
            if (!digest || *digest != expected) {
                std::filesystem::remove(part, error);
                std::filesystem::path meta = part;
                meta += ".json";
                std::filesystem::remove(meta, error);
                file.done = 0;
                fail(&file, digest ? "checksum mismatch: the file was deleted; resume to download it again"
                                   : "cannot read the downloaded file to check it");
                return;
            }
        }
        std::filesystem::rename(part, final_path, error);
        if (error) {
            fail(&file, "cannot rename " + path_utf8(part) + ": " + error.message());
            return;
        }
        std::filesystem::path meta = part;
        meta += ".json";
        std::filesystem::remove(meta, error);
        {
            std::lock_guard lock(mutex_);
            file.state = "done";
            file.done  = file.bytes;
        }
        notify();
    }
    bool finished = false;
    {
        std::lock_guard lock(mutex_);
        if (job.state == "running") {
            job.state       = "done";
            job.finished_at = now_ms();
            finished        = true;
        }
    }
    if (finished && hooks_.finished) { hooks_.finished(job.dest_dir, job.target_dir, job.repo); }
}

bool Downloads::download_file(DownloadJob& job, DownloadFile& file, const std::filesystem::path& final_path) {
    std::filesystem::path part = final_path;
    part += ".part";
    std::filesystem::path meta_path = part;
    meta_path += ".json";
    const std::uint64_t size   = file.bytes;
    const std::uint64_t chunks = size == 0 ? 0 : (size + kChunkBytes - 1) / kChunkBytes;

    // Resume from the recorded chunks when the partial file belongs to the same revision.
    std::vector<char> done(chunks, 0);
    std::error_code error;
    const Json meta = read_meta(meta_path);
    const bool resumable = std::filesystem::is_regular_file(part, error) &&
                           std::filesystem::file_size(part, error) == size &&
                           json_or<std::uint64_t>(meta, "size", 0) == size &&
                           json_or<std::uint64_t>(meta, "chunk", 0) == kChunkBytes &&
                           json_or<std::string>(meta, "revision", "") == job.revision;
    if (resumable && meta.contains("done") && meta["done"].is_array()) {
        for (const Json& index : meta["done"]) {
            if (index.is_number_unsigned() && index.get<std::uint64_t>() < chunks) { done[index.get<std::uint64_t>()] = 1; }
        }
    } else {
        std::filesystem::remove(part, error);
    }
    PartFile out(part);
    if (!out.ok() || (!resumable && !out.resize(size))) {
        file.error = "cannot write " + path_utf8(part);
        return false;
    }
    std::uint64_t already = 0;
    for (std::uint64_t i = 0; i < chunks; ++i) {
        if (done[i]) { already += std::min(kChunkBytes, size - i * kChunkBytes); }
    }
    file.done = already;

    std::mutex meta_mutex;
    std::mutex save_mutex; // one record write at a time
    auto last_meta_save = std::chrono::steady_clock::now();
    const auto save_meta = [&](bool force) {
        Json indices = Json::array();
        {
            std::lock_guard lock(meta_mutex);
            const auto now = std::chrono::steady_clock::now();
            if (!force && now - last_meta_save < std::chrono::seconds(3)) { return; }
            last_meta_save = now;
            for (std::uint64_t i = 0; i < chunks; ++i) {
                if (done[i]) { indices.push_back(i); }
            }
        }
        // The record names only chunks already on the disk: after a power cut it must not call
        // the zeros of a preallocated file done. (An older record written last names fewer.)
        std::lock_guard save(save_mutex);
        if (!out.flush()) { return; }
        try {
            write_file_atomic(meta_path, Json{{"size", size}, {"chunk", kChunkBytes}, {"revision", job.revision},
                                              {"done", indices}}
                                             .dump());
        } catch (const std::exception&) {}
    };

    const std::string url   = hf_resolve_url(job.repo, job.revision, file.path);
    const std::string token = hooks_.token ? hooks_.token() : std::string{};
    std::atomic<std::uint64_t> next{0};
    std::atomic<bool> abort{false};
    std::mutex error_mutex;
    std::string failure;

    const auto worker = [&] {
        std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> curl(curl_easy_init(), &curl_easy_cleanup);
        if (!curl) {
            abort = true;
            return;
        }
        curl_slist* headers = nullptr;
        if (!token.empty()) { headers = curl_slist_append(headers, ("Authorization: Bearer " + token).c_str()); }
        for (;;) {
            const std::uint64_t index = next++;
            if (index >= chunks || abort || interrupt_) { break; }
            {
                std::lock_guard lock(meta_mutex);
                if (done[index]) { continue; }
            }
            const std::uint64_t begin = index * kChunkBytes;
            const std::uint64_t bytes = std::min(kChunkBytes, size - begin);
            bool ok                   = false;
            std::string last_error;
            for (int attempt = 0; attempt < kAttempts && !ok && !abort && !interrupt_; ++attempt) {
                if (attempt > 0) {
                    for (int waited = 0; waited < (1 << attempt) * 10 && !abort && !interrupt_; ++waited) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    }
                }
                ChunkTransfer transfer;
                transfer.file       = &out;
                transfer.progress   = &file;
                transfer.offset     = begin;
                transfer.expected   = bytes;
                transfer.whole_file = begin == 0 && bytes == size;
                transfer.interrupt  = &interrupt_;
                transfer.abort      = &abort;
                transfer.curl       = curl.get();
                const std::string range = std::to_string(begin) + "-" + std::to_string(begin + bytes - 1);
                curl_easy_reset(curl.get());
                curl_easy_setopt(curl.get(), CURLOPT_URL, url.c_str());
                curl_easy_setopt(curl.get(), CURLOPT_HTTPHEADER, headers);
                curl_easy_setopt(curl.get(), CURLOPT_RANGE, range.c_str());
                curl_easy_setopt(curl.get(), CURLOPT_FOLLOWLOCATION, 1L);
                curl_easy_setopt(curl.get(), CURLOPT_MAXREDIRS, 10L);
                curl_easy_setopt(curl.get(), CURLOPT_PROTOCOLS_STR, "https");
                curl_easy_setopt(curl.get(), CURLOPT_REDIR_PROTOCOLS_STR, "https");
                curl_easy_setopt(curl.get(), CURLOPT_CONNECTTIMEOUT, 30L);
                curl_easy_setopt(curl.get(), CURLOPT_LOW_SPEED_LIMIT, 16384L);
                curl_easy_setopt(curl.get(), CURLOPT_LOW_SPEED_TIME, 60L);
                curl_easy_setopt(curl.get(), CURLOPT_NOSIGNAL, 1L);
                curl_easy_setopt(curl.get(), CURLOPT_USERAGENT, "infernix-launcher");
#ifdef _WIN32
                curl_easy_setopt(curl.get(), CURLOPT_SSL_OPTIONS, static_cast<long>(CURLSSLOPT_NATIVE_CA));
#endif
                curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION, write_chunk);
                curl_easy_setopt(curl.get(), CURLOPT_WRITEDATA, &transfer);
                curl_easy_setopt(curl.get(), CURLOPT_NOPROGRESS, 0L);
                curl_easy_setopt(curl.get(), CURLOPT_XFERINFOFUNCTION, transfer_progress);
                curl_easy_setopt(curl.get(), CURLOPT_XFERINFODATA, &transfer);
                const CURLcode code = curl_easy_perform(curl.get());
                long status         = 0;
                curl_easy_getinfo(curl.get(), CURLINFO_RESPONSE_CODE, &status);
                if (code == CURLE_OK && transfer.written == bytes && !transfer.bad_status) {
                    ok = true;
                    break;
                }
                // Discard the partial chunk; it is fetched again whole.
                file.done -= transfer.written;
                if (status == 401 || status == 403 || status == 404) {
                    last_error = "huggingface.co answered HTTP " + std::to_string(status) + " for " + file.path;
                    abort      = true;
                    break;
                }
                last_error = transfer.bad_status ? "unexpected HTTP " + std::to_string(status) + " for a range request"
                                                 : std::string(curl_easy_strerror(code));
            }
            if (ok) {
                {
                    std::lock_guard lock(meta_mutex);
                    done[index] = 1;
                }
                save_meta(false);
            } else if (!interrupt_) {
                std::lock_guard lock(error_mutex);
                if (failure.empty()) { failure = last_error.empty() ? "download failed" : last_error; }
                abort = true;
            }
        }
        curl_slist_free_all(headers);
    };

    const int connections = std::clamp(hooks_.connections ? hooks_.connections() : 8, 1, 32);
    std::uint64_t pending = 0;
    for (const char flag : done) { pending += flag == 0 ? 1 : 0; }
    const auto threads = static_cast<std::size_t>(std::min<std::uint64_t>(pending, static_cast<std::uint64_t>(connections)));
    std::vector<std::thread> pool;
    for (std::size_t i = 0; i < threads; ++i) { pool.emplace_back(worker); }
    for (std::thread& thread : pool) { thread.join(); }
    save_meta(true);
    const bool complete = std::all_of(done.begin(), done.end(), [](char flag) { return flag != 0; });
    if (complete) { out.finish(); }
    if (!complete && !interrupt_) { file.error = failure.empty() ? "download failed" : failure; }
    return complete && !interrupt_;
}

void Downloads::save_locked() const {
    Json jobs = Json::array();
    for (const auto& job : jobs_) {
        Json files = Json::array();
        for (const auto& file : job->files) {
            files.push_back({{"path", file->path},
                             {"bytes", file->bytes},
                             {"sha256", file->sha256 ? Json(*file->sha256) : Json(nullptr)},
                             {"git_blob", file->git_blob ? Json(*file->git_blob) : Json(nullptr)},
                             {"done", file->done.load()},
                             {"state", file->state},
                             {"error", file->error ? Json(*file->error) : Json(nullptr)}});
        }
        jobs.push_back({{"id", job->id},
                        {"repo", job->repo},
                        {"revision", job->revision},
                        {"dest_dir", path_utf8(job->dest_dir)},
                        {"target_dir", path_utf8(job->target_dir)},
                        {"replace", job->replace},
                        {"state", job->state},
                        {"error", job->error ? Json(*job->error) : Json(nullptr)},
                        {"created_at", job->created_at},
                        {"finished_at", job->finished_at ? Json(*job->finished_at) : Json(nullptr)},
                        {"files", files}});
    }
    if (hooks_.save_state) { hooks_.save_state(jobs); }
}

void Downloads::load() {
    const Json jobs = hooks_.load_state ? hooks_.load_state() : Json::array();
    if (!jobs.is_array()) { return; }
    for (const Json& value : jobs) {
        auto job        = std::make_unique<DownloadJob>();
        job->id         = json_or<std::string>(value, "id", new_id());
        job->repo       = json_or<std::string>(value, "repo", "");
        job->revision   = json_or<std::string>(value, "revision", "main");
        job->dest_dir   = utf8_path(json_or<std::string>(value, "dest_dir", ""));
        job->target_dir = utf8_path(json_or<std::string>(value, "target_dir", ""));
        job->replace    = json_or<bool>(value, "replace", false);
        job->state      = json_or<std::string>(value, "state", "queued");
        if (value.contains("error") && value["error"].is_string()) { job->error = value["error"].get<std::string>(); }
        job->created_at = json_or<std::int64_t>(value, "created_at", now_ms());
        if (value.contains("finished_at") && value["finished_at"].is_number()) {
            job->finished_at = value["finished_at"].get<std::int64_t>();
        }
        // A job the launcher stopped in the middle continues.
        if (job->state == "running" || job->state == "verifying") { job->state = "queued"; }
        for (const Json& f : value.value("files", Json::array())) {
            auto file   = std::make_unique<DownloadFile>();
            file->path  = json_or<std::string>(f, "path", "");
            file->bytes = json_or<std::uint64_t>(f, "bytes", 0);
            if (f.contains("sha256") && f["sha256"].is_string()) { file->sha256 = f["sha256"].get<std::string>(); }
            if (f.contains("git_blob") && f["git_blob"].is_string()) { file->git_blob = f["git_blob"].get<std::string>(); }
            file->done  = json_or<std::uint64_t>(f, "done", 0);
            file->state = json_or<std::string>(f, "state", "queued");
            if (file->state == "downloading" || file->state == "verifying") { file->state = "queued"; }
            if (f.contains("error") && f["error"].is_string()) { file->error = f["error"].get<std::string>(); }
            if (!file->path.empty() && safe_relative(file->path)) { job->files.push_back(std::move(file)); }
        }
        if (valid_repo_id(job->repo) && !job->target_dir.empty()) { jobs_.push_back(std::move(job)); }
    }
}

} // namespace infernix::launcher
