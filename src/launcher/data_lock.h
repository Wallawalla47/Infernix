#pragma once

#include <filesystem>

namespace infernix::launcher {

// One launcher a data folder: an exclusive lock on <data dir>/launcher.lock, held for the launcher's lifetime and
// released by the operating system when the process ends (a crash included). A second launcher on the same folder
// fails before it reads or changes anything there: its constructor would otherwise end the first one's ablation
// engines as leftovers of a run, and both would write the same memory files.
class DataDirLock {
public:
    // Throws std::runtime_error naming the folder when another process holds the lock.
    explicit DataDirLock(const std::filesystem::path& data_dir);
    ~DataDirLock();
    DataDirLock(const DataDirLock&)            = delete;
    DataDirLock& operator=(const DataDirLock&) = delete;

private:
#ifdef _WIN32
    void* handle_ = nullptr;
#else
    int fd_ = -1;
#endif
};

} // namespace infernix::launcher
