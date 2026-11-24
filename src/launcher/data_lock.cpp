#include "launcher/data_lock.h"

#include "launcher/util.h"

#include <stdexcept>
#include <string>
#include <system_error>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace infernix::launcher {
namespace {

std::runtime_error held(const std::filesystem::path& data_dir) {
    return std::runtime_error("another infernix-launcher uses the data folder " + path_utf8(data_dir) +
                              "; close it first, or start this one with another --data-dir");
}

} // namespace

DataDirLock::DataDirLock(const std::filesystem::path& data_dir) {
    std::error_code ignored;
    std::filesystem::create_directories(data_dir, ignored);
    const std::filesystem::path file = data_dir / "launcher.lock";
#ifdef _WIN32
    // No sharing: a second open fails with ERROR_SHARING_VIOLATION while this handle lives.
    HANDLE handle = CreateFileW(file.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_ALWAYS,
                                FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        const DWORD error = GetLastError();
        if (error == ERROR_SHARING_VIOLATION || error == ERROR_LOCK_VIOLATION) { throw held(data_dir); }
        throw std::runtime_error("cannot open " + path_utf8(file) + " (Windows error " + std::to_string(error) + ")");
    }
    handle_ = handle;
#else
    const int fd = ::open(file.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
    if (fd < 0) { throw std::runtime_error("cannot open " + path_utf8(file)); }
    if (::flock(fd, LOCK_EX | LOCK_NB) != 0) {
        ::close(fd);
        throw held(data_dir);
    }
    fd_ = fd;
#endif
}

DataDirLock::~DataDirLock() {
#ifdef _WIN32
    if (handle_ != nullptr) { CloseHandle(static_cast<HANDLE>(handle_)); }
#else
    if (fd_ >= 0) { ::close(fd_); }
#endif
}

} // namespace infernix::launcher
