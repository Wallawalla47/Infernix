#pragma once

#include <filesystem>
#include <string>
#include <string_view>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace infernix::test {

// `name` in the system temp directory, prefixed with this process's id: concurrent test processes
// (CTest -j, other build trees and worktrees running the suite) never share a file.
inline std::filesystem::path unique_temp_path(std::string_view name) {
#ifdef _WIN32
    const long long process = ::_getpid();
#else
    const long long process = static_cast<long long>(::getpid());
#endif
    return std::filesystem::temp_directory_path() /
           ("infernix-" + std::to_string(process) + "-" + std::string(name));
}

} // namespace infernix::test
