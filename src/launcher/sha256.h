#pragma once

// SHA-256 of a file (Windows CNG, which uses the SHA extensions; a portable implementation
// elsewhere), for checking downloads against Hugging Face's LFS hashes.

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <string_view>

namespace infernix::launcher {

// Lowercase hex digest; nullopt when the file cannot be read or `cancel` turned true. `progress`
// receives the bytes hashed so far.
[[nodiscard]] std::optional<std::string> sha256_file(const std::filesystem::path& path,
                                                     const std::function<void(std::uint64_t)>& progress,
                                                     const std::atomic<bool>& cancel);

[[nodiscard]] std::string sha256_hex(std::string_view data);

} // namespace infernix::launcher
