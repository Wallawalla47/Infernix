#pragma once

// SHA-256 of a file (Windows CNG, which uses the SHA extensions; a portable implementation
// elsewhere), for checking downloads against Hugging Face's LFS hashes; and the git blob id
// (SHA-1) Hugging Face lists for a file kept in git itself, under its 10 MiB limit.

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

// `git hash-object`: SHA-1 of "blob <size>\0" and the contents, lowercase hex.
[[nodiscard]] std::optional<std::string> git_blob_id_file(const std::filesystem::path& path,
                                                          const std::atomic<bool>& cancel);
[[nodiscard]] std::string git_blob_id(std::string_view data);

} // namespace infernix::launcher
