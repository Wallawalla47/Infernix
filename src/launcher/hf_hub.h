#pragma once

// Hugging Face Hub access over libcurl: a repository's files at a pinned commit (sizes and LFS
// SHA-256), resolve URLs, and small HTTPS GETs.

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace infernix::launcher {

struct HfFile {
    std::string path;
    std::uint64_t size = 0;
    std::optional<std::string> sha256;
};

struct HfListing {
    std::string repo;
    std::string revision; // commit sha
    std::vector<HfFile> files;
};

struct HttpResult {
    long status = 0;
    std::string body;
    std::string error;
};

// GET with optional bearer token; follows redirects. `error` is set when no response arrived.
[[nodiscard]] HttpResult http_get(const std::string& url, const std::string& token, long timeout_seconds);

// `owner/name` validation: letters, digits, '-', '_', '.' in two parts.
[[nodiscard]] bool valid_repo_id(const std::string& repo);

// Lists a model repository's files at `main` (throws std::runtime_error with a readable reason).
[[nodiscard]] HfListing hf_list_files(const std::string& repo, const std::string& token);

[[nodiscard]] std::string hf_resolve_url(const std::string& repo, const std::string& revision, const std::string& path);

// Percent-encodes one URL path (keeps '/').
[[nodiscard]] std::string url_encode_path(const std::string& path);

} // namespace infernix::launcher
