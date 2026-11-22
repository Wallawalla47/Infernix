#pragma once

// Hugging Face Hub access over libcurl: a repository's files at a pinned commit (sizes and LFS
// SHA-256), resolve URLs, and small HTTPS GETs.

#include <atomic>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace infernix::launcher {

struct HfFile {
    std::string path;
    std::uint64_t size = 0;
    std::optional<std::string> sha256;   // an LFS file's
    std::optional<std::string> git_blob; // a file kept in git itself: its blob id
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

// GET with optional bearer token; follows redirects. `error` is set when no response arrived. A set `cancel` ends
// the request within about a second (a launcher that is stopping does not wait for a slow or unreachable host).
[[nodiscard]] HttpResult http_get(const std::string& url, const std::string& token, long timeout_seconds,
                                  const std::atomic<bool>* cancel = nullptr);

// `owner/name` validation: letters, digits, '-', '_', '.' in two parts.
[[nodiscard]] bool valid_repo_id(const std::string& repo);

// Lists a model repository's files at `main` (throws std::runtime_error with a readable reason); `cancel` as http_get.
[[nodiscard]] HfListing hf_list_files(const std::string& repo, const std::string& token,
                                      const std::atomic<bool>* cancel = nullptr);

[[nodiscard]] std::string hf_resolve_url(const std::string& repo, const std::string& revision, const std::string& path);

// A PEFT LoRA adapter repository on the Hub (one `--lora` can read: adapter_config.json and adapter_model.safetensors
// at its root).
struct HfAdapter {
    std::string repo;
    std::uint64_t downloads = 0; // the Hub's count of the last 30 days
    std::uint64_t likes     = 0;
    std::string last_modified;   // ISO 8601
    bool gated = false;          // its files need a token from an account that accepted its terms
};

// The PEFT adapters of `base` (`owner/name`) whose files `--lora` reads, most downloaded first: at most `limit`,
// only `author`'s when it is not empty, only those whose name contains `search` when that is not empty. Throws
// std::runtime_error with a readable reason; `cancel` as http_get.
[[nodiscard]] std::vector<HfAdapter> hf_search_adapters(const std::string& base, const std::string& author,
                                                        const std::string& search, int limit,
                                                        const std::string& token,
                                                        const std::atomic<bool>* cancel = nullptr);

// The adapters among the Hub's model list (an /api/models answer with `siblings`), in its order (exposed for tests).
[[nodiscard]] std::vector<HfAdapter> parse_adapter_search(const std::string& body);

// Percent-encodes one URL path (keeps '/').
[[nodiscard]] std::string url_encode_path(const std::string& path);

} // namespace infernix::launcher
