#include "launcher/hf_hub.h"

#include "launcher/util.h"

#include <curl/curl.h>

#include <cctype>
#include <memory>
#include <stdexcept>

namespace infernix::launcher {
namespace {

size_t append_body(char* data, size_t size, size_t count, void* user) {
    auto* body = static_cast<std::string*>(user);
    body->append(data, size * count);
    // Bound API answers; a file listing is a few hundred kB.
    return body->size() > (64u << 20) ? 0 : size * count;
}

} // namespace

HttpResult http_get(const std::string& url, const std::string& token, long timeout_seconds) {
    HttpResult result;
    std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> curl(curl_easy_init(), &curl_easy_cleanup);
    if (!curl) {
        result.error = "cannot initialize libcurl";
        return result;
    }
    curl_slist* headers = nullptr;
    if (!token.empty()) { headers = curl_slist_append(headers, ("Authorization: Bearer " + token).c_str()); }
    headers = curl_slist_append(headers, "Accept: application/json");
    curl_easy_setopt(curl.get(), CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl.get(), CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl.get(), CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl.get(), CURLOPT_MAXREDIRS, 10L);
    curl_easy_setopt(curl.get(), CURLOPT_PROTOCOLS_STR, "https,http");
    curl_easy_setopt(curl.get(), CURLOPT_REDIR_PROTOCOLS_STR, "https");
    curl_easy_setopt(curl.get(), CURLOPT_CONNECTTIMEOUT, 20L);
    curl_easy_setopt(curl.get(), CURLOPT_TIMEOUT, timeout_seconds);
    curl_easy_setopt(curl.get(), CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(curl.get(), CURLOPT_USERAGENT, "infernix-launcher");
#ifdef _WIN32
    curl_easy_setopt(curl.get(), CURLOPT_SSL_OPTIONS, static_cast<long>(CURLSSLOPT_NATIVE_CA));
#endif
    curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION, append_body);
    curl_easy_setopt(curl.get(), CURLOPT_WRITEDATA, &result.body);
    const CURLcode code = curl_easy_perform(curl.get());
    curl_slist_free_all(headers);
    if (code != CURLE_OK) {
        result.error = curl_easy_strerror(code);
        return result;
    }
    curl_easy_getinfo(curl.get(), CURLINFO_RESPONSE_CODE, &result.status);
    return result;
}

bool valid_repo_id(const std::string& repo) {
    const std::size_t slash = repo.find('/');
    if (slash == std::string::npos || slash == 0 || slash + 1 >= repo.size() ||
        repo.find('/', slash + 1) != std::string::npos || repo.size() > 200) {
        return false;
    }
    for (const char c : repo) {
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_' || c == '.' || c == '/')) {
            return false;
        }
    }
    return repo.find("..") == std::string::npos;
}

std::string url_encode_path(const std::string& path) {
    static constexpr char kHex[] = "0123456789ABCDEF";
    std::string out;
    for (const unsigned char c : path) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~' || c == '/') {
            out.push_back(static_cast<char>(c));
        } else {
            out.push_back('%');
            out.push_back(kHex[c >> 4]);
            out.push_back(kHex[c & 15]);
        }
    }
    return out;
}

std::string hf_resolve_url(const std::string& repo, const std::string& revision, const std::string& path) {
    return "https://huggingface.co/" + repo + "/resolve/" + revision + "/" + url_encode_path(path);
}

HfListing hf_list_files(const std::string& repo, const std::string& token) {
    if (!valid_repo_id(repo)) { throw std::runtime_error("not a Hugging Face repository id: " + repo); }
    const HttpResult result =
        http_get("https://huggingface.co/api/models/" + repo + "/revision/main?blobs=true", token, 60);
    if (!result.error.empty()) { throw std::runtime_error("cannot reach huggingface.co: " + result.error); }
    if (result.status == 401 || result.status == 403) {
        throw std::runtime_error(repo + " needs a Hugging Face token with access (Settings, Downloads)");
    }
    if (result.status == 404) { throw std::runtime_error("no such repository: " + repo); }
    if (result.status != 200) {
        throw std::runtime_error("huggingface.co answered HTTP " + std::to_string(result.status));
    }
    HfListing listing;
    listing.repo = repo;
    try {
        const Json body  = Json::parse(result.body);
        listing.revision = json_or<std::string>(body, "sha", "main");
        for (const Json& sibling : body.value("siblings", Json::array())) {
            HfFile file;
            file.path = json_or<std::string>(sibling, "rfilename", "");
            if (file.path.empty()) { continue; }
            file.size = json_or<std::uint64_t>(sibling, "size", 0);
            if (sibling.contains("lfs") && sibling["lfs"].is_object()) {
                const Json& lfs = sibling["lfs"];
                if (lfs.contains("sha256") && lfs["sha256"].is_string()) {
                    file.sha256 = lower(lfs["sha256"].get<std::string>());
                }
                file.size = json_or<std::uint64_t>(lfs, "size", file.size);
            }
            listing.files.push_back(std::move(file));
        }
    } catch (const Json::exception& error) {
        throw std::runtime_error(std::string("unexpected answer from huggingface.co: ") + error.what());
    }
    return listing;
}

} // namespace infernix::launcher
