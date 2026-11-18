#pragma once

// Small helpers shared by the launcher's modules: UTF-8 paths, time, strings and atomic files.

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace infernix::launcher {

using Json = nlohmann::json;

// Unix time in milliseconds.
[[nodiscard]] std::int64_t now_ms();

// Paths travel through JSON and the API as UTF-8 strings.
[[nodiscard]] std::string path_utf8(const std::filesystem::path& path);
[[nodiscard]] std::filesystem::path utf8_path(std::string_view text);
// A path that stays inside the folder it is joined to: relative, no drive, no ".." part (a file a repository listing
// or an artifact header names).
[[nodiscard]] bool safe_relative(std::string_view path);

// An OS error's message as UTF-8: std::error_code::message() gives a Windows system error in the system code page,
// which JSON and the event log cannot carry.
[[nodiscard]] std::string error_text(const std::error_code& error);

[[nodiscard]] std::string lower(std::string_view text);
[[nodiscard]] bool iequals(std::string_view a, std::string_view b);
[[nodiscard]] std::string trim(std::string_view text);
[[nodiscard]] bool icontains(std::string_view haystack, std::string_view needle);
[[nodiscard]] std::string hex_string(const std::uint8_t* data, std::size_t size);

// Whole-file reads; nullopt when the file cannot be read.
[[nodiscard]] std::optional<std::string> read_file(const std::filesystem::path& path);
// Writes a temporary file beside `path` (a name unique to the call), makes its bytes durable (FlushFileBuffers /
// fsync), then renames it over `path` (write-through on Windows; the folder synced on POSIX): a crash or a power
// loss leaves either the previous file or the new one, whole. With `keep_backup` the previous file becomes
// `backup_path(path)` first (read_file_or_backup reads it when `path` is missing or unreadable).
void write_file_atomic(const std::filesystem::path& path, std::string_view content, bool keep_backup = false);
[[nodiscard]] std::filesystem::path backup_path(const std::filesystem::path& path);

// A file's last write time in Unix milliseconds (0 when unknown).
[[nodiscard]] std::int64_t modified_ms(const std::filesystem::path& path);

// "76.9 GB" style, decimal units (as file managers and Hugging Face show sizes).
[[nodiscard]] std::string format_bytes(std::uint64_t bytes);

// Removes ANSI escape sequences (CSI and OSC) from a line of console output.
[[nodiscard]] std::string strip_ansi(std::string_view text);

// The directory holding the running executable.
[[nodiscard]] std::filesystem::path executable_dir();

// The deepest nesting of arrays and objects a client's JSON body may have (infernix-serve refuses deeper
// ones too): the DOM is parsed iteratively, but serializing and copying it recurse once per level, so a
// deeper body would overflow a worker thread's stack.
inline constexpr std::size_t kMaxJsonNesting = 256;

// The deepest array / object nesting of JSON text, outside its strings, without parsing it.
[[nodiscard]] std::size_t json_text_nesting(std::string_view text) noexcept;

// A client's JSON body; std::invalid_argument when it nests deeper than kMaxJsonNesting.
[[nodiscard]] Json parse_client_json(const std::string& text);

// JSON members with a fallback when absent or of another type.
template <class T>
[[nodiscard]] T json_or(const Json& object, const char* key, T fallback) {
    if (!object.is_object()) { return fallback; }
    const auto it = object.find(key);
    if (it == object.end() || it->is_null()) { return fallback; }
    try {
        return it->get<T>();
    } catch (const Json::exception&) { return fallback; }
}

} // namespace infernix::launcher
