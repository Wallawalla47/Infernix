#include "launcher/util.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <system_error>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <climits>
#include <unistd.h>
#endif

namespace infernix::launcher {

std::int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::string path_utf8(const std::filesystem::path& path) {
    const std::u8string text = path.u8string();
    return {reinterpret_cast<const char*>(text.data()), text.size()};
}

std::filesystem::path utf8_path(std::string_view text) {
    return std::filesystem::path(
        std::u8string(reinterpret_cast<const char8_t*>(text.data()), text.size()));
}

std::string lower(std::string_view text) {
    std::string out(text);
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

bool iequals(std::string_view a, std::string_view b) {
    return a.size() == b.size() &&
           std::equal(a.begin(), a.end(), b.begin(), [](unsigned char x, unsigned char y) {
               return std::tolower(x) == std::tolower(y);
           });
}

std::string trim(std::string_view text) {
    std::size_t begin = 0;
    std::size_t end   = text.size();
    while (begin < end && std::isspace(static_cast<unsigned char>(text[begin]))) { ++begin; }
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1]))) { --end; }
    return std::string(text.substr(begin, end - begin));
}

bool icontains(std::string_view haystack, std::string_view needle) {
    return lower(haystack).find(lower(needle)) != std::string::npos;
}

std::string hex_string(const std::uint8_t* data, std::size_t size) {
    static constexpr char kDigits[] = "0123456789abcdef";
    std::string out(size * 2, '0');
    for (std::size_t i = 0; i < size; ++i) {
        out[2 * i]     = kDigits[data[i] >> 4];
        out[2 * i + 1] = kDigits[data[i] & 15];
    }
    return out;
}

std::optional<std::string> read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { return std::nullopt; }
    std::ostringstream text;
    text << in.rdbuf();
    if (in.bad()) { return std::nullopt; }
    return text.str();
}

void write_file_atomic(const std::filesystem::path& path, std::string_view content) {
    std::error_code ignored;
    std::filesystem::create_directories(path.parent_path(), ignored);
    std::filesystem::path temporary = path;
    temporary += ".tmp";
    {
        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        if (!out) { throw std::runtime_error("cannot write " + path_utf8(temporary)); }
        out.write(content.data(), static_cast<std::streamsize>(content.size()));
        out.flush();
        if (!out) { throw std::runtime_error("cannot write " + path_utf8(temporary)); }
    }
    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    if (error) {
        throw std::runtime_error("cannot replace " + path_utf8(path) + ": " + error.message());
    }
}

std::int64_t modified_ms(const std::filesystem::path& path) {
    std::error_code error;
    const auto time = std::filesystem::last_write_time(path, error);
    if (error) { return 0; }
    // MSVC's file_clock converts through UTC (to_utc), libstdc++'s through to_sys.
#ifdef _MSC_VER
    const auto system = std::chrono::clock_cast<std::chrono::system_clock>(time);
#else
    const auto system = std::chrono::file_clock::to_sys(time);
#endif
    return std::chrono::duration_cast<std::chrono::milliseconds>(system.time_since_epoch()).count();
}

std::string format_bytes(std::uint64_t bytes) {
    static constexpr const char* kUnits[] = {"B", "kB", "MB", "GB", "TB"};
    double value      = static_cast<double>(bytes);
    std::size_t index = 0;
    while (value >= 1000.0 && index + 1 < std::size(kUnits)) {
        value /= 1000.0;
        ++index;
    }
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), index == 0 ? "%.0f %s" : "%.1f %s", value, kUnits[index]);
    return buffer;
}

std::string strip_ansi(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    for (std::size_t i = 0; i < text.size(); ++i) {
        const char c = text[i];
        if (c != '\x1b') {
            if (c != '\r') { out.push_back(c); }
            continue;
        }
        if (i + 1 >= text.size()) { break; }
        const char kind = text[i + 1];
        if (kind == '[') {
            // CSI: parameters and intermediates, then one final byte in 0x40..0x7E.
            i += 2;
            while (i < text.size() && !(text[i] >= 0x40 && text[i] <= 0x7E)) { ++i; }
        } else if (kind == ']') {
            // OSC: ends with BEL or ESC backslash.
            i += 2;
            while (i < text.size() && text[i] != '\x07' &&
                   !(text[i] == '\x1b' && i + 1 < text.size() && text[i + 1] == '\\')) {
                ++i;
            }
            if (i < text.size() && text[i] == '\x1b') { ++i; }
        } else {
            ++i;
        }
    }
    return out;
}

std::filesystem::path executable_dir() {
#ifdef _WIN32
    std::wstring buffer(MAX_PATH, L'\0');
    for (;;) {
        const DWORD length = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
        if (length == 0) { return std::filesystem::current_path(); }
        if (length < buffer.size()) {
            buffer.resize(length);
            break;
        }
        buffer.resize(buffer.size() * 2);
    }
    return std::filesystem::path(buffer).parent_path();
#else
    std::error_code error;
    const auto self = std::filesystem::read_symlink("/proc/self/exe", error);
    return error ? std::filesystem::current_path() : self.parent_path();
#endif
}

} // namespace infernix::launcher
