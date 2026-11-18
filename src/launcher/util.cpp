#include "launcher/util.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <climits>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace infernix::launcher {

std::size_t json_text_nesting(std::string_view text) noexcept {
    std::size_t depth = 0, deepest = 0;
    bool quoted = false, escaped = false;
    for (const char c : text) {
        if (quoted) {
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                quoted = false;
            }
        } else if (c == '"') {
            quoted = true;
        } else if (c == '[' || c == '{') {
            deepest = std::max(deepest, ++depth);
        } else if ((c == ']' || c == '}') && depth > 0) {
            --depth;
        }
    }
    return deepest;
}

Json parse_client_json(const std::string& text) {
    if (json_text_nesting(text) > kMaxJsonNesting) {
        throw std::invalid_argument("the JSON body nests arrays or objects deeper than " +
                                    std::to_string(kMaxJsonNesting) + " levels");
    }
    return Json::parse(text);
}

std::int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::string path_utf8(const std::filesystem::path& path) {
    const std::u8string text = path.u8string();
    return {reinterpret_cast<const char*>(text.data()), text.size()};
}

std::string error_text(const std::error_code& error) {
#ifdef _WIN32
    if (error.category() == std::system_category()) {
        wchar_t* buffer    = nullptr;
        const DWORD length = FormatMessageW(
            FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr,
            static_cast<DWORD>(error.value()), 0, reinterpret_cast<LPWSTR>(&buffer), 0, nullptr);
        std::string text;
        if (length != 0 && buffer != nullptr) { text = trim(path_utf8(std::filesystem::path(std::wstring(buffer, length)))); }
        if (buffer != nullptr) { LocalFree(buffer); }
        if (!text.empty()) { return text; }
    }
#endif
    return error.message();
}

std::filesystem::path utf8_path(std::string_view text) {
    return std::filesystem::path(
        std::u8string(reinterpret_cast<const char8_t*>(text.data()), text.size()));
}

bool safe_relative(std::string_view path) {
    if (path.empty() || path.front() == '/' || path.front() == '\\' || path.find(':') != std::string_view::npos) {
        return false;
    }
    std::size_t start = 0;
    while (start <= path.size()) {
        const std::size_t end      = path.find_first_of("/\\", start);
        const std::string_view part = path.substr(start, end == std::string_view::npos ? std::string_view::npos : end - start);
        if (part == "..") { return false; }
        if (end == std::string_view::npos) { break; }
        start = end + 1;
    }
    return true;
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
#ifdef _WIN32
    // Shares delete too (std::ifstream does not), so a writer's atomic replace of this file, a rename over it, is not
    // refused while it is read: chats are read unlocked while the next message's save replaces them.
    const HANDLE file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                    nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (file == INVALID_HANDLE_VALUE) { return std::nullopt; }
    constexpr std::size_t kChunk = std::size_t{1} << 20;
    std::string text;
    if (LARGE_INTEGER size{}; GetFileSizeEx(file, &size) != 0 && size.QuadPart > 0) {
        text.reserve(static_cast<std::size_t>(size.QuadPart) + kChunk); // the last read's room included
    }
    bool ok = true;
    for (;;) { // to the end, as the stream read: a file that grows meanwhile is read whole
        const std::size_t at = text.size();
        text.resize(at + kChunk);
        DWORD got = 0;
        ok        = ReadFile(file, text.data() + at, static_cast<DWORD>(kChunk), &got, nullptr) != 0;
        text.resize(at + (ok ? got : 0));
        if (!ok || got == 0) { break; }
    }
    CloseHandle(file);
    if (!ok) { return std::nullopt; }
    return text;
#else
    std::ifstream in(path, std::ios::binary);
    if (!in) { return std::nullopt; }
    std::ostringstream text;
    text << in.rdbuf();
    if (in.bad()) { return std::nullopt; }
    return text.str();
#endif
}

std::filesystem::path backup_path(const std::filesystem::path& path) {
    std::filesystem::path out = path;
    out += ".bak";
    return out;
}

namespace {

// The temporary file's bytes, durable before the rename publishes them (a rename is journaled, its data is not).
void write_durable(const std::filesystem::path& temporary, std::string_view content) {
#ifdef _WIN32
    const HANDLE file = CreateFileW(temporary.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL,
                                    nullptr);
    if (file == INVALID_HANDLE_VALUE) { throw std::runtime_error("cannot write " + path_utf8(temporary)); }
    bool ok = true;
    for (std::size_t at = 0; ok && at < content.size();) {
        const DWORD want = static_cast<DWORD>(std::min<std::size_t>(content.size() - at, 1U << 30));
        DWORD wrote      = 0;
        ok               = WriteFile(file, content.data() + at, want, &wrote, nullptr) != 0 && wrote == want;
        at += wrote;
    }
    ok = ok && FlushFileBuffers(file) != 0;
    ok = CloseHandle(file) != 0 && ok;
#else
    const int file = ::open(temporary.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (file < 0) { throw std::runtime_error("cannot write " + path_utf8(temporary)); }
    bool ok = true;
    for (std::size_t at = 0; ok && at < content.size();) {
        const ssize_t wrote = ::write(file, content.data() + at, content.size() - at);
        if (wrote < 0 && errno == EINTR) { continue; }
        ok = wrote > 0;
        if (ok) { at += static_cast<std::size_t>(wrote); }
    }
    ok = ok && ::fsync(file) == 0;
    ok = ::close(file) == 0 && ok;
#endif
    if (!ok) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw std::runtime_error("cannot write " + path_utf8(temporary));
    }
}

#ifdef _WIN32
// The POSIX rename (Windows 10 1709 and later, NTFS): `to` is replaced even while another handle has it open (sharing
// delete, as read_file does); that handle keeps reading the old file. The classic rename refuses an open target.
// False, with nothing renamed, where it is not supported (another file system, an older Windows).
bool posix_rename(const std::filesystem::path& from, const std::filesystem::path& to) {
    const HANDLE file = CreateFileW(from.c_str(), DELETE | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                    nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) { return false; }
    const std::wstring target = std::filesystem::absolute(to).wstring();
    const std::size_t bytes   = sizeof(FILE_RENAME_INFO) + target.size() * sizeof(wchar_t);
    std::vector<std::byte> buffer(bytes);
    auto* info           = reinterpret_cast<FILE_RENAME_INFO*>(buffer.data());
    info->Flags          = FILE_RENAME_FLAG_REPLACE_IF_EXISTS | FILE_RENAME_FLAG_POSIX_SEMANTICS;
    info->RootDirectory  = nullptr;
    info->FileNameLength = static_cast<DWORD>(target.size() * sizeof(wchar_t));
    std::memcpy(info->FileName, target.data(), target.size() * sizeof(wchar_t));
    bool ok = SetFileInformationByHandle(file, FileRenameInfoEx, info, static_cast<DWORD>(bytes)) != 0;
    // The rename durable before it is reported, as MOVEFILE_WRITE_THROUGH makes the classic one.
    ok = ok && FlushFileBuffers(file) != 0;
    CloseHandle(file);
    return ok;
}
#endif

// Renames `from` over `to`, durably: write-through on Windows; the folder's entry synced on POSIX.
bool rename_durable(const std::filesystem::path& from, const std::filesystem::path& to, std::string& error) {
#ifdef _WIN32
    if (posix_rename(from, to)) { return true; }
    if (MoveFileExW(from.c_str(), to.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) == 0) {
        error = error_text(std::error_code(static_cast<int>(GetLastError()), std::system_category()));
        return false;
    }
#else
    if (::rename(from.c_str(), to.c_str()) != 0) {
        error = std::system_category().message(errno);
        return false;
    }
    const std::filesystem::path folder = to.has_parent_path() ? to.parent_path() : std::filesystem::path(".");
    if (const int dir = ::open(folder.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC); dir >= 0) {
        (void)::fsync(dir);
        (void)::close(dir);
    }
#endif
    return true;
}

} // namespace

void write_file_atomic(const std::filesystem::path& path, std::string_view content, bool keep_backup) {
    std::error_code ignored;
    std::filesystem::create_directories(path.parent_path(), ignored);
    // Unique to the call: two writers of one path never write one temporary file.
    static std::atomic<std::uint64_t> sequence{0};
#ifdef _WIN32
    const unsigned long process = GetCurrentProcessId();
#else
    const unsigned long process = static_cast<unsigned long>(::getpid());
#endif
    std::filesystem::path temporary = path;
    temporary += ".tmp." + std::to_string(process) + "." + std::to_string(sequence.fetch_add(1) + 1);
    write_durable(temporary, content);
    std::string error;
    // The previous file becomes the backup (it was made durable when it was written), then the new one replaces it:
    // a crash between the two leaves the backup, which the reader takes when the file is missing.
    if (keep_backup && std::filesystem::exists(path, ignored) && !rename_durable(path, backup_path(path), error)) {
        std::filesystem::remove(temporary, ignored);
        throw std::runtime_error("cannot keep " + path_utf8(backup_path(path)) + ": " + error);
    }
    if (!rename_durable(temporary, path, error)) {
        std::filesystem::remove(temporary, ignored);
        throw std::runtime_error("cannot replace " + path_utf8(path) + ": " + error);
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
