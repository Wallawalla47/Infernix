#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace infernix::product::media_acquire {

enum class SourceKind {
    Path,
    Url,
    Data,
    Bytes,
};

struct Source {
    SourceKind kind = SourceKind::Path;
    std::string value;
    std::string media_type;
    std::vector<std::uint8_t> bytes;
};

} // namespace infernix::product::media_acquire
