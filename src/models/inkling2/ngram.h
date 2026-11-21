#pragma once

// The n-gram archive type the EngineCore names (EngineOptions::speculative.ngram_archive_bytes). Inkling 2 has no
// n-gram drafting: normalize_inkling2_options keeps the archive's bytes 0, so the Engine never constructs one and
// PreparedPrompt::bind_ngram binds nothing.

#include "infernix/types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace infernix::models::inkling2 {

class NgramArchive {
public:
    struct Limits {
        std::size_t session_bytes = 0;
        std::size_t total_bytes   = 0;
    };
    struct Request {};

    explicit NgramArchive(Limits) { throw std::logic_error("Inkling 2 has no n-gram archive"); }
    [[nodiscard]] bool publish(std::unique_ptr<Request>&&, const std::vector<TokenId>&, std::uint32_t) { return false; }
    [[nodiscard]] NgramArchiveStats stats(const std::string&) const { return {}; }
    [[nodiscard]] std::size_t bytes() const noexcept { return 0; }
};

} // namespace infernix::models::inkling2
