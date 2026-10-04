// PLE n-gram row ids (docs/maintainer/qwen3_8-flash-next-design.md §12) against the fixture written
// by tools/flash_next/ngram.py: multipliers, head primes, padded table size and every row, exactly.

#include "models/qwen4_exp/frontend/ngram_hash.h"

#include <cinttypes>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef NINFER_SOURCE_DIR
#    define NINFER_SOURCE_DIR "."
#endif

using namespace ninfer::models::qwen4_exp;

int main() {
    std::ifstream in(std::string(NINFER_SOURCE_DIR) + "/tests/fixtures/qwen4_exp/ngram_rows.txt");
    if (!in) {
        std::fprintf(stderr, "FAIL: fixture unreadable\n");
        return 1;
    }
    std::string line, tag;
    std::getline(in, line);
    NgramConfig c;
    std::sscanf(line.c_str(), "# vocab %u eos %d n %u heads %u base %" SCNu64 " div %" SCNu64 " seed %" SCNu64,
                &c.vocab_size, &c.eos_token_id, &c.ngram_size, &c.heads_per_ngram, &c.ngram_vocab_size_base,
                &c.make_ngram_vocab_size_divisible_by, &c.seed);
    std::getline(in, line);
    std::istringstream hs(line);
    hs >> tag;
    std::vector<std::int32_t> history;
    for (std::int32_t t; hs >> t;) { history.push_back(t); }
    long failures = 0, rows_checked = 0;
    std::vector<NgramHash> layers{NgramHash(c, 0), NgramHash(c, 1)};
    std::vector<std::vector<std::uint32_t>> rows(2);
    const std::size_t count = history.size() - (c.ngram_size - 1);
    for (int l = 0; l < 2; ++l) {
        rows[l].resize(count * layers[l].heads());
        layers[l].row_ids(history, count, rows[l].data());
    }
    while (std::getline(in, line)) {
        std::istringstream s(line);
        s >> tag;
        int layer = 0;
        s >> layer;
        const NgramHash& h = layers[static_cast<std::size_t>(layer)];
        if (tag == "L") {
            std::string kind;
            s >> kind;
            std::vector<std::uint64_t> v;
            std::string w;
            std::uint64_t padded = 0;
            while (s >> w) {
                if (w == "P") {
                    s >> padded;
                    break;
                }
                v.push_back(std::stoull(w));
            }
            const auto got = kind == "M" ? h.multipliers() : h.sizes();
            failures += !(std::vector<std::uint64_t>(got.begin(), got.end()) == v);
            if (kind == "S") { failures += h.table_rows() != padded; }
        } else if (tag == "R") {
            std::size_t p = 0;
            s >> p;
            for (std::uint32_t j = 0; j < h.heads(); ++j) {
                std::uint32_t want = 0;
                s >> want;
                failures += rows[static_cast<std::size_t>(layer)][p * h.heads() + j] != want;
            }
            ++rows_checked;
        }
    }
    std::printf("n-gram rows: %ld positions checked, %ld mismatches\n", rows_checked, failures);
    if (failures != 0 || rows_checked < 700) {
        std::fprintf(stderr, "FAIL: n-gram row ids differ from the reference\n");
        return 1;
    }
    std::printf("all qwen4_exp n-gram checks passed\n");
    return 0;
}
