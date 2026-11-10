#include "ops/offloaded_sparse_moe/csf_record.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>

namespace infernix::ops::offloaded_moe::csf {
namespace {

constexpr std::size_t pad16(std::size_t bytes) { return (bytes + 15) / 16 * 16; }

using Histograms = std::array<std::array<std::uint16_t, 256>, 16>;

// Each row's base: the lowest start of a 15-code window holding the most of its codes (the row's lowest code when
// they all fit). `units` are `blocks` expanded units; `hist` is zero on entry and on return.
void row_bases(const std::uint8_t* units, int blocks, Histograms& hist, std::uint8_t (&bases)[16]) {
    int lo[16], hi[16];
    std::fill(std::begin(lo), std::end(lo), 255);
    std::fill(std::begin(hi), std::end(hi), 0);
    for (int b = 0; b < blocks; ++b) {
        const std::uint8_t* s = units + static_cast<std::size_t>(b) * kUnitBytes + kCodeBytes;
        for (int r = 0; r < 16; ++r) {
            ++hist[static_cast<std::size_t>(r)][s[r]];
            lo[r] = std::min<int>(lo[r], s[r]);
            hi[r] = std::max<int>(hi[r], s[r]);
        }
    }
    for (int r = 0; r < 16; ++r) {
        auto& h = hist[static_cast<std::size_t>(r)];
        if (hi[r] - lo[r] < kWindow) {
            bases[r] = static_cast<std::uint8_t>(lo[r]);
        } else {
            int count = 0;
            for (int c = lo[r]; c < lo[r] + kWindow; ++c) { count += h[static_cast<std::size_t>(c)]; }
            int best = count, start = lo[r];
            for (int b = lo[r] + 1; b + kWindow - 1 <= hi[r]; ++b) {
                count += h[static_cast<std::size_t>(b + kWindow - 1)] - h[static_cast<std::size_t>(b - 1)];
                if (count > best) {
                    best  = count;
                    start = b;
                }
            }
            bases[r] = static_cast<std::uint8_t>(start);
        }
        std::fill(h.begin() + lo[r], h.begin() + hi[r] + 1, std::uint16_t{0});
    }
}

bool escaped(int code, int base) { return code - base < 0 || code - base >= kWindow; }

} // namespace

std::size_t compress_record(const ExpertShape& shape, const std::uint8_t* expanded, std::uint8_t* out, std::uint32_t* table) {
    if (!registered_shape(shape)) { throw std::invalid_argument("csf: unregistered expert shape"); }
    const Geometry g = Geometry::of(shape);
    // One row group's expanded units: read before its region is written (in place, the region may cover them).
    std::array<std::uint8_t, kMaxChunkBytes> units{};
    Histograms hist{};
    std::size_t at = 0;
    for (int rg = 0; rg < g.rgs(); ++rg) {
        const int blocks       = g.blocks(rg);
        const std::size_t raw  = static_cast<std::size_t>(blocks) * kUnitBytes;
        std::memcpy(units.data(), expanded + g.expanded_offset(rg), raw);
        std::uint8_t bases[16];
        row_bases(units.data(), blocks, hist, bases);
        std::size_t escapes = 0;
        for (int b = 0; b < blocks; ++b) {
            const std::uint8_t* s = units.data() + static_cast<std::size_t>(b) * kUnitBytes + kCodeBytes;
            for (int r = 0; r < 16; ++r) { escapes += escaped(s[r], bases[r]) ? 1U : 0U; }
        }
        const std::size_t bodies_at = kBaseBytes + pad16(escapes);
        const std::size_t size      = bodies_at + static_cast<std::size_t>(blocks) * kBodyBytes;
        table[rg]                   = static_cast<std::uint32_t>(at);
        std::uint8_t* region        = out + at;
        if (size >= raw) {
            std::memcpy(region, units.data(), raw);
            at += raw;
            continue;
        }
        std::memcpy(region, bases, kBaseBytes);
        std::uint8_t* escape = region + kBaseBytes;
        for (int b = 0; b < blocks; ++b) {
            const std::uint8_t* unit = units.data() + static_cast<std::size_t>(b) * kUnitBytes;
            std::uint8_t* body       = region + bodies_at + static_cast<std::size_t>(b) * kBodyBytes;
            std::memcpy(body, unit, kCodeBytes);
            std::uint8_t offsets[8] = {};
            for (int r = 0; r < 16; ++r) {
                const int code = unit[kCodeBytes + r];
                int stored     = kEscape;
                if (!escaped(code, bases[r])) {
                    stored = code - bases[r];
                } else {
                    *escape++ = static_cast<std::uint8_t>(code);
                }
                offsets[r / 2] = static_cast<std::uint8_t>(offsets[r / 2] | ((r & 1) == 0 ? stored : stored << 4));
            }
            std::memcpy(body + kCodeBytes, offsets, 8);
        }
        std::memset(escape, 0, static_cast<std::size_t>(region + bodies_at - escape));
        at += size;
    }
    table[g.rgs()] = static_cast<std::uint32_t>(at);
    return at;
}

void expand_record(const ExpertShape& shape, const std::uint8_t* data, const std::uint32_t* table, std::uint8_t* out) {
    if (!registered_shape(shape)) { throw std::invalid_argument("csf: unregistered expert shape"); }
    const Geometry g = Geometry::of(shape);
    for (int rg = 0; rg < g.rgs(); ++rg) {
        const int blocks   = g.blocks(rg);
        const Group group_ = group(data, table, rg, blocks);
        std::uint8_t* dst  = out + g.expanded_offset(rg);
        if (group_.bases == nullptr) {
            std::memcpy(dst, group_.units, static_cast<std::size_t>(blocks) * kUnitBytes);
            continue;
        }
        const std::uint8_t* escape = group_.escapes;
        for (int b = 0; b < blocks; ++b) {
            const std::uint8_t* body = group_.units + static_cast<std::size_t>(b) * kBodyBytes;
            std::uint8_t* unit       = dst + static_cast<std::size_t>(b) * kUnitBytes;
            std::memcpy(unit, body, kCodeBytes);
            std::uint8_t scales[16];
            escape = unit_scales(body, group_.bases, escape, scales);
            std::memcpy(unit + kCodeBytes, scales, 16);
        }
    }
}

void expand_in_place_host(const ExpertShape& shape, std::uint8_t* record, const std::uint32_t* table) {
    if (!registered_shape(shape)) { throw std::invalid_argument("csf: unregistered expert shape"); }
    const Geometry g = Geometry::of(shape);
    std::array<std::uint8_t, kMaxChunkBytes> region{};
    for (int rg = g.rgs() - 1; rg >= 0; --rg) {
        const int blocks          = g.blocks(rg);
        const std::uint32_t begin = table[rg], end = table[rg + 1];
        std::uint8_t* dst         = record + g.expanded_offset(rg);
        std::memcpy(region.data(), record + begin, end - begin); // read before the units below overwrite it
        const std::uint32_t local[2] = {0, end - begin};
        const Group group_ = group(region.data(), local, 0, blocks);
        if (group_.bases == nullptr) {
            std::memcpy(dst, region.data(), static_cast<std::size_t>(blocks) * kUnitBytes);
            continue;
        }
        const std::uint8_t* escape = group_.escapes;
        for (int b = 0; b < blocks; ++b) {
            const std::uint8_t* body = group_.units + static_cast<std::size_t>(b) * kBodyBytes;
            std::uint8_t* unit       = dst + static_cast<std::size_t>(b) * kUnitBytes;
            std::memcpy(unit, body, kCodeBytes);
            std::uint8_t scales[16];
            escape = unit_scales(body, group_.bases, escape, scales);
            std::memcpy(unit + kCodeBytes, scales, 16);
        }
    }
}

bool compress_slot(const ExpertShape& shape, std::uint8_t* slot, std::uint64_t pitch) {
    if (!registered_shape(shape) || pitch < shape.record_bytes()) { throw std::invalid_argument("csf: bad slot"); }
    const Geometry g        = Geometry::of(shape);
    const std::uint64_t at  = slot_table_offset(g.table_words(), pitch);
    std::array<std::uint32_t, kMaxTableWords> table{};
    const std::size_t bytes = compress_record(shape, slot, slot, table.data());
    if (bytes > at) {
        expand_in_place_host(shape, slot, table.data()); // no room for the table: back to the expanded record
        return false;
    }
    std::memcpy(slot + at, table.data(), static_cast<std::size_t>(g.table_words()) * sizeof(std::uint32_t));
    return true;
}

bool valid_table(const ExpertShape& shape, const std::uint32_t* table, std::uint64_t stride) {
    if (!registered_shape(shape) || table[0] != 0) { return false; }
    const Geometry g = Geometry::of(shape);
    for (int rg = 0; rg < g.rgs(); ++rg) {
        const std::uint64_t begin = table[rg], end = table[rg + 1];
        const std::uint64_t raw    = static_cast<std::uint64_t>(g.blocks(rg)) * kUnitBytes;
        const std::uint64_t bodies = static_cast<std::uint64_t>(g.blocks(rg)) * kBodyBytes;
        if (begin % 16 != 0 || end <= begin || end > stride || begin > g.expanded_offset(rg)) { return false; }
        const std::uint64_t size = end - begin;
        if (size != raw && (size > raw || size < kBaseBytes + bodies || (size - kBaseBytes - bodies) % 16 != 0)) { return false; }
    }
    return true;
}

} // namespace infernix::ops::offloaded_moe::csf
