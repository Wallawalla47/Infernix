#pragma once

#include <cstdint>

namespace ninfer::runtime {

// Portable unsigned 128-bit integer. MSVC has no native 128-bit type, so this
// emulates the operations NInfer's host code needs (the context-cost model and
// prefill work) with 64-bit limb arithmetic; every operation is constexpr.
// Multiplication assumes both operands' high 64 bits are zero (all call sites
// multiply 64-bit values); division supports an arbitrary nonzero 64-bit divisor.
struct uint128 {
    std::uint64_t hi;
    std::uint64_t lo;

    constexpr uint128() noexcept : hi(0), lo(0) {}
    constexpr uint128(std::uint64_t value) noexcept : hi(0), lo(value) {}
    constexpr uint128(std::uint64_t high, std::uint64_t low) noexcept : hi(high), lo(low) {}
    explicit constexpr operator std::uint64_t() const noexcept { return lo; }

    friend constexpr bool operator==(uint128 a, uint128 b) noexcept {
        return a.hi == b.hi && a.lo == b.lo;
    }
    friend constexpr bool operator<(uint128 a, uint128 b) noexcept {
        return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo);
    }

    constexpr uint128 operator+(uint128 rhs) const noexcept {
        return uint128(hi + rhs.hi + (lo + rhs.lo < lo ? 1 : 0), lo + rhs.lo);
    }
    constexpr uint128 operator-(uint128 rhs) const noexcept {
        return uint128(hi - rhs.hi - (lo < rhs.lo ? 1 : 0), lo - rhs.lo);
    }
    constexpr uint128 operator+(std::uint64_t rhs) const noexcept {
        return uint128(hi + (lo + rhs < lo ? 1 : 0), lo + rhs);
    }
    constexpr uint128 operator-(std::uint64_t rhs) const noexcept {
        return uint128(hi - (lo < rhs ? 1 : 0), lo - rhs);
    }
    constexpr uint128 operator~() const noexcept { return uint128(~hi, ~lo); }
    constexpr uint128 operator<<(unsigned shift) const noexcept {
        if (shift == 0) { return *this; }
        if (shift >= 64) { return uint128(shift == 64 ? lo : 0, 0); }
        return uint128((hi << shift) | (lo >> (64 - shift)), lo << shift);
    }
    constexpr uint128 operator>>(unsigned shift) const noexcept {
        if (shift == 0) { return *this; }
        if (shift >= 64) { return uint128(0, shift == 64 ? hi : 0); }
        return uint128(hi >> shift, (hi << (64 - shift)) | (lo >> shift));
    }

    // High 64 bits of the 128-bit product of two 64-bit values.
    static constexpr std::uint64_t mul_hi(std::uint64_t a, std::uint64_t b) noexcept {
        const std::uint64_t a0 = a & 0xFFFFFFFFULL, a1 = a >> 32;
        const std::uint64_t b0 = b & 0xFFFFFFFFULL, b1 = b >> 32;
        const std::uint64_t p0 = a0 * b0;
        const std::uint64_t p1 = a0 * b1;
        const std::uint64_t p2 = a1 * b0;
        const std::uint64_t p3 = a1 * b1;
        const std::uint64_t mid = (p0 >> 32) + (p1 & 0xFFFFFFFFULL) + (p2 & 0xFFFFFFFFULL);
        return p3 + (p1 >> 32) + (p2 >> 32) + (mid >> 32);
    }
    constexpr uint128 operator*(std::uint64_t rhs) const noexcept {
        return uint128(mul_hi(lo, rhs), lo * rhs);
    }
    constexpr uint128 operator*(uint128 rhs) const noexcept {
        return uint128(mul_hi(lo, rhs.lo), lo * rhs.lo);
    }
    constexpr uint128 operator/(std::uint64_t divisor) const noexcept {
        uint128 rem{};
        uint128 quot{};
        for (int k = 127; k >= 0; --k) {
            rem = rem << 1;
            const std::uint64_t bit = (k >= 64) ? (hi >> (k - 64) & 1ULL) : (lo >> k & 1ULL);
            if (bit != 0) { rem = rem + uint128(1ULL); }
            if (!(rem < uint128(divisor))) {
                rem = rem - uint128(divisor);
                if (k >= 64) { quot.hi |= std::uint64_t(1) << (k - 64); }
                else { quot.lo |= std::uint64_t(1) << k; }
            }
        }
        return quot;
    }
};

inline constexpr bool operator!=(uint128 a, uint128 b) noexcept { return !(a == b); }
inline constexpr bool operator>=(uint128 a, uint128 b) noexcept { return !(a < b); }
inline constexpr bool operator>(uint128 a, uint128 b) noexcept { return b < a; }
inline constexpr bool operator<=(uint128 a, uint128 b) noexcept { return !(b < a); }

} // namespace ninfer::runtime