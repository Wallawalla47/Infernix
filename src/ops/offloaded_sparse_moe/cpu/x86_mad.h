#pragma once

// x86 building blocks of the exact CPU expert routes (ggml_expert.cpp, fp8_expert.cpp): per-function
// ISA targets, the 16-bit multiply-add of each ISA (vpmaddwd, or vpdpwssd with AVX-VNNI or
// AVX-512 VNNI; every one exact in int32 lanes) and a batched horizontal sum.

#if defined(__x86_64__) || defined(_M_X64)
#    include <immintrin.h>
#    define INFERNIX_CPU_X86 1
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#    define INFERNIX_TARGET(x)
#else
#    define INFERNIX_TARGET(x) __attribute__((target(x)))
#endif

#if defined(INFERNIX_CPU_X86)

namespace infernix::ops::offloaded_moe::x86 {

struct Avx2Mad {
    INFERNIX_TARGET("avx2") static __m256i mad(__m256i acc, __m256i w, __m256i x) {
        return _mm256_add_epi32(acc, _mm256_madd_epi16(w, x));
    }
};
struct AvxVnniMad {
    INFERNIX_TARGET("avx2,avxvnni") static __m256i mad(__m256i acc, __m256i w, __m256i x) {
        return _mm256_dpwssd_avx_epi32(acc, w, x);
    }
};
struct Avx512VnniMad {
    INFERNIX_TARGET("avx2,avx512f,avx512vl,avx512vnni") static __m256i mad(__m256i acc, __m256i w, __m256i x) {
        return _mm256_dpwssd_epi32(acc, w, x);
    }
};

// out[i] = sum of v[i]'s eight lanes (the caller keeps every sum below 2^31).
INFERNIX_TARGET("avx2")
inline __m256i hsum8(const __m256i (&v)[8]) {
    const __m256i t0 = _mm256_hadd_epi32(v[0], v[1]);
    const __m256i t1 = _mm256_hadd_epi32(v[2], v[3]);
    const __m256i t2 = _mm256_hadd_epi32(v[4], v[5]);
    const __m256i t3 = _mm256_hadd_epi32(v[6], v[7]);
    const __m256i u0 = _mm256_hadd_epi32(t0, t1); // v0..v3 sums of lanes 0-3, then of lanes 4-7
    const __m256i u1 = _mm256_hadd_epi32(t2, t3);
    return _mm256_add_epi32(_mm256_permute2x128_si256(u0, u1, 0x20), _mm256_permute2x128_si256(u0, u1, 0x31));
}

} // namespace infernix::ops::offloaded_moe::x86

#endif // INFERNIX_CPU_X86
