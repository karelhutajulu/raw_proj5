#include "bitwise.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#define BITWISE_HAS_X86 1
#endif

namespace {

constexpr std::uint8_t  kBase   = 0xA5u;
constexpr std::uint8_t  kDelta  = 0x99u;
constexpr std::uint64_t kBase64  = 0xA5A5A5A5A5A5A5A5ull;
constexpr std::uint64_t kDelta64 = 0x9999999999999999ull;

#if defined(BITWISE_HAS_X86) && (defined(__GNUC__) || defined(__clang__))
[[gnu::target("avx2")]]
void bitwise_avx2(std::uint8_t *__restrict__ dst,
                  const std::uint8_t *__restrict__ pa,
                  const std::uint8_t *__restrict__ pb,
                  std::size_t n) {
    const __m256i vbase  = _mm256_set1_epi8(static_cast<char>(kBase));
    const __m256i vdelta = _mm256_set1_epi8(static_cast<char>(kDelta));

    std::size_t i = 0;
    for (; i + 64 <= n; i += 64) {
        __m256i a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pa + i));
        __m256i b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pb + i));
        __m256i a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pa + i + 32));
        __m256i b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pb + i + 32));

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i),
            _mm256_xor_si256(vbase, _mm256_and_si256(_mm256_or_si256(a0, b0), vdelta)));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i + 32),
            _mm256_xor_si256(vbase, _mm256_and_si256(_mm256_or_si256(a1, b1), vdelta)));
    }
    for (; i + 32 <= n; i += 32) {
        __m256i a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pa + i));
        __m256i b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pb + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i),
            _mm256_xor_si256(vbase, _mm256_and_si256(_mm256_or_si256(a0, b0), vdelta)));
    }
    for (; i < n; ++i) {
        dst[i] = kBase ^ ((pa[i] | pb[i]) & kDelta);
    }
}
#endif

} // namespace

void initialize_bitwise(bitwise_args *args, const size_t size,
                                  const std::uint_fast64_t seed) {
    if (!args) {
        return;
    }

    constexpr std::int8_t LOWER_BOUND = std::numeric_limits<std::int8_t>::min();
    constexpr std::int8_t UPPER_BOUND = std::numeric_limits<std::int8_t>::max();

    std::mt19937_64 gen(seed);
    std::uniform_int_distribution<int> dist(LOWER_BOUND, UPPER_BOUND);

    args->a.resize(size);
    args->b.resize(size);
    args->result.resize(size);

    for (std::size_t i = 0; i < size; ++i) {
        args->a[i] = static_cast<std::int8_t>(dist(gen));
        args->b[i] = static_cast<std::int8_t>(dist(gen));
        args->result[i] = 0;
    }
}


// The reference implementation of bitwise
// Student should not change this function
void naive_bitwise(std::span<std::int8_t> result,
                   std::span<const std::int8_t> a,
                   std::span<const std::int8_t> b) {
    constexpr std::uint8_t kMaskLo = 0x5Au;
    constexpr std::uint8_t kMaskHi = 0xC3u;

    const std::size_t n = std::min({result.size(), a.size(), b.size()});
    for (std::size_t i = 0; i < n; ++i) {
        const auto ua = static_cast<std::uint8_t>(a[i]);
        const auto ub = static_cast<std::uint8_t>(b[i]);

        const auto shared = static_cast<std::uint8_t>(ua & ub);
        const auto either = static_cast<std::uint8_t>(ua | ub);
        const auto diff = static_cast<std::uint8_t>(ua ^ ub);
        const auto mixed0 =
            static_cast<std::uint8_t>((diff & kMaskLo) | (~shared & ~kMaskLo));
        const auto mixed1 = static_cast<std::uint8_t>(
            ((either ^ kMaskHi) & (shared | ~kMaskHi)) ^ diff);

        result[i] = static_cast<std::int8_t>(mixed0 ^ mixed1);
    }
}

// TODO: Optimize the bitwise function
void stu_bitwise(std::span<std::int8_t> result, std::span<const std::int8_t> a,
                 std::span<const std::int8_t> b) {
    std::size_t n = std::min(result.size(), a.size());
    n = std::min(n, b.size());
    auto *__restrict__ dst = reinterpret_cast<std::uint8_t*>(result.data());
    const auto *__restrict__ pa = reinterpret_cast<const std::uint8_t*>(a.data());
    const auto *__restrict__ pb = reinterpret_cast<const std::uint8_t*>(b.data());

#if defined(BITWISE_HAS_X86) && (defined(__GNUC__) || defined(__clang__))
    if (__builtin_cpu_supports("avx2")) {
        bitwise_avx2(dst, pa, pb, n);
        return;
    }
#endif

    std::size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        std::uint64_t a0, b0, a1, b1, a2, b2, a3, b3;
        std::memcpy(&a0, pa + i,      8);
        std::memcpy(&b0, pb + i,      8);
        std::memcpy(&a1, pa + i + 8,  8);
        std::memcpy(&b1, pb + i + 8,  8);
        std::memcpy(&a2, pa + i + 16, 8);
        std::memcpy(&b2, pb + i + 16, 8);
        std::memcpy(&a3, pa + i + 24, 8);
        std::memcpy(&b3, pb + i + 24, 8);

        std::uint64_t r0 = kBase64 ^ ((a0 | b0) & kDelta64);
        std::uint64_t r1 = kBase64 ^ ((a1 | b1) & kDelta64);
        std::uint64_t r2 = kBase64 ^ ((a2 | b2) & kDelta64);
        std::uint64_t r3 = kBase64 ^ ((a3 | b3) & kDelta64);

        std::memcpy(dst + i,      &r0, 8);
        std::memcpy(dst + i + 8,  &r1, 8);
        std::memcpy(dst + i + 16, &r2, 8);
        std::memcpy(dst + i + 24, &r3, 8);
    }
    for (; i + 8 <= n; i += 8) {
        std::uint64_t va, vb;
        std::memcpy(&va, pa + i, 8);
        std::memcpy(&vb, pb + i, 8);
        std::uint64_t res = kBase64 ^ ((va | vb) & kDelta64);
        std::memcpy(dst + i, &res, 8);
    }
    for (; i < n; ++i) {
        dst[i] = kBase ^ ((pa[i] | pb[i]) & kDelta);
    }
}

void naive_bitwise_wrapper(void *ctx) {
    auto &args = *static_cast<bitwise_args *>(ctx);
    naive_bitwise(args.result, args.a, args.b);
}

void stu_bitwise_wrapper(void *ctx) {
    // Call your verion here
    auto &args = *static_cast<bitwise_args *>(ctx);
    stu_bitwise(args.result, args.a, args.b);
}

bool bitwise_check(void *stu_ctx, void *ref_ctx, lab_test_func naive_func) {
    // Compute reference
    naive_func(ref_ctx);

    auto &stu_args = *static_cast<bitwise_args *>(stu_ctx);
    auto &ref_args = *static_cast<bitwise_args *>(ref_ctx);

    if (stu_args.result.size() != ref_args.result.size()) {
        debug_log("\tDEBUG: size mismatch: stu={} ref={}\n",
                  stu_args.result.size(),
                  ref_args.result.size());
        return false;
    }

    std::int32_t max_abs_diff = 0;
    size_t worst_i = 0;

    for (size_t i = 0; i < ref_args.result.size(); ++i) {
        const auto r = static_cast<std::int32_t>(ref_args.result[i]);
        const auto s = static_cast<std::int32_t>(stu_args.result[i]);

        if (r != s) {
            max_abs_diff = std::abs(r - s);
            worst_i = i;

            debug_log("\tDEBUG: fail at {}: ref={} stu={} abs_diff={}\n",
                      i,
                      r,
                      s,
                      max_abs_diff);
            return false;
        }
    }

    debug_log("\tDEBUG: bitwise_check passed. max_abs_diff={} at i={}\n",
              max_abs_diff,
              worst_i);
    return true;
}
