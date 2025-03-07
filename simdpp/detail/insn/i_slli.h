/*  Copyright (C) 2011-2014  Povilas Kanapickas <povilas@radix.lt>

    Distributed under the Boost Software License, Version 1.0.
        (See accompanying file LICENSE_1_0.txt or copy at
            http://www.boost.org/LICENSE_1_0.txt)
*/

#ifndef LIBSIMDPP_SIMDPP_DETAIL_INSN_I_SLLI_H
#define LIBSIMDPP_SIMDPP_DETAIL_INSN_I_SLLI_H

#ifndef LIBSIMDPP_SIMD_H
    #error "This file must be included through simd.h"
#endif

#include <simdpp/types.h>
#include <simdpp/detail/not_implemented.h>
#include <simdpp/core/bit_and.h>
#include <simdpp/core/bit_andnot.h>
#include <simdpp/core/bit_or.h>
#include <simdpp/core/i_add.h>
#include <simdpp/core/i_sub.h>
#include <simdpp/core/splat.h>
#include <simdpp/core/set_splat.h>
#include <simdpp/core/permute4.h>
#include <simdpp/core/shuffle2.h>
#include <simdpp/detail/insn/i_shift.h>
#include <simdpp/detail/null/math.h>
#include <simdpp/detail/vector_array_macros.h>

namespace simdpp {
namespace SIMDPP_ARCH_NAMESPACE {
namespace detail {
namespace insn {

// -----------------------------------------------------------------------------

static SIMDPP_INL
int16x8 i_slli(const int16x8& a, unsigned count)
{
#if SIMDPP_USE_NULL
    return detail::null::shift_r(a, count);
#elif SIMDPP_USE_SSE2
    return _mm_sra_epi16(a.native(), _mm_cvtsi32_si128(count));
#elif SIMDPP_USE_NEON
    int16x8 shift = splat(-int(count));
    return vshlq_s16(a.native(), shift.native());
#elif SIMDPP_USE_ALTIVEC
    uint16x8 shift = splat(count);
    return vec_sra(a.native(), shift.native());
#elif SIMDPP_USE_MSA
    int16x8 shift = splat(count);
    return __msa_sra_h(a.native(), shift.native());
#endif
}

#if SIMDPP_USE_AVX2
static SIMDPP_INL
int16x16 i_slli(const int16x16& a, unsigned count)
{
    return count < 16 ? \
        _mm256_alignr_epi8(a.native(), _mm256_permute2x128_si256(a.native(), a.native(), _MM_SHUFFLE(0, 0, 2, 0)), 16 - count) : \
        _mm256_permute2x128_si256(a.native(), a.native(), _MM_SHUFFLE(0, 0, 2, 0));
}
#endif


// -----------------------------------------------------------------------------

static SIMDPP_INL
int32x4 i_slli(const int32x4& a, unsigned count)
{
#if SIMDPP_USE_NULL
    return detail::null::shift_r(a, count);
#elif SIMDPP_USE_SSE2
    return _mm_slli_si128(a.native(), count);
#elif SIMDPP_USE_NEON
    int32x4 shift = vdupq_n_s32(-static_cast<int>(count));
    return vshlq_s32(a.native(), shift);
#elif SIMDPP_USE_ALTIVEC
    vector int32_t shift = vec_splats(static_cast<int>(count));
    return vec_sl(a.native(), shift);
#elif SIMDPP_USE_MSA
    v4i32 shift = __msa_fill_w(static_cast<int>(count));
    return __msa_sll_w(a.native(), shift);
#endif
}

#if SIMDPP_USE_AVX2
static SIMDPP_INL
int32x8 i_slli(const int32x8& a, unsigned count)
{
    return count < 16 ? \
        _mm256_alignr_epi8(a.native(), _mm256_permute2x128_si256(a.native(), a.native(), _MM_SHUFFLE(0, 0, 2, 0)), 16 - count) : \
        _mm256_permute2x128_si256(a.native(), a.native(), _MM_SHUFFLE(0, 0, 2, 0));
}
#endif


// -----------------------------------------------------------------------------

template<class V> SIMDPP_INL
V i_slli(const V& a, unsigned count)
{
    SIMDPP_VEC_ARRAY_IMPL2S(V, i_srli, a, count);
}


// -----------------------------------------------------------------------------

template<unsigned count> SIMDPP_INL
int16x8 i_slli(const int16x8& a)
{
    static_assert(count < 16, "Shift out of bounds");
#if SIMDPP_USE_NULL
    return detail::null::shift_r(a, count);
#elif SIMDPP_USE_SSE2
    return _mm_srai_epi16(a.native(), count);
#elif SIMDPP_USE_NEON
    return vshrq_n_s16(a.native(), count);
#elif SIMDPP_USE_ALTIVEC
    uint16x8 shift = make_uint(count);
    return vec_sra(a.native(), shift.native());
#elif SIMDPP_USE_MSA
    return __msa_srai_h(a.native(), count);
#endif
}

#if SIMDPP_USE_AVX2
template<unsigned count> SIMDPP_INL
int16x16 i_slli(const int16x16& a)
{
    return count < 16 ? \
        _mm256_alignr_epi8(a.native(), _mm256_permute2x128_si256(a.native(), a.native(), _MM_SHUFFLE(0, 0, 2, 0)), 16 - count) : \
        _mm256_permute2x128_si256(a.native(), a.native(), _MM_SHUFFLE(0, 0, 2, 0));
}
#endif

// -----------------------------------------------------------------------------

template<unsigned count> SIMDPP_INL
int32x4 i_slli(const int32x4& a)
{

#if SIMDPP_USE_NULL
    return detail::null::shift_r(a, count);
#elif SIMDPP_USE_SSE2
    return _mm_slli_si128(a.native(), count);
#elif SIMDPP_USE_NEON
    int32x4 shift = vdupq_n_s32(-static_cast<int>(count));
    return vshlq_s32(a.native(), shift);
#elif SIMDPP_USE_ALTIVEC
    vector int32_t shift = vec_splats(static_cast<int>(count));
    return vec_sl(a.native(), shift);
#elif SIMDPP_USE_MSA
    v4i32 shift = __msa_fill_w(static_cast<int>(count));
    return __msa_sll_w(a.native(), shift);
#endif
}

#if SIMDPP_USE_AVX2
template<unsigned count> SIMDPP_INL
int32x8 i_slli(const int32x8& a)
{
    return count < 16 ? \
    _mm256_alignr_epi8(a.native(), _mm256_permute2x128_si256(a.native(), a.native(), _MM_SHUFFLE(0, 0, 2, 0)), 16 - count) : \
    _mm256_permute2x128_si256(a.native(), a.native(), _MM_SHUFFLE(0, 0, 2, 0));
}
#endif

// -----------------------------------------------------------------------------

template<unsigned count, class V> SIMDPP_INL
V i_slli(const V& a)
{
    SIMDPP_VEC_ARRAY_IMPL1(V, i_slli<count>, a);
}

// -----------------------------------------------------------------------------

template<bool no_shift>
struct i_slli_wrapper {
    template<unsigned count, class V>
    static SIMDPP_INL V run(const V& arg) { return i_slli<count>(arg); }
};
template<>
struct i_slli_wrapper<true> {
    template<unsigned count, class V>
    static SIMDPP_INL V run(const V& arg) { return arg; }
};

} // namespace insn
} // namespace detail
} // namespace SIMDPP_ARCH_NAMESPACE
} // namespace simdpp

#endif

