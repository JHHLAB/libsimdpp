/*  Copyright (C) 2012  Povilas Kanapickas <povilas@radix.lt>

    Distributed under the Boost Software License, Version 1.0.
        (See accompanying file LICENSE_1_0.txt or copy at
            http://www.boost.org/LICENSE_1_0.txt)
*/

#ifndef LIBSIMDPP_DETAIL_NEON_SHUFFLE_H
#define LIBSIMDPP_DETAIL_NEON_SHUFFLE_H
#if SIMDPP_USE_NEON || defined(SIMDPP_DOXYGEN)

#ifndef LIBSIMDPP_SIMD_H
    #error "This file must be included through simd.h"
#endif

#include <simdpp/types.h>

namespace simdpp {
namespace SIMDPP_ARCH_NAMESPACE {
namespace neon {

/** @code
    a0 = b0
    a1 = a1
    b0 = a0
    b1 = b1
    @endcode
*/
SIMDPP_INL void swap_lo(uint64x2& a, uint64x2& b)
{
    int64x1_t ah, bh, al, bl;
    al = vget_low_u64(a);
    bl = vget_low_u64(b);
    ah = vget_high_u64(a);
    bh = vget_high_u64(b);
    a = vcombine_u64(bl, ah);
    b = vcombine_u64(al, bh);
}

/** @code
    a0 = a0
    a1 = b1
    b0 = b0
    b1 = a1
    @endcode
*/
SIMDPP_INL void swap_hi(uint64x2& a, uint64x2& b)
{
    int64x1_t ah, bh, al, bl;
    al = vget_low_u64(a);
    bl = vget_low_u64(b);
    ah = vget_high_u64(a);
    bh = vget_high_u64(b);
    a = vcombine_u64(al, bh);
    b = vcombine_u64(bl, ah);
}

SIMDPP_INL void transpose2(uint64x2& a, uint64x2& b)
{
    int64x1_t ah, bh, al, bl;
    al = vget_low_u64(a);
    bl = vget_low_u64(b);
    ah = vget_high_u64(a);
    bh = vget_high_u64(b);
    a = vcombine_u64(al, bl);
    b = vcombine_u64(ah, bh);
}

SIMDPP_INL uint64x2 zip2_lo(const uint64x2& ca, const uint64x2& cb)
{
    uint64<2> a = ca, b = cb;
    transpose2(a, b);
    return a;
}

SIMDPP_INL uint64x2 zip2_hi(const uint64x2& ca, const uint64x2& cb)
{
    uint64<2> a = ca, b = cb;
    transpose2(a, b);
    return b;
}

} // namespace neon
} // namespace SIMDPP_ARCH_NAMESPACE
} // namespace simdpp

#endif
#endif
