/*  Copyright (C) 2013-2014  Povilas Kanapickas <povilas@radix.lt>

    Distributed under the Boost Software License, Version 1.0.
        (See accompanying file LICENSE_1_0.txt or copy at
            http://www.boost.org/LICENSE_1_0.txt)
*/

#ifndef LIBSIMDPP_SIMDPP_CORE_I_SLLI_H
#define LIBSIMDPP_SIMDPP_CORE_I_SLLI_H

#ifndef LIBSIMDPP_SIMD_H
    #error "This file must be included through simd.h"
#endif

#include <simdpp/types.h>
#include <simdpp/capabilities.h>
#include <simdpp/detail/insn/i_slli.h>
#include <simdpp/detail/insn/i_shift_r.h>
#include <simdpp/detail/insn/i_shift_r_v.h>
#include <simdpp/detail/not_implemented.h>

namespace simdpp {
namespace SIMDPP_ARCH_NAMESPACE {

// -----------------------------------------------------------------------------
// shift by scalar
/** Shifts signed 16-bit values right by @a count bits while shifting in the
    sign bit.

    @code
    r0 = a0 >> count
    ...
    rN = aN >> count
    @endcode
*/
template<unsigned N, class E> SIMDPP_INL
int16<N,expr_empty> slli(const int16<N,E>& a, unsigned count)
{
    return detail::insn::i_slli(a.eval(), count);
}

/** Shifts signed 32-bit values right by @a count bits while shifting in the
    sign bit.

    @code
    r0 = a0 >> count
    ...
    rN = aN >> count
    @endcode
*/
template<unsigned N, class E> SIMDPP_INL
int32<N,expr_empty> slli(const int32<N,E>& a, unsigned count)
{
    return detail::insn::i_slli(a.eval(), count);
}

// -----------------------------------------------------------------------------
// shift by compile-time constant

/** Shifts signed 16-bit values right by @a count bits while shifting in the
    sign bit.

    @code
    r0 = a0 >> count
    ...
    rN = aN >> count
    @endcode
*/
template<unsigned count, unsigned N, class E> SIMDPP_INL
int16<N,expr_empty> slli(const int16<N,E>& a)
{
    return detail::insn::i_slli_wrapper<count == 0>::template run<count>(a.eval());
}

/** Shifts signed 32-bit values right by @a count bits while shifting in the
    sign bit.

    @code
    r0 = a0 >> count
    ...
    rN = aN >> count
    @endcode
*/
template<unsigned count, unsigned N, class E> SIMDPP_INL
int32<N,expr_empty> slli(const int32<N,E>& a)
{
    return detail::insn::i_slli_wrapper<count == 0>::template run<count>(a.eval());
}


} // namespace SIMDPP_ARCH_NAMESPACE
} // namespace simdpp

#endif

