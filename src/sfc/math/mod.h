#pragma once

#if !defined(__clang__) && !defined(__GNUC__)
#include <math.h>
#define __builtin_cosf   ::cosf
#define __builtin_sinf   ::sinf
#define __builtin_sqrtf  ::sqrtf
#endif

#ifdef __CUDACC__
#define _hd __host__ __device__
#else
#define _hd
#endif

namespace sfc::math {

using i8 = signed char;
using i16 = short;
using i32 = int;

using u8 = unsigned char;
using u16 = unsigned short;
using u32 = unsigned int;

using f32 = float;
using f64 = double;

using usize = decltype(sizeof(0));

static constexpr float PI = 3.1415927F;
}  // namespace sfc::math
