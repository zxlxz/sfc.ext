#pragma once

#include <sfc/core/ops.h>

#ifdef __CUDACC__
#define __hd __host__ __device__
#else
#define __hd
#endif

namespace sfc::math {

static constexpr f32 PI = 3.1415927F;
static constexpr f32 E = 2.7182817F;
static constexpr f32 Ln2 = 0.6931472F;
static constexpr f32 Ln10 = 2.302585F;

// AVX: vroundps(0)
inline auto roundf(f32 x) -> f32 {
  return __builtin_roundf(x);
}

// AVX: vroundps(1)
inline auto floorf(f32 x) -> f32 {
  return __builtin_floorf(x);
}

// AVX: vroundps(2)
inline auto ceilf(f32 x) -> f32 {
  return __builtin_ceilf(x);
}

// AVX: andps(0x7FFFFFFF)
inline auto fabsf(f32 x) -> f32 {
  return __builtin_fabsf(x);
}

// AVX: vsqrtps
inline auto sqrtf(f32 x) -> f32 {
  return __builtin_sqrtf(x);
}

// libm: exp
inline auto expf(f32 x) -> f32 {
  return __builtin_expf(x);
}

// libm: log
inline auto logf(f32 x) -> f32 {
  return __builtin_logf(x);
}

// libm: log2
inline auto log2f(f32 x) -> f32 {
  return __builtin_log2f(x);
}

// libm: log10
inline auto log10f(f32 x) -> f32 {
  return __builtin_log10f(x);
}

// libm: pow
inline auto powf(f32 x, f32 y) -> f32 {
  return __builtin_powf(x, y);
}

// libm: hypot
inline auto hypotf(f32 x, f32 y) -> f32 {
  return __builtin_hypotf(x, y);
}

// libm: sin
inline auto sinf(f32 x) -> f32 {
  return __builtin_sinf(x);
}

// libm: cos
inline auto cosf(f32 x) -> f32 {
  return __builtin_cosf(x);
}

// libm: tan
inline auto tanf(f32 x) -> f32 {
  return __builtin_tanf(x);
}

// libm: asin
inline auto asinf(f32 x) -> f32 {
  return __builtin_asinf(x);
}

// libm: acos
inline auto acosf(f32 x) -> f32 {
  return __builtin_acosf(x);
}

// libm: atan2
inline auto atan2f(f32 y, f32 x) -> f32 {
  return __builtin_atan2f(y, x);
}

template <class T>
auto clamp(T x, T min_val, T max_val) -> T {
  if (x < min_val) return min_val;
  if (x > max_val) return max_val;
  return x;
}

}  // namespace sfc::math
