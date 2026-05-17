#pragma once

#include "sfc/core/mod.h"

#ifdef __CUDACC__
#define __hd __host__ __device__
#else
#define __hd
#endif

namespace sfc::math {

static constexpr float PI = 3.1415927F;

__hd inline auto fabsf(f32 x) -> f32 {
  return __builtin_fabsf(x);
}

__hd inline auto sqrtf(f32 x) -> float {
  return __builtin_sqrtf(x);
}

__hd inline auto hypotf(f32 x, f32 y) -> f32 {
  return __builtin_sqrtf(x * x + y * y);
}

}  // namespace sfc::math
