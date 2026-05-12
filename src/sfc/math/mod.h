#pragma once

#include "sfc/core/mod.h"

#ifdef __device__
#define _hd __host__ __device__
#else
#define _hd
#endif

namespace sfc::math {

static constexpr float PI = 3.1415927F;

[[gnu::always_inline]] inline auto fabs(f32 x) -> f32 {
  return __builtin_fabsf(x);
}

[[gnu::always_inline]] inline auto sqrt(f32 x) -> float {
  return __builtin_sqrtf(x);
}

}  // namespace sfc::math
