#pragma once

#include <math.h>
#include <sfc/core/mod.h>

#ifdef __CUDACC__
#define __hd __host__ __device__
#else
#define __hd
#endif

namespace sfc::math {

static constexpr f64 PI = 3.14159265358979323846;

using ::fabs;
using ::fabsf;

using ::sqrt;
using ::sqrtf;

using ::hypot;
using ::hypotf;

using ::exp;
using ::expf;

using ::pow;
using ::powf;

using ::sin;
using ::sinf;

using ::cos;
using ::cosf;

using ::tan;
using ::tanf;

using ::asin;
using ::asinf;

using ::acos;
using ::acosf;

using ::atan2;
using ::atan2f;

template <class T>
auto clamp(T x, T min_val, T max_val) -> T {
  if (x < min_val) return min_val;
  if (x > max_val) return max_val;
  return x;
}

}  // namespace sfc::math
