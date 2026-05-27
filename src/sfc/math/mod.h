#pragma once

#include <math.h>
#include "sfc/core.h"

#ifdef __CUDACC__
#define __hd __host__ __device__
#else
#define __hd
#endif

namespace sfc::math {

static constexpr float PI = 3.14159265358979323846f;

using ::fabsf;
using ::sqrtf;
using ::hypotf;

using ::sinf;
using ::cosf;
using ::tanf;

}  // namespace sfc::math
