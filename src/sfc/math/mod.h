#pragma once

#include <math.h>
#include "sfc/core.h"

#ifdef __CUDACC__
#define __hd __host__ __device__
#else
#define __hd
#endif

namespace sfc::math {

static constexpr float PI = 3.1415927F;

using ::fabsf;
using ::sqrtf;
using ::hypotf;

using ::sinf;
using ::cosf;
using ::tanf;

}  // namespace sfc::math
