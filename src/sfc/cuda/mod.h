#pragma once

#include "sfc/core/mod.h"

// don't really define these macros, just to make intellisense happy
#if defined(__INTELLISENSE__) || defined(__clang_analyzer__)
#ifndef __global__
#define __host__
#define __device__
#define __global__
#endif
#endif

// don't use cuda macros directly, to avoid including cuda headers in non-cuda files
#ifdef __device__
#define __dev __device__
#else
#define __dev
#endif

namespace sfc::cuda {}
