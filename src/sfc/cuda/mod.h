#pragma once

#include "sfc/core/mod.h"

#ifdef __INTELLISENSE__
#ifndef __device__
#define __device__
#endif
#ifndef __global__
#define __global__
#endif
#endif

#ifdef __device__
#define __dev __device__
#define __hd __host__ __device__
#else
#define __dev
#define __hd
#endif

struct dim3;

namespace sfc::cuda {

enum class Error;
auto to_str(Error err) -> const char*;

template <class T = Unit>
using Result = result::Result<T, Error>;

struct dim3_t {
  u32 x = 1;
  u32 y = 1;
  u32 z = 1;
};

auto grid_dim() -> dim3;
auto block_dim() -> dim3;

template <u32 N>
void config(const u32 (&work_size)[N], const u32 (&block_size)[N]);

}  // namespace sfc::cuda

#if defined(__INTELLISENSE__) || defined(__clang_analyzer__)
extern const sfc::cuda::dim3_t gridDim;
extern const sfc::cuda::dim3_t blockIdx;
extern const sfc::cuda::dim3_t blockDim;
extern const sfc::cuda::dim3_t threadIdx;
#endif

#ifdef __CUDACC__
#define CUDA_EXEC(f) f<<<sfc::cuda::grid_dim(), sfc::cuda::block_dim()>>>
#else
#define CUDA_EXEC(f) f
#endif
