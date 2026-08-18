#pragma once

#include "sfc/core.h"

#if defined(__INTELLISENSE__) || defined(__clang_analyzer__)
#ifndef __device__
#define __device__
#endif
#ifndef __global__
#define __global__
#endif
#endif

#ifdef __device__
#define __dev __device__
#else
#define __dev
#endif

namespace sfc::cuda {

enum class Error;
auto to_str(Error err) -> str::Str;

template <class T = Unit>
using Result = result::Result<T, Error>;

struct dim3_t {
  u32 x = 1;
  u32 y = 1;
  u32 z = 1;

  template <u32 N>
  static auto from(const u32 (&n)[N]) -> dim3_t {
    static_assert(N <= 3);
    return dim3_t{N > 0 ? n[0] : 1, N > 1 ? n[1] : 1, N > 2 ? n[2] : 1};
  }
};

auto grid_dim() -> dim3_t;
auto block_dim() -> dim3_t;
void set_worksize(dim3_t ws, dim3_t bs);

template <u32 N>
void config(const u32 (&work_size)[N], const u32 (&block_size)[N]) {
  const auto work_dim = dim3_t::from(work_size);
  const auto block_dim = dim3_t::from(block_size);
  cuda::set_worksize(work_dim, block_dim);
}

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
