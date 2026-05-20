#pragma once

#include "sfc/core/mod.h"

struct CUstream_st;

namespace sfc::math {
template <class T, int N>
struct vec;
}

namespace sfc::cuda {

struct dim3_t {
  unsigned x;
  unsigned y;
  unsigned z;

 public:
  dim3_t(unsigned x, unsigned y, unsigned z = 1) : x{x}, y{y}, z{z} {}

  template <int N>
  dim3_t(const math::vec<unsigned, N>& v) : x{1}, y{1}, z{1} {
    static_assert(N >= 1 && N <= 3, "invalid dimension");
    if constexpr (N >= 1) x = v.x;
    if constexpr (N >= 2) y = v.y;
    if constexpr (N >= 3) z = v.z;
  }

#ifdef __CUDACC__
  operator ::dim3() const {
    return {x, y, z};
  }
#endif
};

auto get_blks() -> dim3_t;
auto get_trds() -> dim3_t;
void config(dim3_t work_size, dim3_t block_size);

}  // namespace sfc::cuda

#if defined(__INTELLISENSE__) || defined(__clang_analyzer__)
#define CUDA_EXEC(f) f
#else
#define CUDA_EXEC(f) f<<<sfc::cuda::get_blks(), sfc::cuda::get_trds()>>>
#endif

#if defined(__INTELLISENSE__) || defined(__clang_analyzer__)
extern const sfc::cuda::dim3_t gridDim;
extern const sfc::cuda::dim3_t blockIdx;
extern const sfc::cuda::dim3_t blockDim;
extern const sfc::cuda::dim3_t threadIdx;
#endif
