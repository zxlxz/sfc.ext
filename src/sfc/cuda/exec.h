#pragma once

#include "sfc/cuda/mod.h"

struct CUstream_st;

namespace sfc::math {
template<class T, int N>
struct vec;
}

namespace sfc::cuda {

struct dim3_t {
  unsigned x = 1;
  unsigned y = 1;
  unsigned z = 1;

 public:
  dim3_t(unsigned x, unsigned y = 1, unsigned z = 1) : x{x}, y{y}, z{z} {}

  template <int N>
  dim3_t(const math::vec<unsigned, N>& v) {
    static_assert(N > 0 && N <= 3, "sfc::cuda::dime_t: dims must be 1, 2 or 3");
    if constexpr (N > 0) this->x = v.x;
    if constexpr (N > 1) this->y = v.y;
    if constexpr (N > 2) this->z = v.z;
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
#define __global__
#define _device__
extern const sfc::cuda::dim3_t gridDim;
extern const sfc::cuda::dim3_t blockIdx;
extern const sfc::cuda::dim3_t blockDim;
extern const sfc::cuda::dim3_t threadIdx;
#endif
