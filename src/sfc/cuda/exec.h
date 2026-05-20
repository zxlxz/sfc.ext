#pragma once

#include "sfc/core/mod.h"

#ifdef __device__
#define __dev __device__
#else
#define __dev
#endif

struct CUstream_st;

namespace sfc::math {
template <class T, int N>
struct vec;
}

namespace sfc::cuda {

struct dim3_t {
  unsigned x = 1;
  unsigned y = 1;
  unsigned z = 1;

 public:
#ifdef __CUDACC__
  operator ::dim3() const {
    return {x, y, z};
  }
#endif
};

auto get_blks() -> dim3_t;
auto get_trds() -> dim3_t;
void set_worksize(dim3_t work_size, dim3_t block_size);

template<int N>
void config(math::vec<unsigned, N> work_size, math::vec<unsigned, N> block_size);

}  // namespace sfc::cuda

#if defined(__INTELLISENSE__) || defined(__clang_analyzer__)
extern const sfc::cuda::dim3_t gridDim;
extern const sfc::cuda::dim3_t blockIdx;
extern const sfc::cuda::dim3_t blockDim;
extern const sfc::cuda::dim3_t threadIdx;
#endif

#ifdef __CUDACC__
#define CUDA_EXEC(f) f<<<sfc::cuda::get_blks(), sfc::cuda::get_trds()>>>
#else
#define CUDA_EXEC(f) f
#endif
