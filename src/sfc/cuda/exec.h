#pragma once

#include "sfc/cuda/mod.h"
#include "sfc/math/vec.h"

struct CUstream_st;

namespace sfc::cuda {

struct dim3_t {
  unsigned x = 1;
  unsigned y = 1;
  unsigned z = 1;

 public:
  dim3_t(unsigned x, unsigned y, unsigned z = 1) : x{x}, y{y}, z{z} {}

  dim3_t(const math::vec<unsigned, 1>& v) : x{v.x} {}
  dim3_t(const math::vec<unsigned, 2>& v) : x{v.x}, y{v.y} {}
  dim3_t(const math::vec<unsigned, 3>& v) : x{v.x}, y{v.y}, z{v.z} {}

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
