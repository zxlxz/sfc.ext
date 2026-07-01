#pragma once

#ifdef __device__
#define __dev __device__
#else
#define __dev
#endif

namespace sfc::cuda {

struct dim3_t {
  unsigned x = 1;
  unsigned y = 1;
  unsigned z = 1;

 public:
#ifdef __device_builtin__
  operator ::dim3() const {
    return {x, y, z};
  }
#endif
};

auto grid_dim() -> dim3_t;
auto block_dim() -> dim3_t;
void set_worksize(dim3_t work_size, dim3_t block_size);

template <unsigned N>
void config(const unsigned (&work_size)[N], const unsigned (&block_size)[N]) {
  static_assert(N >= 1 && N <= 3, "cuda::config: work_size and block_size must be 1D, 2D or 3D");

  const auto wx = N > 0 ? work_size[0] : 1;
  const auto wy = N > 1 ? work_size[1] : 1;
  const auto wz = N > 2 ? work_size[2] : 1;

  const auto bx = N > 0 ? block_size[0] : 1;
  const auto by = N > 1 ? block_size[1] : 1;
  const auto bz = N > 2 ? block_size[2] : 1;
  cuda::set_worksize({wx, wy, wz}, {bx, by, bz});
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
