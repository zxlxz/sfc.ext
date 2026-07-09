#include <cuda_runtime_api.h>

#include "sfc/math/vec.h"
#include "sfc/cuda/mod.h"
#include "sfc/cuda/stream.h"

namespace sfc::cuda {

static thread_local dim3 _tls_work_size{1, 1, 1};
static thread_local dim3 _tls_block_size{1, 1, 1};

auto block_dim() -> dim3 {
  return _tls_block_size;
}

auto grid_dim() -> dim3 {
  const auto ws = _tls_work_size;
  const auto bs = _tls_block_size;
  const auto gs = dim3{
      (ws.x + bs.x - 1) / bs.x,
      (ws.y + bs.y - 1) / bs.y,
      (ws.z + bs.z - 1) / bs.z,
  };
  return gs;
}

void set_worksize(dim3 ws, dim3 bs) {
  _tls_work_size = ws;
  _tls_block_size = bs;
}

template <u32 N>
void config(const u32 (&work_size)[N], const u32 (&block_size)[N]) {
  const auto wx = N > 0 ? work_size[0] : 1;
  const auto wy = N > 1 ? work_size[1] : 1;
  const auto wz = N > 2 ? work_size[2] : 1;

  const auto bx = N > 0 ? block_size[0] : 1;
  const auto by = N > 1 ? block_size[1] : 1;
  const auto bz = N > 2 ? block_size[2] : 1;

  const auto ws = dim3{wx, wy, wz};
  const auto bs = dim3{bx, by, bz};
  cuda::set_worksize(ws, bs);
}

template void config(const u32 (&)[1], const u32 (&)[1]);
template void config(const u32 (&)[2], const u32 (&)[2]);
template void config(const u32 (&)[3], const u32 (&)[3]);

auto to_str(Error err) -> const char* {
  const auto err_code = cudaError(err);
  const auto err_name = cudaGetErrorName(err_code);
  return err_name;
}

}  // namespace sfc::cuda
