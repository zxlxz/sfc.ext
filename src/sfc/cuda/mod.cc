#include <cuda.h>

#include "sfc/math/vec.h"
#include "sfc/cuda/stream.h"

namespace sfc::cuda {

static thread_local dim3_t _tls_work_size{1, 1, 1};
static thread_local dim3_t _tls_block_size{1, 1, 1};

auto block_dim() -> dim3_t {
  return _tls_block_size;
}

auto grid_dim() -> dim3_t {
  const auto ws = _tls_work_size;
  const auto bs = _tls_block_size;
  const auto gs = dim3_t{
      (ws.x + bs.x - 1) / bs.x,
      (ws.y + bs.y - 1) / bs.y,
      (ws.z + bs.z - 1) / bs.z,
  };
  return gs;
}

void set_worksize(dim3_t ws, dim3_t bs) {
  _tls_work_size = ws;
  _tls_block_size = bs;
}

auto to_str(Error err) -> str::Str {
  const char* err_name = nullptr;
  cuGetErrorName(static_cast<CUresult>(err), &err_name);
  return str::Str(err_name ? err_name : "CUDA_ERROR_UNKNOWN");
}

}  // namespace sfc::cuda
