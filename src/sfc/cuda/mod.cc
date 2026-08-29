#include <cuda.h>

#include "sfc/math/vec.h"
#include "sfc/cuda/stream.h"

namespace sfc::cuda {

namespace detail {

auto err_name(CUresult err) -> const char* {
  const char* name = "CUDA_ERROR_UNKNOWN";
  (void)cuGetErrorName(err, &name);
  return name;
}

}  // namespace detail

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
  ws.x = (ws.x == 0) ? 1 : ws.x;
  ws.y = (ws.y == 0) ? 1 : ws.y;
  ws.z = (ws.z == 0) ? 1 : ws.z;

  bs.x = (bs.x == 0) ? 1 : bs.x;
  bs.y = (bs.y == 0) ? 1 : bs.y;
  bs.z = (bs.z == 0) ? 1 : bs.z;

  _tls_work_size = ws;
  _tls_block_size = bs;
}

auto to_str(Error err) -> str::Str {
  const auto err_code = static_cast<CUresult>(err);
  const auto err_name = detail::err_name(err_code);
  return err_name;
}

}  // namespace sfc::cuda
