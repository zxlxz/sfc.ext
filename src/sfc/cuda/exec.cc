#include <cuda.h>

#include "sfc/cuda/exec.h"
#include "sfc/cuda/stream.h"
#include "sfc/cuda/error.h"

namespace sfc::cuda {

struct ExecConfig {
  dim3_t work_size = {1, 1, 1};
  dim3_t block_dim = {1, 1, 1};
  dim3_t grid_dim = {1, 1, 1};
};

static thread_local auto _tls_exec_conf = ExecConfig{};

auto get_trds() -> dim3_t {
  return _tls_exec_conf.block_dim;
}

auto get_blks() -> dim3_t {
  return _tls_exec_conf.grid_dim;
}

void config(dim3_t work_size, dim3_t block_size) {
  const auto nx = work_size.x ? work_size.x : 1U;
  const auto ny = work_size.y ? work_size.y : 1U;
  const auto nz = work_size.z ? work_size.z : 1U;

  const auto tx = block_size.x ? block_size.x : 1U;
  const auto ty = block_size.y ? block_size.y : 1U;
  const auto tz = block_size.z ? block_size.z : 1U;

  const auto bx = (nx + tx - 1) / tx;
  const auto by = (ny + ty - 1) / ty;
  const auto bz = (nz + tz - 1) / tz;

  _tls_exec_conf.work_size = {nx, ny, nz};
  _tls_exec_conf.block_dim = {tx, ty, tz};
  _tls_exec_conf.grid_dim = {bx, by, bz};
}

}  // namespace sfc::cuda
