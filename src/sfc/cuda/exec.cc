#include <cuda.h>

#include "sfc/cuda/exec.h"
#include "sfc/cuda/stream.h"
#include "sfc/cuda/error.h"

namespace sfc::cuda {

struct ExecConfig {
  dim3_t work_size = {1, 1, 1};
  dim3_t block_dim = {1, 1, 1};
  dim3_t grid_dim = {1, 1, 1};

 public:
  void set_work_size(dim3_t work_size, dim3_t block_dim) {
    const auto nx = work_size.x ? work_size.x : 1U;
    const auto ny = work_size.y ? work_size.y : 1U;
    const auto nz = work_size.z ? work_size.z : 1U;

    const auto tx = block_dim.x ? block_dim.x : 1U;
    const auto ty = block_dim.y ? block_dim.y : 1U;
    const auto tz = block_dim.z ? block_dim.z : 1U;

    const auto bx = (nx + tx - 1) / tx;
    const auto by = (ny + ty - 1) / ty;
    const auto bz = (nz + tz - 1) / tz;

    this->work_size = {nx, ny, nz};
    this->block_dim = {tx, ty, tz};
    this->grid_dim = {bx, by, bz};
  }

  auto get_trds() const -> dim3_t {
    return block_dim;
  }

  auto get_blks() const -> dim3_t {
    return grid_dim;
  }
};

static thread_local auto _tls_exec_conf = ExecConfig{};

auto get_trds() -> dim3_t {
  return _tls_exec_conf.get_trds();
}

auto get_blks() -> dim3_t {
  return _tls_exec_conf.get_blks();
}

void config(dim3_t work_size, dim3_t block_size) {
  _tls_exec_conf.set_work_size(work_size, block_size);
}

}  // namespace sfc::cuda
