#include <cuda_runtime_api.h>

#include "sfc/math/vec.h"
#include "sfc/cuda/mod.inl"
#include "sfc/cuda/exec.h"
#include "sfc/cuda/stream.h"

namespace sfc::cuda {

struct ExecConf {
  dim3_t work_size = {1, 1, 1};
  dim3_t grid_dim = {1, 1, 1};
  dim3_t block_dim = {1, 1, 1};

 public:
  static auto instance() -> ExecConf& {
    static thread_local auto cfg = ExecConf{};
    return cfg;
  }

  void set_worksize(dim3_t ws, dim3_t bs) {
    // fix ws
    const auto wx = ws.x == 0 ? 1 : ws.x;
    const auto wy = ws.y == 0 ? 1 : ws.y;
    const auto wz = ws.z == 0 ? 1 : ws.z;

    // fix bs
    const auto bx = bs.x == 0 ? 1 : bs.x;
    const auto by = bs.y == 0 ? 1 : bs.y;
    const auto bz = bs.z == 0 ? 1 : bs.z;

    // cacl grid size
    const auto gx = (wx + bx - 1) / bx;
    const auto gy = (wy + by - 1) / by;
    const auto gz = (wz + bz - 1) / bz;

    // set config
    this->work_size = {wx, wy, wz};
    this->grid_dim = {gx, gy, gz};
    this->block_dim = {bx, by, bz};
  }
};

auto block_dim() -> dim3_t {
  return ExecConf::instance().block_dim;
}

auto grid_dim() -> dim3_t {
  return ExecConf::instance().grid_dim;
}

void set_worksize(dim3_t ws, dim3_t bs) {
  ExecConf::instance().set_worksize(ws, bs);
}

template <>
void config(math::vec<unsigned, 1> ws, math::vec<unsigned, 1> bs) {
  cuda::set_worksize({ws.x, 1, 1}, {bs.x, 1, 1});
}

template <>
void config(math::vec<unsigned, 2> ws, math::vec<unsigned, 2> bs) {
  cuda::set_worksize({ws.x, ws.y, 1}, {bs.x, bs.y, 1});
}

template <>
void config(math::vec<unsigned, 3> ws, math::vec<unsigned, 3> bs) {
  cuda::set_worksize({ws.x, ws.y, ws.z}, {bs.x, bs.y, bs.z});
}

}  // namespace sfc::cuda
