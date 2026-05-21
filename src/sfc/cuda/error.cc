#include <cuda.h>

#include "sfc/cuda/error.h"

namespace sfc::cuda {

auto Error::to_str() const -> cstr_t {
  const auto err_code = static_cast<CUresult>(_code);

  auto err_name = cstr_t{nullptr};
  if (auto err = ::cuGetErrorName(err_code, &err_name)) {
    return "CUDA_ERROR_UNKNOWN";
  }
  return err_name;
}

}  // namespace sfc::cuda
