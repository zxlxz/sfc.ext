#include <cuda.h>

#include "sfc/cuda/error.h"

namespace sfc::cuda {

auto Error::name() const noexcept -> const char* {
  const char* res = nullptr;
  if (auto err = ::cuGetErrorName(CUresult(_code), &res)) {
    return "CUDA_ERROR_UNKNOWN";
  }
  return res;
}

}  // namespace sfc::cuda
