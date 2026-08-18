#include <cuda.h>

#include "sfc/cuda/stream.h"
#include "sfc/cuda/library.h"

namespace sfc::cuda {

static auto lib_load(const char* path) -> Result<lib_t> {
  if (path == nullptr) {
    return Error(CUDA_ERROR_INVALID_VALUE);
  }

  auto lib = lib_t{nullptr};
  if (auto err = cuModuleLoad(&lib, path); err != CUDA_SUCCESS) {
    return Error(err);
  }
  return Ok{lib};
}

static auto lib_unload(lib_t lib) -> Result<> {
  if (lib == nullptr) {
    return Ok{};
  }

  if (auto err = cuModuleUnload(lib); err != CUDA_SUCCESS) {
    return Error(err);
  }

  return Ok{};
}

static auto lib_kernel(lib_t lib, const char* name) -> Result<kernel_t> {
  if (lib == nullptr || name == nullptr) {
    return Error(CUDA_ERROR_INVALID_VALUE);
  }

  auto func = kernel_t{nullptr};
  if (auto err = cuModuleGetFunction(&func, lib, name); err != CUDA_SUCCESS) {
    return Error(err);
  }
  return Ok{func};
}

auto launch_kernel(kernel_t f, void** args) -> Result<> {
  if (f == nullptr) {
    return Error(CUDA_ERROR_INVALID_VALUE);
  }

  const auto stream = cuda::stream_current();
  const auto grid_dim = cuda::grid_dim();
  const auto block_dim = cuda::block_dim();
  const auto config = CUlaunchConfig{
      grid_dim.x,
      grid_dim.y,
      grid_dim.z,
      block_dim.x,
      block_dim.y,
      block_dim.z,
      0,
      stream,
      nullptr,
      0,
  };
  if (auto err = cuLaunchKernelEx(&config, f, args, nullptr); err != CUDA_SUCCESS) {
    return Error(err);
  }
  return Ok{};
}

Library::Library() noexcept = default;

Library::~Library() noexcept {
  if (_lib == nullptr) {
    return;
  }

  cuda::lib_unload(_lib).unwrap();
  _lib = nullptr;
}

Library::Library(Library&& other) noexcept : _lib{mem::take(other._lib)} {}

auto Library::load(const char* path) -> Library {
  auto lib = cuda::lib_load(path).unwrap();

  auto res = Library{};
  res._lib = lib;
  return res;
}

auto Library::get_kernel(const char* name) const -> Result<kernel_t> {
  return cuda::lib_kernel(_lib, name);
}

}  // namespace sfc::cuda
