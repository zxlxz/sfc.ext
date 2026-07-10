#include <cuda_runtime_api.h>

#include "sfc/core.h"
#include "sfc/cuda/stream.h"
#include "sfc/cuda/library.h"

namespace sfc::cuda {

static auto lib_load(const char* path) -> Result<lib_t> {
  if (path == nullptr) {
    return Error(cudaErrorInvalidValue);
  }

  const auto jit_options = nullptr;
  const auto jit_options_values = nullptr;
  const auto num_jit_options = 0U;

  const auto library_options = nullptr;
  const auto library_option_values = nullptr;
  const auto num_library_options = 0U;

  auto lib = lib_t{nullptr};
  const auto err = cudaLibraryLoadFromFile(&lib,
                                           path,
                                           jit_options,
                                           jit_options_values,
                                           num_jit_options,
                                           library_options,
                                           library_option_values,
                                           num_library_options);
  if (err != cudaSuccess) {
    return Error(err);
  }
  return Ok{lib};
}

static auto lib_unload(lib_t lib) -> Result<> {
  if (lib == nullptr) {
    return Ok{};
  }

  if (auto err = cudaLibraryUnload(lib); err != cudaSuccess) {
    return Error(err);
  }

  return Ok{};
}

static auto lib_kernel(lib_t lib, const char* name) -> Result<kernel_t> {
  if (lib == nullptr || name == nullptr) {
    return Error(cudaErrorInvalidValue);
  }

  auto func = kernel_t{nullptr};
  if (auto err = cudaLibraryGetKernel(&func, lib, name); err != cudaSuccess) {
    return Error(err);
  }
  return Ok{func};
}

auto launch_kernel(kernel_t f, void* args[]) -> Result<> {
  if (f == nullptr) {
    return Error(cudaErrorInvalidValue);
  }

  const auto stream = cuda::stream_current();
  const auto grid_dim = cuda::grid_dim();
  const auto block_dim = cuda::block_dim();
  if (auto err = cudaLaunchKernel(f, grid_dim, block_dim, args, 0, stream); err != cudaSuccess) {
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
