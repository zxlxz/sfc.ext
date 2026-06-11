
#include "sfc/cuda/mod.inl"
#include "sfc/cuda/stream.h"
#include "sfc/cuda/library.h"

namespace sfc::cuda {

auto lib_load(const char* path) -> lib_t {
  if (path == nullptr) {
    return nullptr;
  }

  const auto jit_options = nullptr;
  const auto jit_options_values = nullptr;
  const auto num_jit_options = 0U;

  const auto library_options = nullptr;
  const auto library_option_values = nullptr;
  const auto num_library_options = 0U;

  auto lib = lib_t{nullptr};
  CHECK_RET(cudaLibraryLoadFromFile,
            &lib,
            path,
            jit_options,
            jit_options_values,
            num_jit_options,
            library_options,
            library_option_values,
            num_library_options);
  return lib;
}

void lib_unload(lib_t lib) {
  if (lib == nullptr) {
    return;
  }
  CHECK_RET(cudaLibraryUnload, lib);
}

auto lib_get_kernel(lib_t lib, const char* name) -> kernel_t {
  if (lib == nullptr) {
    return nullptr;
  }

  if (name == nullptr) {
    return nullptr;
  }

  auto func = kernel_t{nullptr};
  CHECK_RET(cudaLibraryGetKernel, &func, lib, name);
  return func;
}

void launch_kernel(kernel_t f, void* args[]) {
  if (f == nullptr) {
    return;
  }

  const auto stream = cuda::stream_get();
  const auto grid_dim = cuda::grid_dim();
  const auto block_dim = cuda::block_dim();
  CHECK_RET(cudaLaunchKernel, f, grid_dim, block_dim, args, 0, stream);
}

Library::Library() noexcept = default;

Library::~Library() noexcept {
  if (_lib == nullptr) {
    return;
  }

  cuda::lib_unload(_lib);
  _lib = nullptr;
}

Library::Library(Library&& other) noexcept : _lib{other._lib} {
  other._lib = nullptr;
}

auto Library::load(const char* path) -> Library {
  auto res = Library{};
  res._lib = cuda::lib_load(path);
  return res;
}

auto Library::get_kernel(const char* name) const -> kernel_t {
  return cuda::lib_get_kernel(_lib, name);
}

}  // namespace sfc::cuda
