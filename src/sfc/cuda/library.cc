#include <cuda.h>

#include "sfc/ffi/cstr.h"
#include "sfc/cuda/stream.h"
#include "sfc/cuda/library.h"

namespace sfc::cuda {

namespace detail {

auto lib_load(Str path,
              Slice<CUjit_option> jit_opts,
              Slice<void*> jit_opt_vals,
              Slice<CUlibraryOption> lib_opts,
              Slice<void*> lib_opt_vals) -> Result<lib_t> {
  const auto c_path = ffi::CString::from(path);
  auto lib = lib_t{nullptr};
  if (auto err = cuLibraryLoadFromFile(&lib,
                                       c_path.as_ptr(),
                                       jit_opts._ptr,
                                       jit_opt_vals._ptr,
                                       u32(jit_opts.len()),
                                       lib_opts._ptr,
                                       lib_opt_vals._ptr,
                                       u32(lib_opts.len()))) {
    return Error(err);
  }
  return Ok{lib};
}

auto lib_unload(lib_t lib) -> Result<> {
  if (auto err = cuLibraryUnload(lib)) {
    return Error(err);
  }
  return Ok{};
}

auto lib_kern(lib_t lib, const char* name) -> Result<ker_t> {
  auto func = ker_t{nullptr};
  if (auto err = cuLibraryGetKernel(&func, lib, name)) {
    return Error(err);
  }
  return Ok{func};
}

auto ker_func(ker_t ker) -> Result<CUfunction> {
  auto func = CUfunction{nullptr};
  if (auto err = cuKernelGetFunction(&func, ker)) {
    return Error(err);
  }
  return Ok{func};
}

auto func_launch(CUfunction f, dim3_t grid_dim, dim3_t blk_dim, stream_t stream, void** args) -> Result<> {
  if (f == nullptr) {
    return Error(CUDA_ERROR_INVALID_VALUE);
  }

  const auto config = CUlaunchConfig{
      grid_dim.x,
      grid_dim.y,
      grid_dim.z,
      blk_dim.x,
      blk_dim.y,
      blk_dim.z,
      0,
      stream,
      nullptr,
      0,
  };

  if (auto err = cuLaunchKernelEx(&config, f, args, nullptr)) {
    return Error(err);
  }
  return Ok{};
}

}  // namespace detail

Library::Library() noexcept {}

Library::~Library() noexcept {
  if (_lib == nullptr) {
    return;
  }
  (void)detail::lib_unload(_lib);
  _lib = nullptr;
}

Library::Library(Library&& other) noexcept : _lib{mem::take(other._lib)} {}

auto Library::load(Str path) -> Result<Library> {
  auto to_lib = [](lib_t lib) {
    auto res = Library{};
    res._lib = lib;
    return res;
  };

  auto lib = detail::lib_load(path, {}, {}, {}, {}).map(to_lib);
  return lib;
}

auto Library::get_kern(const char* name) const -> Result<Kernel> {
  auto kern = _TRY(detail::lib_kern(_lib, name));
  auto func = _TRY(detail::ker_func(kern));
  return Kernel{kern, func};
}

auto launch(fun_t f, Slice<void*> args) -> Result<> {
  if (f == nullptr) {
    return Error(CUDA_ERROR_INVALID_VALUE);
  }

  const auto stream = cuda::stream_current();
  const auto grid_dim = cuda::grid_dim();
  const auto block_dim = cuda::block_dim();
  return detail::func_launch(f, grid_dim, block_dim, stream, args._ptr);
}

}  // namespace sfc::cuda
