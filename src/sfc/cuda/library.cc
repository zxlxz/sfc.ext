#include <cuda.h>

#include "sfc/ffi/cstr.h"
#include "sfc/cuda/stream.h"
#include "sfc/cuda/library.h"

namespace sfc::cuda {

namespace detail {

struct JitOptions {
  static constexpr u32 MAX_OPT_CNT{32};

  u32 cnt = {0};
  CUjit_option keys[MAX_OPT_CNT] = {};
  void* vals[MAX_OPT_CNT] = {};
};

struct LibOptions {
  static constexpr u32 MAX_OPT_CNT{32};

  u32 cnt = {0};
  CUlibraryOption keys[MAX_OPT_CNT] = {};
  void* vals[MAX_OPT_CNT] = {};
};

auto load_lib(const char* path, JitOptions jit_opts, LibOptions lib_opts) -> Result<CUlibrary> {
  auto lib = CUlibrary{nullptr};
  const auto ret = cuLibraryLoadFromFile(
      &lib, path, jit_opts.keys, jit_opts.vals, jit_opts.cnt, lib_opts.keys, lib_opts.vals, lib_opts.cnt);
  if (ret) {
    return Error(ret);
  }
  return Ok{lib};
}

auto unload_lib(CUlibrary lib) -> Result<> {
  if (auto err = cuLibraryUnload(lib)) {
    return Error(err);
  }
  return Ok{};
}

auto get_kernel(CUlibrary lib, const char* name) -> Result<CUkernel> {
  auto kernel = CUkernel{nullptr};
  if (auto err = cuLibraryGetKernel(&kernel, lib, name)) {
    return Error(err);
  }
  return Ok{kernel};
}

auto get_function(CUkernel ker) -> Result<CUfunction> {
  auto func = CUfunction{nullptr};
  if (auto err = cuKernelGetFunction(&func, ker)) {
    return Error(err);
  }
  return Ok{func};
}

auto launch_kernel(CUfunction f, dim3_t grid_dim, dim3_t blk_dim, stream_t stream, const void* args[]) -> Result<> {
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

  if (auto err = cuLaunchKernelEx(&config, f, (void**)args, nullptr)) {
    return Error(err);
  }
  return Ok{};
}

}  // namespace detail

Library::Library(CUlib_st& lib) noexcept : _lib{&lib} {}

Library::~Library() noexcept {
  if (_lib == nullptr) {
    return;
  }
  (void)detail::unload_lib(_lib);
}

Library::Library(Library&& other) noexcept : _lib{mem::take(other._lib)} {}

auto Library::load(Str path) -> Result<Library> {
  const auto c_path = ffi::CString::from_str(path);
  auto jit_opts = detail::JitOptions{};
  auto lib_opts = detail::LibOptions{};
  auto lib = _TRY(detail::load_lib(c_path.as_ptr(), jit_opts, lib_opts));

  return Library{*lib};
}

auto Library::get_func_imp(const char* name) const -> Result<CUfunc_st*> {
  auto kern = _TRY(detail::get_kernel(_lib, name));
  auto func = _TRY(detail::get_function(kern));
  return Ok{func};
}

auto launch(CUfunc_st* f, Slice<const void*> args) -> Result<> {
  if (f == nullptr) {
    return Error(CUDA_ERROR_INVALID_VALUE);
  }

  const auto stream = cuda::stream_current();
  const auto grid_dim = cuda::grid_dim();
  const auto block_dim = cuda::block_dim();
  return detail::launch_kernel(f, grid_dim, block_dim, stream, args._ptr);
}

}  // namespace sfc::cuda
