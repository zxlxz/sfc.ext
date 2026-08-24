#include <cuda.h>

#include "sfc/ffi/cstr.h"
#include "sfc/cuda/stream.h"
#include "sfc/cuda/library.h"

namespace sfc::cuda {

namespace detail {

struct JITOption {
  CUjit_option opt;
  void* val;
};

struct LibOption {
  CUlibraryOption opt;
  void* val;
};

auto lib_load(const char* path, Slice<JITOption> jit_opt_list, Slice<LibOption> lib_opt_list) -> Result<lib_t> {
  const auto kMaxJItOptCnt = 32U;
  const auto kMaxLibOptCnt = 8U;
  if (jit_opt_list.len() > kMaxJItOptCnt) {
    return Error(CUDA_ERROR_INVALID_VALUE);
  }
  if (lib_opt_list.len() > kMaxLibOptCnt) {
    return Error(CUDA_ERROR_INVALID_VALUE);
  }

  auto jit_opt_cnt = u32(jit_opt_list.len());
  CUjit_option jit_opts[kMaxJItOptCnt]{};
  void* jit_opt_vals[kMaxJItOptCnt]{};
  for (auto i = 0U; i < jit_opt_cnt; ++i) {
    jit_opts[i] = jit_opt_list[i].opt;
    jit_opt_vals[i] = jit_opt_list[i].val;
  }

  auto lib_opt_cnt = u32(lib_opt_list.len());
  CUlibraryOption lib_opts[kMaxLibOptCnt]{};
  void* lib_opt_vals[kMaxLibOptCnt]{};
  for (auto i = 0U; i < lib_opt_cnt; ++i) {
    lib_opts[i] = lib_opt_list[i].opt;
    lib_opt_vals[i] = lib_opt_list[i].val;
  }

  auto lib = lib_t{nullptr};
  if (auto err =
          cuLibraryLoadFromFile(&lib, path, jit_opts, jit_opt_vals, jit_opt_cnt, lib_opts, lib_opt_vals, lib_opt_cnt)) {
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
  const auto c_path = ffi::CString::from(path);

  auto to_lib = [](lib_t lib) {
    auto res = Library{};
    res._lib = lib;
    return res;
  };

  auto lib = detail::lib_load(c_path.as_ptr(), {}, {}).map(to_lib);
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
