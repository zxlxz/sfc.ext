#include <cuda.h>

#include "sfc/cuda/module.h"
#include "sfc/cuda/stream.h"
#include "sfc/cuda/error.h"

#define CU_TRY(expr)       \
  if (auto err = (expr)) { \
    throw Error{err};      \
  }

namespace sfc::cuda {

auto mod_load(const char* path) -> mod_t {
  auto mod = mod_t{nullptr};
  CU_TRY(::cuModuleLoad(&mod, path));
  return mod;
}

void mod_unload(mod_t mod) {
  if (mod == nullptr) return;
  CU_TRY(::cuModuleUnload(mod));
}

auto mod_func(mod_t mod, const char* name) -> func_t {
  if (name == nullptr) return nullptr;
  if (mod == nullptr) {
    throw Error{CUDA_ERROR_INVALID_VALUE};
  }
  auto func = func_t{nullptr};
  CU_TRY(::cuModuleGetFunction(&func, mod, name));
  return func;
}

void func_launch(func_t f, void* args[]) {
  if (f == nullptr) {
    throw Error{CUDA_ERROR_INVALID_VALUE};
  }

  const auto stream = cuda::stream_get();
  const auto blks = cuda::get_blks();
  const auto trds = cuda::get_trds();

  const auto conf = CUlaunchConfig{
      .gridDimX = blks.x,
      .gridDimY = blks.y,
      .gridDimZ = blks.z,
      .blockDimX = trds.x,
      .blockDimY = trds.y,
      .blockDimZ = trds.z,
      .hStream = stream,
  };

  CU_TRY(::cuLaunchKernelEx(&conf, f, args, nullptr));
}

Module::Module() noexcept {}

Module::~Module() noexcept {
  cuda::mod_unload(_raw);
}

Module::Module(Module&& other) noexcept : _raw{other._raw} {
  other._raw = nullptr;
}

auto Module::load(const char* path) -> Module {
  auto res = Module{};
  res._raw = cuda::mod_load(path);
  return res;
}

auto Module::get(const char* name) const -> CUfunc_st* {
  return cuda::mod_func(_raw, name);
}

}  // namespace sfc::cuda
