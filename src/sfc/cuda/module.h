#pragma once

#include "sfc/cuda/exec.h"

struct CUfunc_st;
struct CUmod_st;

namespace sfc::cuda {

using mod_t = struct CUmod_st*;
using func_t = struct CUfunc_st*;

auto mod_load(const char* path) -> mod_t;
void mod_unload(mod_t mod);
auto mod_func(mod_t mod, const char* name) -> func_t;
void func_launch(func_t f, void* args[]);

template <class>
struct KFunc;

template <class... T>
struct KFunc<void(T...)> {
  func_t _func;

 public:
  void operator()(const T&... args) {
    void* argv[] = {&args...};
    cuda::func_launch(_func, &argv);
  }
};

class Module {
  mod_t _raw{nullptr};

 public:
  Module() noexcept;
  ~Module() noexcept;
  Module(Module&& other) noexcept;

  Module& operator=(Module&& other) noexcept = delete;

  static auto load(const char* path) -> Module;
  auto get(const char* name) const -> func_t;
};

}  // namespace sfc::cuda
