#pragma once

#include "sfc/cuda/mod.h"

namespace sfc::cuda {

using lib_t = struct CUlib_st*;
using ker_t = struct CUkern_st*;
using fun_t = struct CUfunc_st*;

auto launch(fun_t func, Slice<void*> args) -> Result<>;

struct Kernel;

template <class>
struct Function;

class Library {
  lib_t _lib{nullptr};

 public:
  Library() noexcept;
  ~Library() noexcept;

  Library(Library&& other) noexcept;
  Library& operator=(Library&& other) noexcept;

  static auto load(Str path) -> Result<Library>;

 public:
  auto get_kern(const char* name) const -> Result<Kernel>;
};

struct Kernel {
  ker_t _ker;
  fun_t _func;

 public:
  template <class F>
  auto as_func() -> Function<F> {
    return Function<F>{_func};
  }
};

template <class... T>
struct Function<void(T...)> {
  fun_t _func;

 public:
  auto operator()(const T&... args) -> Result<> {
    void* argv[] = {&args...};
    return cuda::launch(_func, argv);
  }
};

}  // namespace sfc::cuda
