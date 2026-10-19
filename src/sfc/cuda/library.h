#pragma once

#include "sfc/cuda/mod.h"

struct CUlib_st;
struct CUkern_st;
struct CUfunc_st;

namespace sfc::cuda {

auto launch(CUfunc_st* func, Slice<const void*> args) -> Result<>;

template <class>
struct Function;

template <class... T>
struct Function<void(T...)> {
  CUfunc_st* _func;

 public:
  auto operator()(const T&... args) -> Result<> {
    const void* argv[] = {&args..., nullptr};
    return cuda::launch(_func, argv);
  }
};

class Library {
  CUlib_st* _lib{nullptr};

 public:
  Library(CUlib_st& lib) noexcept;
  ~Library() noexcept;

  Library(Library&& other) noexcept;
  Library& operator=(Library&& other) noexcept;

  static auto load(Str path) -> Result<Library>;

 public:
  template<class F>
  auto get_func(const char* name) -> Result<Function<F>> {
    auto f = _TRY(this->get_func_imp(name));
    return Function<F>{f};
  }

  auto get_func_imp(const char* name) const -> Result<CUfunc_st*>;
};

}  // namespace sfc::cuda
