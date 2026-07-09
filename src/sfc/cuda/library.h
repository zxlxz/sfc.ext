#pragma once

#include "sfc/cuda/mod.h"

namespace sfc::cuda {

using lib_t = struct CUlib_st*;
using kernel_t = struct CUkern_st*;

auto launch_kernel(kernel_t f, void* args[]) -> Result<>;

template <class>
struct Kernel;

template <class... T>
struct Kernel<void(T...)> {
  kernel_t _kernel;

 public:
  auto operator()(const T&... args) -> Result<> {
    void* argv[] = {&args...};
    return cuda::launch_kernel(_kernel, &argv);
  }
};

class Library {
  lib_t _lib{nullptr};

 public:
  Library() noexcept;
  ~Library() noexcept;

  Library(Library&& other) noexcept;
  Library& operator=(Library&& other) noexcept;

  static auto load(const char* path) -> Library;

 public:
  auto get_kernel(const char* name) const -> Result<kernel_t>;
};

}  // namespace sfc::cuda
