#pragma once

#include "sfc/cuda/exec.h"

struct CUfunc_st;
struct CUkern_st;

namespace sfc::cuda {

using lib_t = struct CUlib_st*;
using kernel_t = struct CUkern_st*;

void launch_kernel(kernel_t f, void* args[]);

template <class>
struct Kernel;

template <class... T>
struct Kernel<void(T...)> {
  kernel_t _kernel;

 public:
  void operator()(const T&... args) {
    void* argv[] = {&args...};
    cuda::launch_kernel(_kernel, &argv);
  }
};

class Library {
  lib_t _lib{nullptr};

 public:
  Library() noexcept;
  ~Library() noexcept;
  Library(Library&& other) noexcept;

  Library& operator=(Library&& other) noexcept = delete;

  static auto load(const char* path) -> Library;
  auto get_kernel(const char* name) const -> kernel_t;
};

}  // namespace sfc::cuda
