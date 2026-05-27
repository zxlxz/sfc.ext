#pragma once

#include <cuda.h>
#include <cufft.h>
#include "sfc/core.h"

namespace sfc::cuda {

using panic::SourceLoc;

auto to_str(auto code) -> cstr_t;

template<>
inline auto to_str(CUresult code) -> cstr_t {
  auto err_name = cstr_t{nullptr};
  const auto ret = ::cuGetErrorName(code, &err_name);
  if (ret != 0) {
    return "CUDA_ERROR_UNKNOWN";
  }
  return err_name;
}

inline void check_ret(auto ret, const char* func, SourceLoc loc = SourceLoc::current()) {
  if (ret == 0) {
    return;
  }

  const auto err_name = cuda::to_str(ret);
  panic::panic_fmt(fmt::Args{"cuda API(`{}`) called failed, err=`{}`", func, err_name}, loc);
}

}  // namespace sfc::cuda

#if !defined(__INTELLISENSE__) && !defined(__clang_analyzer__)
#define CHECK_RET(func, ...) cuda::check_ret(func(__VA_ARGS__), #func)
#else
#define CHECK_RET(func, ...) func(__VA_ARGS__)
#endif
