#pragma once

#include <cuda_runtime_api.h>
#include "sfc/core.h"

namespace sfc::cuda {

using panic::SourceLoc;

auto error_name(cudaError_t code) -> cstr_t;

inline void check_ret(cudaError_t ret, const char* func, SourceLoc loc = SourceLoc::current()) {
  if (ret == 0) {
    return;
  }

  const auto err_name = cuda::error_name(ret);
  panic::panic_fmt(fmt::Args{"cuda API(`{}`) called failed, err=`{}`", func, err_name}, loc);
}

}  // namespace sfc::cuda

#if !defined(__INTELLISENSE__) && !defined(__clang_analyzer__)
#define CHECK_RET(func, ...) cuda::check_ret(func(__VA_ARGS__), #func)
#else
#define CHECK_RET(func, ...) func(__VA_ARGS__)
#endif
