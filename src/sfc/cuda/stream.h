#pragma once

#include "sfc/cuda/device.h"

struct CUstream_st;

namespace sfc::cuda {

using stream_t = ::CUstream_st*;

class Stream {
  stream_t _raw{nullptr};

 public:
  Stream() noexcept;
  ~Stream() noexcept;

  Stream(Stream&& other) noexcept;
  Stream& operator=(Stream&& other) noexcept;

  static auto create(u32 flags = 0) -> Stream;

 public:
  auto sync() -> Result<>;

  class ScopeGuard;
  auto scope() -> ScopeGuard;
};

class Stream::ScopeGuard {
  stream_t _enter;
  stream_t _exit;

 public:
  ScopeGuard(stream_t enter, stream_t exit);
  ~ScopeGuard();

  ScopeGuard(const ScopeGuard&) = delete;
  ScopeGuard& operator=(const ScopeGuard&) = delete;
};

auto stream_current() -> stream_t;
auto stream_sync() -> Result<>;

}  // namespace sfc::cuda
