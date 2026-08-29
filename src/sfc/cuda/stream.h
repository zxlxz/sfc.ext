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

  static auto new_(u32 flags = 0) -> Stream;

 public:
  auto sync() -> Result<>;

  class Entered;
  auto enter() -> Entered;
};

class Stream::Entered {
  stream_t _stream_in;
  stream_t _stream_out;

 public:
  explicit Entered(Stream& stream);
  ~Entered() noexcept;

  Entered(const Entered&) = delete;
  Entered& operator=(const Entered&) = delete;
};

auto stream_current() -> stream_t;
auto stream_sync() -> Result<>;

}  // namespace sfc::cuda
