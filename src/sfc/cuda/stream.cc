#include <cuda.h>

#include "sfc/core.h"
#include "sfc/cuda/stream.h"
#include "sfc/cuda/error.h"


namespace sfc::cuda {

auto stream_new(unsigned int flags) -> stream_t {
  auto stream = stream_t{nullptr};
  if (auto e = ::cuStreamCreate(&stream, flags)) {
    panic::panic_fmt("cuStreamCreate failed, err={}", Error{e});
  }
  return stream;
}

void stream_del(stream_t s) {
  if (s == nullptr) return;
  if (auto e = ::cuStreamDestroy_v2(s)) {
    panic::panic_fmt("cuStreamDestroy_v2 failed, err={}", Error{e});
  }
}

void stream_sync(stream_t s) {
  if (s == nullptr) return;
  if (auto e = ::cuStreamSynchronize(s)) {
    panic::panic_fmt("cuStreamSynchronize failed, err={}", Error{e});
  }
}

static thread_local stream_t _tls_stream = nullptr;

void stream_set(stream_t s) {
  _tls_stream = s;
}

auto stream_get() -> stream_t {
  return _tls_stream;
}

Stream::Stream() noexcept {}

Stream::~Stream() noexcept {
  cuda::stream_del(_raw);
}

Stream::Stream(Stream&& other) noexcept : _raw{other._raw} {
  other._raw = nullptr;
}

Stream& Stream::operator=(Stream&& other) noexcept {
  if (this == &other) {
    return *this;
  }

  cuda::stream_del(_raw);
  _raw = other._raw;
  other._raw = nullptr;
  return *this;
}

auto Stream::create(u32 flags) -> Stream {
  auto res = Stream{};
  res._raw = cuda::stream_new(flags);
  return res;
}

void Stream::sync() {
  cuda::stream_sync(_raw);
}

}  // namespace sfc::cuda
