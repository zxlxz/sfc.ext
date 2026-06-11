#include "sfc/cuda/mod.inl"
#include "sfc/cuda/stream.h"

namespace sfc::cuda {

auto stream_new(unsigned int flags) -> stream_t {
  auto stream = stream_t{nullptr};
  CHECK_RET(cudaStreamCreateWithFlags, &stream, flags);
  return stream;
}

void stream_del(stream_t s) {
  if (s == nullptr) return;
  CHECK_RET(cudaStreamDestroy, s);
}

void stream_sync(stream_t s) {
  if (s == nullptr) return;
  CHECK_RET(cudaStreamSynchronize, s);
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
