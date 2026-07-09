#include <cuda_runtime_api.h>

#include "sfc/core.h"
#include "sfc/cuda/device.h"
#include "sfc/cuda/stream.h"

namespace sfc::cuda {

static auto stream_new(unsigned int flags) -> Result<stream_t> {
  auto stream = stream_t{nullptr};
  if (auto err = ::cudaStreamCreateWithFlags(&stream, flags)) {
    return Err{Error(err)};
  }
  return Ok{stream};
}

static auto stream_del(stream_t s) -> Result<> {
  if (s == nullptr) {
    return Ok{};
  }

  if (auto err = ::cudaStreamDestroy(s)) {
    return Err{Error(err)};
  }
  return Ok{};
}

static auto stream_sync(stream_t s) -> Result<> {
  if (s == nullptr) {
    return Ok{};
  }

  if (auto err = ::cudaStreamSynchronize(s)) {
    return Err{Error(err)};
  }
  return Ok{};
}

static thread_local stream_t _tls_stream = nullptr;

void stream_set(stream_t s) {
  _tls_stream = s;
}

auto stream_current() -> stream_t {
  return _tls_stream;
}

auto stream_sync() -> Result<> {
  if (auto err = ::cudaStreamSynchronize(_tls_stream)) {
    return Err{Error(err)};
  }
  return Ok{};
}

Stream::Stream() noexcept {}

Stream::~Stream() noexcept {
  if (_raw == nullptr) {
    return;
  }
  (void)cuda::stream_del(_raw);
}

Stream::Stream(Stream&& other) noexcept : _raw{other._raw} {
  other._raw = nullptr;
}

Stream& Stream::operator=(Stream&& other) noexcept {
  if (this != &other) {
    mem::swap(_raw, other._raw);
  }
  return *this;
}

auto Stream::create(u32 flags) -> Stream {
  auto res = Stream{};
  res._raw = cuda::stream_new(flags).unwrap();
  return res;
}

auto Stream::sync() -> Result<> {
  return cuda::stream_sync(_raw);
}

auto Stream::scope() -> ScopeGuard {
  const auto curr = cuda::stream_current();
  const auto next = _raw;
  return ScopeGuard{curr, next};
}

Stream::ScopeGuard::ScopeGuard(stream_t enter, stream_t exit) : _enter{enter}, _exit{exit} {
  cuda::stream_set(_enter);
}

Stream::ScopeGuard::~ScopeGuard() {
  cuda::stream_set(_exit);
}

}  // namespace sfc::cuda
