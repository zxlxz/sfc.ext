#include <cuda.h>

#include "sfc/cuda/device.h"
#include "sfc/cuda/stream.h"

namespace sfc::cuda {

namespace detail {

static auto stream_new(unsigned int flags) -> Result<stream_t> {
  auto stream = stream_t{nullptr};
  if (auto err = cuStreamCreate(&stream, flags)) {
    return Error(err);
  }
  return Ok{stream};
}

static auto stream_del(stream_t s) -> Result<> {
  if (s == nullptr) {
    return Ok{};
  }

  if (auto err = cuStreamDestroy_v2(s)) {
    return Error(err);
  }
  return Ok{};
}

static auto stream_sync(stream_t s) -> Result<> {
  if (s == nullptr) {
    return Ok{};
  }

  if (auto err = cuStreamSynchronize(s)) {
    return Error(err);
  }
  return Ok{};
}

}  // namespace detail

Stream::Stream() noexcept {}

Stream::~Stream() noexcept {
  if (_raw == nullptr) {
    return;
  }
  detail::stream_del(_raw).unwrap();
  _raw = nullptr;
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

auto Stream::new_(u32 flags) -> Stream {
  auto res = Stream{};
  res._raw = detail::stream_new(flags).unwrap();
  return res;
}

auto Stream::sync() -> Result<> {
  return detail::stream_sync(_raw);
}

auto Stream::enter() -> Entered {
  return Entered{*this};
}

static thread_local stream_t _tls_stream = nullptr;

Stream::Entered::Entered(Stream& stream) : _stream_in{stream._raw}, _stream_out{_tls_stream} {
  _tls_stream = _stream_in;
}

Stream::Entered::~Entered() noexcept {
  _tls_stream = _stream_out;
}

auto stream_current() -> stream_t {
  return _tls_stream;
}

auto stream_sync() -> Result<> {
  return detail::stream_sync(_tls_stream);
}

}  // namespace sfc::cuda
