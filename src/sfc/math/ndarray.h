#pragma once

#include "sfc/math/rawbuf.h"
#include "sfc/math/ndslice.h"

namespace sfc::math {

template <int N>
auto numel(vec<u32, N> dims) -> u32 {
  if constexpr (N == 1) return dims.x;
  if constexpr (N == 2) return dims.x * dims.y;
  if constexpr (N == 3) return dims.x * dims.y * dims.z;
  if constexpr (N == 4) return dims.x * dims.y * dims.z * dims.w;
}

template <int N>
auto ndstep(vec<u32, N> dims) -> vec<u32, N> {
  if constexpr (N == 1) return {1};
  if constexpr (N == 2) return {1, dims.x};
  if constexpr (N == 3) return {1, dims.x, dims.x * dims.y};
  if constexpr (N == 4) return {1, dims.x, dims.x * dims.y, dims.x * dims.y * dims.z};
}

template <class T, int N = 1>
class [[nodiscard]] NdArray {
  static constexpr auto NDIM = N;
  using Buf = math::RawBuf<T>;
  using Inn = math::NdSlice<T, NDIM>;
  using dims_t = typename Inn::dims_t;
  using step_t = typename Inn::step_t;
  using idxs_t = typename Inn::idxs_t;

  Buf _buf = {};
  Inn _inn = {};

 public:
  NdArray() noexcept : _inn{}, _buf{} {}

  ~NdArray() {}

  NdArray(NdArray&& other) noexcept
      : _inn{static_cast<Inn&&>(other._inn)}, _buf{static_cast<Buf&&>(other._buf)} {}

  NdArray& operator=(NdArray&& other) noexcept {
    if (this == &other) return *this;
    _inn = static_cast<Inn&&>(other._inn);
    _buf = static_cast<Buf&&>(other._buf);
    return *this;
  }

  static auto with_shape(dims_t dims, cuda::MemType mtype = {}) -> NdArray {
    auto res = NdArray{};
    res._buf = Buf::with_capacity(math::numel(dims), mtype);
    res._inn = Inn{res._buf.ptr(), dims, math::ndstep(dims)};
    return res;
  }

  auto buf() const -> const Buf& {
    return _buf;
  }

  auto buf() -> Buf& {
    return _buf;
  }

  auto len() const -> u32 {
    return _inn.len();
  }

  auto data() const -> const T* {
    return _inn._data;
  }

  auto numel() const -> u32 {
    return _inn.numel();
  }

  auto shape() const -> dims_t {
    return _inn._dims;
  }

 public:
  operator Inn() const {
    return _inn;
  }

  auto operator->() const -> const Inn* {
    return &_inn;
  }

  auto operator->() -> Inn* {
    return &_inn;
  }

  auto operator[](u32 idx) const -> decltype(auto) {
    return _inn[idx];
  }

  auto operator[](u32 idx) -> decltype(auto) {
    return _inn[idx];
  }

  auto operator[](idxs_t idxs) const -> const T& {
    return _inn[idxs];
  }

  auto operator[](idxs_t idxs) -> T& {
    return _inn[idxs];
  }

 public:
  void imap(auto&& f) const {
    _inn.imap(f);
  }

  void imap_mut(auto&& f) {
    _inn.imap_mut(f);
  }

 public:
  void zero() {
    _buf.zero();
  }

  void copy_from(const NdArray& src) {
    _buf.copy_from(src._buf);
  }

  auto clone(MemType mtype = {}) const -> NdArray {
    auto res = NdArray::with_shape(_inn._dims, mtype);
    res.copy_from(*this);
    return res;
  }

  void fmt(auto& f) const {
    _inn.fmt(f);
  }
};

}  // namespace sfc::math
