#pragma once

#include "sfc/math/rawbuf.h"
#include "sfc/math/ndview.h"

namespace sfc::math {

template <class T, int N = 1>
class [[nodiscard]] NdArray {
  static constexpr auto NDIM = N;
  using Buf = math::RawBuf<T>;
  using Inn = math::NdView<T, NDIM>;
  using Shape = typename Inn::Shape;
  using Strides = typename Inn::Strides;
  using Index = typename Inn::Index;

  Buf _buf = {};
  Inn _inn = {};

 public:
  NdArray() noexcept : _buf{}, _inn{nullptr, {}, {}} {}
  ~NdArray() {}

  NdArray(NdArray&& other) noexcept = default;
  NdArray& operator=(NdArray&& other) noexcept = default;

  static auto with_shape(const Shape& shape, MemType mtype = {}) -> NdArray {
    auto inn = Inn{nullptr, {shape}, {}};
    for (auto i = NDIM - 1; i > 0; --i) {
      inn._strides[i] = i == NDIM - 1 ? 1 : inn._shape[i + 1] * inn._strides[i + 1];
    }

    auto res = NdArray{};
    res._buf = Buf::with_capacity(inn.numel(), {mtype});
    res._inn = inn;
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

  auto shape() const -> const Shape& {
    return _inn._shape;
  }

 public:
  operator Inn() const {
    return _inn;
  }

  auto operator*() const -> const Inn& {
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

  auto operator[](Index indices) const -> const T& {
    return _inn[indices];
  }

  auto operator[](Index indices) -> T& {
    return _inn[indices];
  }

 public:
  void imap(auto&& f) const {
    _inn.imap(f);
  }

  void imap_mut(auto&& f) {
    _inn.imap_mut(f);
  }

 public:
  void bzero() {
    _buf.bzero();
  }

  auto clone(MemType mtype) const -> NdArray {
    auto res = NdArray::with_shape(this->shape(), mtype);
    res._buf.copy_from(_buf);
    return res;
  }

  void fmt(auto& f) const {
    _inn.fmt(f);
  }
};

}  // namespace sfc::math
