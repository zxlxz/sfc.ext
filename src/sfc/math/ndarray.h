#pragma once

#include "sfc/math/rawbuf.h"
#include "sfc/math/ndview.h"

namespace sfc::math {

template <class T, u32 N>
class [[nodiscard]] NdArray {
  static constexpr u32 NDIM = N;
  using Buf = RawBuf<T>;
  using Inn = NdSlice<T, NDIM>;

  Buf _buf = {};
  Inn _inn = {};

 public:
  NdArray() noexcept : _buf{}, _inn{nullptr, {}, {}} {}
  ~NdArray() {}

  NdArray(NdArray&& other) noexcept = default;
  NdArray& operator=(NdArray&& other) noexcept = default;

  static auto with_shape(const u32 (&shape)[NDIM], MemType mtype = {}) -> NdArray {
    u32 strides[NDIM] = {};
    for (auto i = NDIM; i != 0; --i) {
      strides[i - 1] = i == NDIM ? 1 : shape[i] * strides[i];
    }
    auto inn = Inn{nullptr, shape, strides};
    auto buf = Buf::with_capacity(inn.numel(), {mtype});
    inn._data = buf.ptr();

    auto res = NdArray{};
    res._buf = mem::move(buf);
    res._inn = inn;
    return res;
  }

  auto buf() -> Buf& {
    return _buf;
  }

  auto len() const -> u32 {
    return _inn.len();
  }

  auto data() -> T* {
    return _inn._data;
  }

  auto numel() const -> u32 {
    return _inn.numel();
  }

  auto shape() const -> const u32 (&)[NDIM] {
    return _inn._shape;
  }

 public:
  auto operator*() const -> Inn {
    return _inn;
  }

  auto operator[](u32 idx) -> NdSlice<T, NDIM - 1> {
    return _inn[idx];
  }

  auto get(const u32 (&indices)[NDIM]) const -> const T& {
    return _inn.get(indices);
  }

  void set(const u32 (&indices)[NDIM], const T& value) {
    _inn.set(indices, value);
  }

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

template <class T>
class [[nodiscard]] NdArray<T, 1> {
  static constexpr u32 NDIM = 1;
  using Buf = RawBuf<T>;
  using Inn = NdSlice<T, NDIM>;
  Buf _buf = {};
  Inn _inn = {};

 public:
  NdArray() noexcept : _buf{}, _inn{nullptr, {}, {}} {}
  ~NdArray() {}

  NdArray(NdArray&& other) noexcept = default;
  NdArray& operator=(NdArray&& other) noexcept = default;

  static auto with_shape(const u32 (&shape)[NDIM], MemType mtype = MemType::CPU) -> NdArray {
    auto inn = Inn{nullptr, shape, {1}};
    auto buf = Buf::with_capacity(inn.numel(), {mtype});
    inn._data = buf.ptr();

    auto res = NdArray{};
    res._buf = mem::move(buf);
    res._inn = inn;
    return res;
  }

  auto buf() -> Buf& {
    return _buf;
  }

  auto data() -> T* {
    return _inn._data;
  }

  auto len() const -> u32 {
    return _inn.len();
  }

  auto numel() const -> u32 {
    return _inn.numel();
  }

  auto shape() const -> const u32 (&)[NDIM] {
    return _inn._shape;
  }

 public:
  auto operator*() const -> Inn {
    return _inn;
  }

  auto operator[](u32 idx) const -> const T& {
    return _inn[idx];
  }

  auto operator[](u32 idx) -> T& {
    return _inn[idx];
  }

  auto get(const u32 (&indices)[1]) const -> const T& {
    return _inn.get(indices);
  }

  void set(const u32 (&indices)[1], const T& value) {
    _inn.set(indices, value);
  }

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
