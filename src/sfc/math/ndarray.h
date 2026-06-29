#pragma once

#include "sfc/math/ndslice.h"
#include "sfc/math/alloc.h"

namespace sfc::math {

class RawBuf {
  using A = PoolAllocator;

  void* _ptr{nullptr};
  usize _size{0};
  MemType _mtype{MemType::CPU};
  [[no_unique_address]] A _alloc{};

 public:
  explicit RawBuf() noexcept;
  ~RawBuf();

  RawBuf(RawBuf&& other) noexcept;
  RawBuf& operator=(RawBuf&& other) noexcept;

  static auto xnew(usize size, MemType mtype) -> RawBuf;

 public:
  auto ptr() const -> void* {
    return _ptr;
  }

  auto size() const -> usize {
    return _size;
  }

  auto mtype() const -> MemType {
    return _mtype;
  }

  void bzero();
};

template <class T, u32 N>
class [[nodiscard]] NdArray {
  static constexpr u32 NDIM = N;
  using Buf = RawBuf;
  using Inn = NdSlice<T, NDIM>;
  using Shape = u32[NDIM];

  Buf _buf{};
  Inn _inn{};

 public:
  NdArray() noexcept : _buf{}, _inn{nullptr, {}, {}} {}
  ~NdArray() {}

  NdArray(NdArray&& other) noexcept = default;
  NdArray& operator=(NdArray&& other) noexcept = default;

  static auto from_buf(Buf buf, const Shape& shape) -> NdArray {
    u32 strides[NDIM] = {};
    for (auto i = NDIM; i != 0; --i) {
      strides[i - 1] = i == NDIM ? 1 : shape[i] * strides[i];
    }

    const auto ptr = ptr::cast<T>(buf.ptr());
    auto res = NdArray{};
    res._buf = mem::move(buf);
    res._inn = Inn{ptr, shape, strides};
    return res;
  }

  static auto xnew(const Shape& shape, MemType mtype) -> NdArray {
    const auto numel = Inn{nullptr, shape, {}}.numel();
    auto buf = Buf::xnew(numel * sizeof(T), mtype);
    return NdArray::from_buf(mem::move(buf), shape);
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

  auto shape() const -> const Shape& {
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
    auto res = NdArray::xnew(_inn.shape, mtype);
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
  using Buf = RawBuf;
  using Inn = NdSlice<T, NDIM>;
  using Shape = u32[NDIM];
  Buf _buf{};
  Inn _inn{};

 public:
  NdArray() noexcept : _buf{}, _inn{nullptr, {}, {}} {}
  ~NdArray() {}

  NdArray(NdArray&& other) noexcept = default;
  NdArray& operator=(NdArray&& other) noexcept = default;

  static auto from_buf(Buf buf, const Shape& shape) -> NdArray {
    const auto ptr = ptr::cast<T>(buf.ptr());

    auto res = NdArray{};
    res._buf = mem::move(buf);
    res._inn = Inn{ptr, shape, {1}};
    return res;
  }

  static auto xnew(const Shape& shape, MemType mtype) -> NdArray {
    const auto size = Inn{nullptr, shape, {}}.numel();
    auto buf = Buf::xnew(size * sizeof(T), mtype);
    return NdArray::from_buf(mem::move(buf), shape);
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
    auto res = NdArray::xnew(_inn.shape, mtype);
    res._buf.copy_from(_buf);
    return res;
  }

  void fmt(auto& f) const {
    _inn.fmt(f);
  }
};

template <class T, u32 N>
auto array(const u32 (&shape)[N], MemType mtype = MemType::CPU) -> NdArray<T, N> {
  return NdArray<T, N>::xnew(shape, mtype);
}

}  // namespace sfc::math
