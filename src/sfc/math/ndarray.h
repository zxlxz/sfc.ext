#pragma once

#include "sfc/math/ndslice.h"
#include "sfc/math/alloc.h"

namespace sfc::math {

class RawBuf {
  using A = PoolAllocator;

  u8* _ptr{nullptr};
  usize _size{0};
  MemKind _kind{MemKind::CPU};
  [[no_unique_address]] A _alloc{};

 public:
  explicit RawBuf() noexcept;
  ~RawBuf();

  RawBuf(RawBuf&& other) noexcept;
  RawBuf& operator=(RawBuf&& other) noexcept;

  static auto xnew(usize size, MemKind memory) -> RawBuf;

 public:
  auto ptr() const -> u8* {
    return _ptr;
  }

  auto size() const -> usize {
    return _size;
  }

  auto kind() const -> MemKind {
    return _kind;
  }

 public:
  void bzero();
  void copy_from(const RawBuf& other);
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

  static auto xnew(const Shape& shape, MemKind memory = MemKind::CPU) -> NdArray {
    const auto numel = Inn{nullptr, shape, {}}.numel();
    auto buf = Buf::xnew(numel * sizeof(T), memory);
    return NdArray::from_buf(mem::move(buf), shape);
  }

  auto kind() const -> MemKind {
    return _buf.kind();
  }

  auto data() const -> T* {
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

  auto operator[](u32 idx) const -> NdSlice<T, NDIM - 1> {
    return _inn[idx];
  }

  auto operator[](const u32 (&idx)[NDIM]) const -> T {
    return _inn[idx];
  }

  auto operator[](const u32 (&idx)[NDIM]) -> T& {
    return _inn[idx];
  }

  auto get(const u32 (&idx)[NDIM]) const -> T {
    return _inn[idx];
  }

  void set(const u32 (&idx)[NDIM], T value) {
    _inn[idx] = value;
  }

  void imap(auto&& f) const {
    _inn.imap(f);
  }

  void imap_mut(auto&& f) {
    _inn.imap_mut(f);
  }

 public:
#ifndef __CUDACC__
  auto as_bytes() const -> slice::Slice<const u8> {
    return {_buf.ptr(), _buf.size()};
  }

  auto as_mut_bytes() -> slice::Slice<u8> {
    return {_buf.ptr(), _buf.size()};
  }
#endif

  void bzero() {
    _buf.bzero();
  }

  auto clone() const -> NdArray {
    auto res = NdArray::xnew(_inn._shape, _buf.kind());
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

  static auto xnew(const Shape& shape, MemKind memory = MemKind::CPU) -> NdArray {
    const auto size = Inn{nullptr, shape, {}}.numel();
    auto buf = Buf::xnew(size * sizeof(T), memory);
    return NdArray::from_buf(mem::move(buf), shape);
  }

  auto data() const -> T* {
    return _inn._data;
  }

  auto kind() const -> MemKind {
    return _buf.kind();
  }

  auto numel() const -> u32 {
    return _inn.numel();
  }

  auto shape() const -> const Shape& {
    return _inn._shape;
  }

  auto operator*() const -> Inn {
    return _inn;
  }

 public:
  auto operator[](u32 idx) const -> const T& {
    return _inn[idx];
  }

  auto operator[](u32 idx) -> T& {
    return _inn[idx];
  }

  auto operator[](const u32 (&idx)[NDIM]) const -> T {
    return _inn[idx];
  }

  auto operator[](const u32 (&idx)[NDIM]) -> T& {
    return _inn[idx];
  }

  auto get(const u32 (&idx)[NDIM]) const -> T {
    return _inn[idx];
  }

  void set(const u32 (&idx)[NDIM], T value) {
    _inn[idx] = value;
  }

  void imap(auto&& f) const {
    _inn.imap(f);
  }

  void imap_mut(auto&& f) {
    _inn.imap_mut(f);
  }

 public:
#ifndef __CUDACC__
  auto as_bytes() const -> slice::Slice<const u8> {
    return {_buf.ptr(), _buf.size()};
  }

  auto as_mut_bytes() -> slice::Slice<u8> {
    return {_buf.ptr(), _buf.size()};
  }
#endif

  void bzero() {
    _buf.bzero();
  }

  auto clone() const -> NdArray {
    auto res = NdArray::xnew(_inn.shape, _buf.kind());
    res._buf.copy_from(_buf);
    return res;
  }

  void fmt(auto& f) const {
    _inn.fmt(f);
  }
};

template <class T, u32 N>
auto array(const u32 (&shape)[N], MemKind memory = MemKind::CPU) -> NdArray<T, N> {
  return NdArray<T, N>::xnew(shape, memory);
}

}  // namespace sfc::math
