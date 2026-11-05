#pragma once

#include "sfc/math/ndslice.h"
#include "sfc/math/rawbuf.h"

namespace sfc::math {

template <class T, u32 N>
class [[nodiscard]] NdArray;

template <class T>
class NdArray<T, 1> {
  using A = Allocator;
  using Buf = RawBuf<T>;
  using Inn = NdSlice<T, 1>;
  Buf _buf{};
  Inn _inn{};

 public:
  static auto from_buf(Buf buf, const u32 (&shape)[1]) -> NdArray {
    auto res = NdArray{};
    res._buf = mem::move(buf);
    res._inn = Inn::with_shape(res._buf.ptr(), shape);
    return res;
  }

  static auto new_(const u32 (&shape)[1], A& alloc = A::global()) -> NdArray {
    auto buf = Buf::with_capacity(shape[0], alloc);
    return NdArray::from_buf(mem::move(buf), shape);
  }

  static auto new_zeroed(const u32 (&shape)[1], A& alloc = A::global()) -> NdArray {
    auto buf = Buf::with_capacity_zeroed(shape[0], alloc);
    return NdArray::from_buf(mem::move(buf), shape);
  }

 public:
  static constexpr auto ndim() -> u32 {
    return 1;
  }

  auto allocator() -> A& {
    return _buf.allocator();
  }

  auto as_slice() const -> Slice<const T> {
    return _buf.as_slice();
  }

  auto as_mut_slice() -> Slice<T> {
    return _buf.as_mut_slice();
  }

  auto as_bytes() const -> Slice<const u8> {
    return _buf.as_bytes();
  }

  auto as_mut_bytes() -> Slice<u8> {
    return _buf.as_mut_bytes();
  }

  auto shape(u32 ndim) const -> u32 {
    return ndim < 1 ? _inn._shape[ndim] : 0;
  }

 public:
  auto operator*() const -> const Inn& {
    return _inn;
  }

  auto operator*() -> Inn& {
    return _inn;
  }

  auto operator->() const -> const Inn* {
    return &_inn;
  }

  auto operator->() -> Inn* {
    return &_inn;
  }

 public:
  auto operator[](u32 idx) const -> const T& {
    return _inn[idx];
  }

  auto operator[](u32 idx) -> T& {
    return _inn[idx];
  }

  auto get(const u32 (&idx)[1]) const -> T {
    return _inn.get(idx);
  }

  void set(const u32 (&idx)[1], T val) {
    _inn.set(idx, val);
  }

  void for_each(auto&& f) const {
    _inn.for_each(f);
  }

  void for_each_mut(auto&& f) {
    _inn.for_each_mut(f);
  }

  void fill_with(auto&& f) {
    _inn.fill_with(f);
  }
};

template <class T, u32 N>
class NdArray {
  using A = Allocator;
  using Buf = RawBuf<T>;
  using Inn = NdSlice<T, N>;
  Buf _buf{};
  Inn _inn{};

 public:
  static auto from_buf(Buf buf, const u32 (&shape)[N]) -> NdArray {
    auto res = NdArray{};
    res._buf = mem::move(buf);
    res._inn = Inn::with_shape(res._buf.ptr(), shape);
    return res;
  }

  static auto new_(const u32 (&shape)[N], A& alloc = A::global()) -> NdArray {
    const auto numel = Inn::with_shape(nullptr, shape).numel();
    auto buf = Buf::with_capacity(numel, alloc);
    return NdArray::from_buf(mem::move(buf), shape);
  }

  static auto new_zeroed(const u32 (&shape)[N], A& alloc = A::global()) -> NdArray {
    const auto numel = Inn::with_shape(nullptr, shape).numel();
    auto buf = Buf::with_capacity_zeroed(numel, alloc);
    return NdArray::from_buf(mem::move(buf), shape);
  }

 public:
  static constexpr auto ndim() -> u32 {
    return N;
  }

  auto allocator() -> A& {
    return _buf.allocator();
  }

  auto as_slice() const -> Slice<const T> {
    return _buf.as_slice();
  }

  auto as_mut_slice() -> Slice<T> {
    return _buf.as_mut_slice();
  }

  auto as_bytes() const -> Slice<const u8> {
    return _buf.as_bytes();
  }

  auto as_mut_bytes() -> Slice<u8> {
    return _buf.as_mut_bytes();
  }

  auto shape(u32 ndim) const -> u32 {
    return ndim < N ? _inn._shape[ndim] : 0;
  }

 public:
  auto operator*() const -> const Inn& {
    return _inn;
  }

  auto operator*() -> Inn& {
    return _inn;
  }

  auto operator->() const -> const Inn* {
    return &_inn;
  }

  auto operator->() -> Inn* {
    return &_inn;
  }

 public:
  auto operator[](u32 idx) const -> NdSlice<T, N - 1> {
    return _inn[idx];
  }

  auto get(const u32 (&idx)[N]) const -> T {
    return _inn.get(idx);
  }

  void set(const u32 (&idx)[N], T val) {
    _inn.set(idx, val);
  }

  void for_each(auto&& f) const {
    _inn.for_each(f);
  }

  void for_each_mut(auto&& f) {
    _inn.for_each_mut(f);
  }

  void fill_with(auto&& f) {
    _inn.fill_with(f);
  }
};

template <class T, u32 N>
auto empty(const u32 (&shape)[N]) -> NdArray<T, N> {
  return NdArray<T, N>::new_(shape);
}

template <class T, u32 N>
auto empty_like(const NdArray<T, N>& array) -> NdArray<T, N> {
  const auto& shape = array->_shape;
  return NdArray<T, N>::new_(shape);
}

template <class T, u32 N>
auto zero(const u32 (&shape)[N]) -> NdArray<T, N> {
  return NdArray<T, N>::new_zeroed(shape);
}

template <class T, u32 N>
auto zero_like(const NdArray<T, N>& array) -> NdArray<T, N> {
  const auto& shape = array->_shape;
  return NdArray<T, N>::new_zeroed(shape);
}

}  // namespace sfc::math
