#pragma once

#include "sfc/math/ndslice.h"
#include "sfc/math/rawbuf.h"

namespace sfc::math {

template <class T, u32 N>
class [[nodiscard]] NdArray;

template <class T>
class NdArray<T, 1> {
  using A = Allocator;
  using Buff = RawBuf<T>;
  using View = NdSlice<T, 1>;
  Buff _buff{};
  View _view{};

 public:
  static auto from_buf(Buff buf, const u32 (&shape)[1]) -> NdArray {
    auto res = NdArray{};
    res._buff = mem::move(buf);
    res._view = View::with_shape(res._buff.ptr(), shape);
    return res;
  }

  static auto new_(const u32 (&shape)[1], A& alloc = A::global()) -> NdArray {
    auto buf = Buff::with_capacity(shape[0], alloc);
    return NdArray::from_buf(mem::move(buf), shape);
  }

  static auto new_zeroed(const u32 (&shape)[1], A& alloc = A::global()) -> NdArray {
    auto buf = Buff::with_capacity_zeroed(shape[0], alloc);
    return NdArray::from_buf(mem::move(buf), shape);
  }

 public:
  auto numel() const -> u32 {
    return _view.numel();
  }

  auto shape() const -> const auto& {
    return _view._shape;
  }

  auto strides() const -> const auto& {
    return _view._strides;
  }

  auto buf() -> Buff& {
    return _buff;
  }

  auto view() -> View {
    return _view;
  }

  auto allocator() -> A& {
    return _buff.allocator();
  }

 public:
  auto operator*() const -> View {
    return _view;
  }

  auto operator*() -> View {
    return _view;
  }

  auto operator->() const -> const View* {
    return &_view;
  }

  auto operator->() -> View* {
    return &_view;
  }

 public:
  auto operator[](u32 idx) const {
    return _view[idx];
  }

  auto operator()(u32 i) const -> const T& {
    return _view(i);
  }

  auto operator()(u32 i) -> T& {
    return _view(i);
  }
};

template <class T, u32 N>
class NdArray {
  using A = Allocator;
  using Buff = RawBuf<T>;
  using View = NdSlice<T, N>;
  Buff _buff{};
  View _view{};

 public:
  static auto from_buf(Buff buf, const u32 (&shape)[N]) -> NdArray {
    auto res = NdArray{};
    res._buff = mem::move(buf);
    res._view = View::with_shape(res._buff.ptr(), shape);
    return res;
  }

  static auto new_(const u32 (&shape)[N], A& alloc = A::global()) -> NdArray {
    const auto numel = View::with_shape(nullptr, shape).numel();
    auto buf = Buff::with_capacity(numel, alloc);
    return NdArray::from_buf(mem::move(buf), shape);
  }

  static auto new_zeroed(const u32 (&shape)[N], A& alloc = A::global()) -> NdArray {
    const auto numel = View::with_shape(nullptr, shape).numel();
    auto buf = Buff::with_capacity_zeroed(numel, alloc);
    return NdArray::from_buf(mem::move(buf), shape);
  }

 public:
  auto numel() const -> u32 {
    return _view.numel();
  }

  auto shape() const -> const auto& {
    return _view._shape;
  }

  auto strides() const -> const auto& {
    return _view._strides;
  }

  auto buf() -> Buff& {
    return _buff;
  }

  auto view() -> View {
    return _view;
  }

  auto allocator() -> A& {
    return _buff.allocator();
  }

 public:
  auto operator*() const -> const View& {
    return _view;
  }

  auto operator*() -> View& {
    return _view;
  }

  auto operator->() const -> const View* {
    return &_view;
  }

  auto operator->() -> View* {
    return &_view;
  }

 public:
  auto operator[](u32 idx) const {
    return _view[idx];
  }

  auto operator()(u32 i) const -> const T& requires(N == 1) {
    return _view(i);
  }

  auto operator()(u32 i) -> T& requires(N == 1) {
    return _view(i);
  }

  auto operator()(u32 i, u32 j) const -> const T& requires(N == 2) {
    return _view(i, j);
  }

  auto operator()(u32 i, u32 j) -> T& requires(N == 2) {
    return _view(i, j);
  }

  auto operator()(u32 i, u32 j, u32 k) const -> const T& requires(N == 3) {
    return _view(i, j, k);
  }

  auto operator()(u32 i, u32 j, u32 k) -> T& requires(N == 3) {
    return _view(i, j, k);
  }

  auto operator()(u32 i, u32 j, u32 k, u32 l) const -> const T& requires(N == 4) {
    return _view(i, j, k, l);
  }

  auto operator()(u32 i, u32 j, u32 k, u32 l) -> T& requires(N == 4) {
    return _view(i, j, k, l);
  }
};

template <class T, u32 N>
auto empty(const u32 (&shape)[N]) -> NdArray<T, N> {
  return NdArray<T, N>::new_(shape);
}

template <class T, u32 N>
auto empty_like(const NdArray<T, N>& array) -> NdArray<T, N> {
  const auto& shape = array.shape();
  return NdArray<T, N>::new_(shape);
}

template <class T, u32 N>
auto zero(const u32 (&shape)[N]) -> NdArray<T, N> {
  return NdArray<T, N>::new_zeroed(shape);
}

template <class T, u32 N>
auto zero_like(const NdArray<T, N>& array) -> NdArray<T, N> {
  const auto& shape = array.shape();
  return NdArray<T, N>::new_zeroed(shape);
}

}  // namespace sfc::math
