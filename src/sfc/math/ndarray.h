#pragma once

#include "sfc/math/ndslice.h"
#include "sfc/alloc.h"

namespace sfc::math {

template <class T, u32 N, class A = alloc::Global>
class [[nodiscard]] NdArray {
  using Buff = raw_buf::RawBuf<T, A>;
  using View = NdSlice<T, N>;
  using shape_t = u32[N];
  using strides_t = u32[N];

  Buff _buff{};
  View _view{};

 public:
  NdArray() noexcept {}
  ~NdArray() {}

  NdArray(NdArray&& other) noexcept = default;
  NdArray& operator=(NdArray&& other) noexcept = default;

  static auto from_buf(Buff buf, const u32 (&shape)[N]) -> NdArray {
    const auto ptr = ptr::cast<T>(buf.ptr());
    auto res = NdArray{};
    res._buff = mem::move(buf);
    res._view = View::with_shape(ptr, shape);
    return res;
  }

  static auto new_(const u32 (&shape)[N], A alloc = {}) -> NdArray {
    const auto numel = View::with_shape(nullptr, shape).numel();
    auto buf = Buff::with_capacity(numel, mem::move(alloc));
    return NdArray::from_buf(mem::move(buf), shape);
  }

  static auto new_zeroerd(const u32 (&shape)[N], A alloc = {}) -> NdArray {
    const auto numel = View::with_shape(nullptr, shape).numel();
    auto buf = Buff::with_capacity_zeroed(numel, mem::move(alloc));
    return NdArray::from_buf(mem::move(buf), shape);
  }

 public:
  auto as_ptr() const -> const T* {
    return _view._data;
  }

  auto as_mut_ptr() -> T* {
    return _view._data;
  }

  auto numel() const -> u32 {
    return _view.numel();
  }

  auto shape() const -> const shape_t& {
    return _view._shape;
  }

  auto strides() const -> const strides_t& {
    return _view._strides;
  }

  auto as_buf() -> Buff& {
    return _buff;
  }

  auto as_view() const -> View {
    return _view;
  }

  auto as_slice() const -> Slice<const T> {
    return {_buff.ptr(), _buff.cap()};
  }

  auto as_mut_slice() -> Slice<T> {
    return {_buff.ptr(), _buff.cap()};
  }

  auto allocator() -> A& {
    return _buff.allocator();
  }

 public:
  operator View() const {
    return _view;
  }

  auto operator[](u32 idx) -> NdSlice<T, N - 1> requires(N > 1) {
    return _view[idx];
  }

 public:
  auto operator[](u32 idx) const -> T requires(N == 1) {
    return _view[idx];
  }

  auto operator[](u32 idx) -> T& requires(N == 1) {
    return _view[idx];
  }

  auto operator[](u32 i, u32 j) const -> T requires(N == 2) {
    return _view[i, j];
  }

  auto operator[](u32 i, u32 j) -> T& requires(N == 2) {
    return _view[i, j];
  }

  auto operator[](u32 i, u32 j, u32 k) const -> T requires(N == 3) {
    return _view[i, j, k];
  }

  auto operator[](u32 i, u32 j, u32 k) -> T& requires(N == 3) {
    return _view[i, j, k];
  }

  auto operator[](u32 i, u32 j, u32 k, u32 l) const -> T requires(N == 4) {
    return _view[i, j, k, l];
  }

  auto operator[](u32 i, u32 j, u32 k, u32 l) -> T& requires(N == 4) {
    return _view[i, j, k, l];
  }

 public:
  void fmt(auto& f) const {
    _view.fmt(f);
  }

  void for_each(auto&& f) const {
    _view.for_each(f);
  }

  void for_each_mut(auto&& f) {
    _view.for_each_mut(f);
  }

  void fill_with(auto&& f) {
    _view.fill_with(f);
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
  return NdArray<T, N>::new_zeroerd(shape);
}

template <class T, u32 N>
auto zero_like(const NdArray<T, N>& array) -> NdArray<T, N> {
  const auto& shape = array.shape();
  return NdArray<T, N>::new_zeroerd(shape);
}

}  // namespace sfc::math
