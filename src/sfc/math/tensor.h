#pragma once

#include "sfc/alloc.h"
#include "sfc/alloc/mem_pool.h"
#include "sfc/math/ndview.h"

namespace sfc::math {

template <class T, u32 N>
class Tensor;

template <class T, u32 N>
class [[nodiscard]] Tensor {
  using A = mem_pool::Allocator;
  using Buff = raw_buf::RawBuf<T, A>;
  using View = NdView<T, N>;
  using shape_t = u32[N];
  using strides_t = u32[N];

  Buff _buff{};
  View _view{};

 public:
  Tensor() noexcept {}
  ~Tensor() {}

  Tensor(Tensor&& other) noexcept = default;
  Tensor& operator=(Tensor&& other) noexcept = default;

  static auto from_buf(Buff buf, const u32 (&shape)[N]) -> Tensor {
    const auto ptr = ptr::cast<T>(buf.ptr());
    auto res = Tensor{};
    res._buff = mem::move(buf);
    res._view = View::with_shape(ptr, shape);
    return res;
  }

  static auto new_(const u32 (&shape)[N], A alloc = {}) -> Tensor {
    const auto numel = View::with_shape(nullptr, shape).numel();
    auto buf = Buff::with_capacity(numel, mem::move(alloc));
    return Tensor::from_buf(mem::move(buf), shape);
  }

  static auto new_zeroerd(const u32 (&shape)[N], A alloc = {}) -> Tensor {
    const auto numel = View::with_shape(nullptr, shape).numel();
    auto buf = Buff::with_capacity_zeroed(numel, mem::move(alloc));
    return Tensor::from_buf(mem::move(buf), shape);
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

  auto as_slice() const -> Slice<const T> {
    return {_buff.ptr(), _buff.cap()};
  }

  auto as_mut_slice() -> Slice<T> {
    return {_buff.ptr(), _buff.cap()};
  }

  auto allocator() -> mem_pool::Allocator& {
    return _buff.allocator();
  }

 public:
  operator View() const {
    return _view;
  }

  auto operator*() const -> View {
    return _view;
  }

  auto operator[](u32 idx) -> NdView<T, N - 1> requires(N > 1) {
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
auto empty(const u32 (&shape)[N]) -> Tensor<T, N> {
  return Tensor<T, N>::new_(shape);
}

template <class T, u32 N>
auto empty_like(const Tensor<T, N>& array) -> Tensor<T, N> {
  const auto& shape = array.shape();
  return Tensor<T, N>::new_(shape);
}

template <class T, u32 N>
auto zero(const u32 (&shape)[N]) -> Tensor<T, N> {
  return Tensor<T, N>::new_zeroerd(shape);
}

template <class T, u32 N>
auto zero_like(const Tensor<T, N>& array) -> Tensor<T, N> {
  const auto& shape = array.shape();
  return Tensor<T, N>::new_zeroerd(shape);
}

}  // namespace sfc::math
