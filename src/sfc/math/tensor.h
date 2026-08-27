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
  using Buff = Buffer<T, A>;
  using View = NdView<T, N>;
  Buff _buff{};
  View _view{};

 public:
  Tensor() noexcept : _buff{}, _view{nullptr, {}, {}} {}
  ~Tensor() {}

  Tensor(Tensor&& other) noexcept = default;
  Tensor& operator=(Tensor&& other) noexcept = default;

  static auto from_buf(Buff buf, const u32 (&shape)[N]) -> Tensor {
    const auto ptr = ptr::cast<T>(buf.ptr());
    auto res = Tensor{};
    res._buff = mem::move(buf);
    res._view = View::from_raw(ptr, shape);
    return res;
  }

  static auto new_(const u32 (&shape)[N], A alloc = {}) -> Tensor {
    const auto numel = View{nullptr, shape, {}}.numel();
    auto buf = Buff::with_capacity(numel, mem::move(alloc));
    return Tensor::from_buf(mem::move(buf), shape);
  }

  static auto new_zeroerd(const u32 (&shape)[N], A alloc = {}) -> Tensor {
    const auto numel = View{nullptr, shape, {}}.numel();
    auto buf = Buff::with_capacity_zeroed(numel, mem::move(alloc));
    return Tensor::from_buf(mem::move(buf), shape);
  }

  auto as_ptr() const -> const T* {
    return _view._data;
  }

  auto as_mut_ptr() -> T* {
    return _view._data;
  }

  auto numel() const -> u32 {
    return _view.numel();
  }

  auto shape() const -> const u32 (&)[N] {
    return _view._shape;
  }

  auto strides() const -> const u32 (&)[N] {
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

  auto operator[](u32 idx) const -> decltype(auto) {
    return _view[idx];
  }

  auto operator[](u32 idx) -> decltype(auto) {
    return _view[idx];
  }

  auto operator[](const u32 (&idx)[N]) const -> T {
    return _view[idx];
  }

  auto operator[](const u32 (&idx)[N]) -> T& {
    return _view[idx];
  }

  auto get(const u32 (&idx)[N]) const -> T {
    return _view[idx];
  }

  void set(const u32 (&idx)[N], T value) {
    _view[idx] = value;
  }

 public:
  void copy_from(const Tensor& other) {
    _buff.copy_from(other._buff);
  }

  auto clone() const -> Tensor {
    auto res = Tensor::new_(_view._shape, _buff.mem_location());
    res.copy_from(*this);
    return res;
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
