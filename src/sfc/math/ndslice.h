#pragma once

#include "sfc/math/vec.h"

namespace sfc::math {

template <u32... I>
struct NdOps {
  __hd static auto eq(const auto& a, const auto& b) -> bool {
    return ((a[I] == b[I]) && ...);
  }

  __hd static auto lt(const auto& a, const auto& b) -> bool {
    return ((a[I] < b[I]) && ...);
  }

  __hd static auto prod(const auto& a) {
    return (a[I] * ...);
  }

  __hd static auto dot(const auto& a, const auto& b) {
    return ((a[I] * b[I]) + ...);
  }
};

template <u32 N>
auto ndops() {
  static_assert(N > 0 && N < 8);
  if constexpr (N == 1) return NdOps<0>{};
  if constexpr (N == 2) return NdOps<0, 1>{};
  if constexpr (N == 3) return NdOps<0, 1, 2>{};
  if constexpr (N == 4) return NdOps<0, 1, 2, 3>{};
  if constexpr (N == 5) return NdOps<0, 1, 2, 3, 4>{};
  if constexpr (N == 6) return NdOps<0, 1, 2, 3, 4, 5>{};
  if constexpr (N == 7) return NdOps<0, 1, 2, 3, 4, 5, 6>{};
  if constexpr (N == 8) return NdOps<0, 1, 2, 3, 4, 5, 6, 7>{};
}

template <class T, u32 N = 1>
struct NdSlice;

template <class T>
struct NdSlice<T, 1> {
  static constexpr u32 NDIM = 1U;
  using Item = T;

  T* _data = nullptr;
  u32 _shape[NDIM] = {};
  u32 _strides[NDIM] = {};

 public:
  __hd NdSlice() noexcept : _data{nullptr}, _shape{0}, _strides{0} {}

  __hd NdSlice(T* data, const u32 (&shape)[NDIM], const u32 (&strides)[NDIM])
      : _data{data}, _shape{shape[0]}, _strides{strides[0]} {}

  __hd static auto from_raw(T* data, const u32 (&shape)[NDIM]) -> NdSlice {
    u32 strides[NDIM] = {1};
    return NdSlice{data, shape, strides};
  }

 public:
  __hd auto len() const -> u32 {
    return _shape[0];
  }

  __hd auto data() const -> T* {
    return _data;
  }

  __hd auto numel() const -> u32 {
    return _shape[0];
  }

  __hd auto operator[](u32 i) const -> T {
    return _data[i];
  }

  __hd auto operator[](u32 i) -> T& {
    return _data[i];
  }

  __hd auto operator[](const u32 (&idx)[1]) const -> const T& {
    const auto offset = idx[0] * _strides[0];
    return _data[offset];
  }

  __hd auto operator[](const u32 (&idx)[1]) -> T& {
    const auto offset = idx[0] * _strides[0];
    return _data[offset];
  }

 public:
  __hd auto contains(u32 i) const -> bool {
    return i < _shape[0];
  }

  __hd auto get(u32 i) const -> const T& {
    const auto offset = i * _strides[0];
    return _data[offset];
  }

  __hd void set(u32 i, const T& value) {
    const auto offset = i * _strides[0];
    _data[offset] = value;
  }

 public:
#ifndef __CUDACC__
  auto is_contiguous() const -> bool {
    return _strides[0] == 1;
  }

  void for_each(auto&& f) const {
    for (auto i = 0U; i < _shape[0]; ++i) {
      const auto& t = (*this)[i];
      f(t, i);
    }
  }

  void for_each_mut(auto&& f) {
    for (auto i = 0U; i < _shape[0]; ++i) {
      auto& t = (*this)[i];
      f(t, i);
    }
  }

  void fmt(auto& f) const {
    auto self = *this;
    f.write_str("[");
    self.for_each_mut([&](const T& val, u32 i) {
      if (i != 0) f.write_str(", ");
      f.write_val(val);
    });
    f.write_str("]");
  }
#endif
};

template <class T, u32 N>
struct NdSlice {
  static constexpr u32 NDIM = N;
  using Item = T;
  using NdOp = decltype(ndops<NDIM>());
  T* _data = nullptr;
  u32 _shape[NDIM] = {};
  u32 _strides[NDIM] = {};

 public:
  __hd NdSlice() noexcept : _data{nullptr}, _shape{0, 0}, _strides{0, 0} {}

  template <u32... I>
  __hd NdSlice(T* data, const auto& shape, const auto& strides, NdOps<I...>)
      : _data{data}, _shape{shape[I]...}, _strides{strides[I]...} {}

  __hd NdSlice(T* data, const u32 (&shape)[NDIM], const u32 (&strides)[NDIM])
      : NdSlice{data, shape, strides, ndops<NDIM>()} {}

  __hd static auto from_raw(T* data, const u32 (&shape)[NDIM]) -> NdSlice {
    u32 strides[NDIM] = {};
    strides[NDIM - 1] = 1;
    for (auto i = NDIM - 1; i > 0; --i) {
      strides[i - 1] = shape[i] * strides[i];
    }
    return NdSlice{data, shape, strides};
  }

 public:
  __hd auto len() const -> u32 {
    return _shape[0];
  }

  __hd auto data() const -> T* {
    return _data;
  }

  __hd auto operator[](u32 x) const -> NdSlice<T, NDIM - 1> {
    const auto data = _data + x * _strides[0];
    const auto& shape = *ptr::cast<const u32[NDIM - 1]>(_shape + 1);
    const auto& strides = *ptr::cast<const u32[NDIM - 1]>(_strides + 1);
    const auto ret = NdSlice<T, NDIM - 1>{data, shape, strides};
    return ret;
  }

  __hd auto operator[](const u32 (&idx)[NDIM]) const -> const T& {
    const auto offset = NdOp::dot(idx, _strides);
    return _data[offset];
  }

  __hd auto operator[](const u32 (&idx)[NDIM]) -> T& {
    const auto offset = NdOp::dot(idx, _strides);
    return _data[offset];
  }

 public:
  __hd auto offset(const u32 (&idx)[NDIM]) const -> u32 {
    return NdOp::dot(idx, _strides);
  }

  __hd auto contains(const u32 (&idx)[NDIM]) const -> bool {
    return NdOp::lt(idx, _shape);
  }

  __hd auto get(const u32 (&idx)[NDIM]) const -> T {
    const auto offset = NdOp::dot(idx, _strides);
    return _data[offset];
  }

  __hd void set(const u32 (&idx)[NDIM], const T& value) {
    const auto offset = NdOp::dot(idx, _strides);
    _data[offset] = value;
  }

 public:
#ifndef __CUDACC__
  __hd auto numel() const -> u32 {
    return NdOp::prod(_shape);
  }

  auto is_contiguous() const -> bool {
    const auto tmp = NdSlice::from_raw(_data, _shape);
    return NdOp::eq(tmp._strides, _strides);
  }

  void for_each(auto&& f) const {
    for (auto i = 0U; i < _shape[0]; ++i) {
      const auto v = (*this)[i];
      v.for_each([=, &f](const T& val, auto... j) { f(val, i, j...); });
    }
  }

  void for_each_mut(auto&& f) {
    for (auto i = 0U; i < _shape[0]; ++i) {
      auto v = (*this)[i];
      v.for_each_mut([=, &f](T& val, auto... j) { f(val, i, j...); });
    }
  }

  void fmt(auto& f) const {
    f.write_str("[");
    for (auto i = 0U; i < _shape[0]; ++i) {
      const auto row = (*this)[i];
      if (i != 0) f.write_str(",\n  ");
      f.write_val(row);
    }
    f.write_str("]");
  }
#endif
};

}  // namespace sfc::math
