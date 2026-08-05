#pragma once

#include "sfc/math/vec.h"

namespace sfc::math {

namespace _ops {

template <u32 N>
__hd auto prod(const u32 (&a)[N]) -> u32 {
  static_assert(N <= 8);
  auto ret = 1U;
  if constexpr (N > 0) ret *= a[0];
  if constexpr (N > 1) ret *= a[1];
  if constexpr (N > 2) ret *= a[2];
  if constexpr (N > 3) ret *= a[3];
  if constexpr (N > 4) ret *= a[4];
  if constexpr (N > 5) ret *= a[5];
  if constexpr (N > 6) ret *= a[6];
  if constexpr (N > 7) ret *= a[7];
  return ret;
}

template <u32 N>
__hd auto lt(const u32 (&a)[N], const u32 (&b)[N]) -> bool {
  static_assert(N <= 8);
  auto ret = true;
  if constexpr (N > 0) ret &= a[0] < b[0];
  if constexpr (N > 1) ret &= a[1] < b[1];
  if constexpr (N > 2) ret &= a[2] < b[2];
  if constexpr (N > 3) ret &= a[3] < b[3];
  if constexpr (N > 4) ret &= a[4] < b[4];
  if constexpr (N > 5) ret &= a[5] < b[5];
  if constexpr (N > 6) ret &= a[6] < b[6];
  if constexpr (N > 7) ret &= a[7] < b[7];
  return ret;
}

template <u32 N>
__hd auto dot(const u32 (&a)[N], const u32 (&b)[N]) -> u32 {
  static_assert(N <= 8);
  auto ret = 1U;
  if constexpr (N > 0) ret += a[0] * b[0];
  if constexpr (N > 1) ret += a[1] * b[1];
  if constexpr (N > 2) ret += a[2] * b[2];
  if constexpr (N > 3) ret += a[3] * b[3];
  if constexpr (N > 4) ret += a[4] * b[4];
  if constexpr (N > 5) ret += a[5] * b[5];
  if constexpr (N > 6) ret += a[6] * b[6];
  if constexpr (N > 7) ret += a[7] * b[7];
  return ret;
}

}  // namespace _ops

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

  template <u32 N>
  __hd NdSlice(T (&data)[N]) : _data{data}, _shape{N}, _strides{1} {}

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

  T* _data = nullptr;
  u32 _shape[NDIM] = {};
  u32 _strides[NDIM] = {};

 public:
  __hd NdSlice() noexcept : _data{nullptr}, _shape{0, 0}, _strides{0, 0} {}

  __hd NdSlice(T* data, const u32 (&shape)[NDIM], const u32 (&strides)[NDIM]) : _data{data} {
    for (auto i = 0U; i < NDIM; ++i) {
      _shape[i] = shape[i];
      _strides[i] = strides[i];
    }
  }

 public:
  __hd auto len() const -> u32 {
    return _shape[0];
  }

  __hd auto data() const -> T* {
    return _data;
  }

  __hd auto operator[](u32 x) const -> NdSlice<T, NDIM - 1> {
    auto res = NdSlice<T, N - 1>{};
    res._data = _data + x * _strides[0];
    __builtin_memcpy(res._shape, _shape + 1, sizeof(u32) * (NDIM - 1));
    __builtin_memcpy(res._strides, _strides + 1, sizeof(u32) * (NDIM - 1));
    return res;
    return res;
  }

  __hd auto operator[](const u32 (&idx)[NDIM]) const -> const T& {
    const auto offset = _ops::dot(idx, _strides);
    return _data[offset];
  }

  __hd auto operator[](const u32 (&idx)[NDIM]) -> T& {
    const auto offset = _ops::dot(idx, _strides);
    return _data[offset];
  }

 public:
  __hd auto contains(const u32 (&idx)[NDIM]) const -> bool {
    return _ops::lt(idx, _shape);
  }

  __hd auto get(const u32 (&idx)[NDIM]) const -> T {
    const auto offset = _ops::dot(idx, _strides);
    return _data[offset];
  }

  __hd void set(const u32 (&idx)[NDIM], const T& value) {
    const auto offset = _ops::dot(idx, _strides);
    _data[offset] = value;
  }

 public:
#ifndef __CUDACC__
  __hd auto numel() const -> u32 {
    const auto res = _ops::prod(_shape);
    return res;
  }

  auto is_contiguous() const -> bool {
    if (_strides[NDIM - 1] != 1) return false;
    for (auto i = NDIM - 1; i > 0; --i) {
      if (_strides[i - 1] != _shape[i] * _strides[i]) return false;
    }
    return true;
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
