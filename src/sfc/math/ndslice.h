#pragma once

#include "sfc/math/vec.h"

namespace sfc::math {

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

  __hd auto len() const -> u32 {
    return _shape[0];
  }

  __hd auto data() const -> const T* {
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

  void for_each(auto&& f, auto... args) {
    for (auto i = 0U; i < _shape[0]; ++i) {
      auto& t = (*this)[i];
      if constexpr (requires { f(args..., i, t); }) {
        f(args..., i, t);
      } else if constexpr (requires { f({i}, t); }) {
        f({args..., i}, t);
      } else {
        f(t);
      }
    }
  }

  void for_each(auto&& f, auto... args) const {
    for (auto i = 0U; i < _shape[0]; ++i) {
      const auto& t = (*this)[i];
      if constexpr (requires { f(args..., i, t); }) {
        f(args..., i, t);
      } else if constexpr (requires { f({i}, t); }) {
        f({args..., i}, t);
      } else {
        f(t);
      }
    }
  }

  void fmt(auto& f) const {
    auto self = *this;
    f.write_str("[");
    self.for_each([&](u32 i, const T& val) {
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
    for (auto i = 0; i < NDIM; ++i) {
      _shape[i] = shape[i];
      _strides[i] = strides[i];
    }
  }

  __hd auto len() const -> u32 {
    return _shape[0];
  }

  __hd auto data() const -> const T* {
    return _data;
  }

  __hd auto numel() const -> u32 {
    return _shape[0] * _shape[1];
  }

  __hd auto operator[](u32 x) const -> NdSlice<T, NDIM - 1> {
    auto res = NdSlice<T, N - 1>{};
    res._data = _data + x * _strides[0];
    for (auto i = 0U; i < NDIM - 1; ++i) {
      res._shape[i] = _shape[i + 1];
      res._strides[i] = _strides[i + 1];
    }
    return res;
  }

  __hd auto offset(const u32 (&idx)[NDIM]) const -> u32 {
    if constexpr (NDIM == 1) {
      return idx[0] * _strides[0];
    } else if constexpr (NDIM == 2) {
      return idx[0] * _strides[0] + idx[1] * _strides[1];
    } else if constexpr (NDIM == 3) {
      return idx[0] * _strides[0] + idx[1] * _strides[1] + idx[2] * _strides[2];
    } else if constexpr (NDIM == 4) {
      return idx[0] * _strides[0] + idx[1] * _strides[1] + idx[2] * _strides[2] + idx[3] * _strides[3];
    } else {
      static_assert(NDIM <= 4, "NdSlice supports up to 4 dimensions.");
    }
  }

  __hd auto operator[](const u32 (&idx)[NDIM]) const -> const T& {
    const auto offset = this->offset(idx);
    return _data[offset];
  }

  __hd auto operator[](const u32 (&idx)[NDIM]) -> T& {
    const auto offset = this->offset(idx);
    return _data[offset];
  }

 public:
  __hd auto contains(const u32 (&idx)[NDIM]) const -> bool {
    return idx[0] < _shape[0] && idx[1] < _shape[1];
  }

  __hd auto get(const u32 (&idx)[NDIM]) const -> T {
    const auto offset = this->offset(idx);
    return _data[offset];
  }

  __hd void set(const u32 (&idx)[NDIM], const T& value) {
    const auto offset = this->offset(idx);
    _data[offset] = value;
  }

 public:
#ifndef __CUDACC__
  auto is_contiguous() const -> bool {
    return _strides[0] == _shape[1] && _strides[1] == 1;
  }

  void for_each(auto&& f, auto... args) {
    for (auto i = 0U; i < _shape[0]; ++i) {
      auto v = (*this)[i];
      v.for_each(f, args..., i);
    }
  }

  void for_each(auto&& f, auto... args) const {
    for (auto i = 0U; i < _shape[0]; ++i) {
      const auto v = (*this)[i];
      v.for_each(f, args..., i);
    }
  }

  void fmt(auto& f) const {
    auto& self = *this;

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
