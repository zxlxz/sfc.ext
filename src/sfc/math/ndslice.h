#pragma once

#include "sfc/math/vec.h"

namespace sfc::math {

template <class T, int N>
struct NdSlice;

template <class T>
struct NdSlice<T, 1> {
  static constexpr u32 NDIM = 1U;
  using Item = T;

  T* _data = nullptr;
  u32 _shape[NDIM] = {};
  u32 _strides[NDIM] = {};

 public:
  __hd NdSlice() noexcept = default;

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

  __hd auto operator[](const u32 (&idx)[1]) const -> T {
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
  void imap(auto&& f) const {
    for (auto i = 0U; i < _shape[0]; ++i) {
      const auto& val = (*this)[i];
      if constexpr (requires { f(i, val); }) {
        f(i, val);
      } else {
        f({i}, val);
      }
    }
  }

  void imap_mut(auto&& f) {
    for (auto i = 0U; i < _shape[0]; ++i) {
      auto& val = (*this)[i];
      if constexpr (requires { f(i, val); }) {
        f(i, val);
      } else {
        f({i}, val);
      }
    }
  }

#ifndef __CUDACC__
  void fmt(auto& f) const {
    Slice{_data, _shape[0]}.fmt(f);
  }
#endif
};

template <class T>
struct NdSlice<T, 2> {
  static constexpr u32 NDIM = 2;
  using Item = T;

  T* _data = nullptr;
  u32 _shape[NDIM] = {};
  u32 _strides[NDIM] = {};

 public:
  __hd NdSlice() noexcept = default;

  __hd NdSlice(T* data, const u32 (&shape)[NDIM], const u32 (&strides)[NDIM])
      : _data{data}, _shape{shape[0], shape[1]}, _strides{strides[0], strides[1]} {}

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
    const auto data = _data + x * _strides[0];
    return NdSlice<T, NDIM - 1>{data, {_shape[1]}, {_strides[1]}};
  }

  __hd auto operator[](const u32 (&idx)[NDIM]) const -> T {
    const auto offset = idx[0] * _strides[0] + idx[1] * _strides[1];
    return _data[offset];
  }

  __hd auto operator[](const u32 (&idx)[NDIM]) -> T& {
    const auto offset = idx[0] * _strides[0] + idx[1] * _strides[1];
    return _data[offset];
  }

 public:
  __hd auto contains(u32 i, u32 j) const -> bool {
    return i < _shape[0] && j < _shape[1];
  }

  __hd auto get(u32 i, u32 j) const -> T {
    return (*this)[{i, j}];
  }

  __hd void set(u32 i, u32 j, const T& value) {
    (*this)[{i, j}] = value;
  }

 public:
  void imap(auto&& f) const {
    for (auto i = 0U; i < _shape[0]; ++i) {
      const auto row = (*this)[i];
      for (auto j = 0U; j < _shape[1]; ++j) {
        const auto& val = row[j];
        if constexpr (requires { f(i, j, val); }) {
          f(i, j, val);
        } else {
          f({i, j}, val);
        }
      }
    }
  }

  void imap_mut(auto&& f) {
    for (auto i = 0U; i < _shape[0]; ++i) {
      auto row = (*this)[i];
      for (auto j = 0U; j < _shape[1]; ++j) {
        auto& val = row[j];
        if constexpr (requires { f(i, j, val); }) {
          f(i, j, val);
        } else {
          f({i, j}, val);
        }
      }
    }
  }

#ifndef __CUDACC__
  void fmt(auto& f) const {
    for (auto i = 0U; i < _shape[0]; ++i) {
      const auto row = (*this)[i];
      f.write_fmt("\n  {}", row);
    }
  }
#endif
};

template <class T>
struct NdSlice<T, 3> {
  static constexpr u32 NDIM = 3U;
  using Item = T;

  T* _data = nullptr;
  u32 _shape[NDIM] = {};
  u32 _strides[NDIM] = {};

 public:
  __hd NdSlice() noexcept = default;

  __hd NdSlice(T* data, const u32 (&shape)[NDIM], const u32 (&strides)[NDIM])
      : _data{data}, _shape{shape[0], shape[1], shape[2]}, _strides{strides[0], strides[1], strides[2]} {}

  __hd auto len() const -> u32 {
    return _shape[0];
  }

  __hd auto data() const -> const T* {
    return _data;
  }

  __hd auto numel() const -> u32 {
    return _shape[0] * _shape[1] * _shape[2];
  }

  __hd auto operator[](u32 x) const -> NdSlice<T, NDIM - 1> {
    const auto data = _data + x * _strides[0];
    return {data, {_shape[1], _shape[2]}, {_strides[1], _strides[2]}};
  }

  __hd auto operator[](const u32 (&idx)[NDIM]) const -> T {
    const auto offset = idx[0] * _strides[0] + idx[1] * _strides[1] + idx[2] * _strides[2];
    return _data[offset];
  }

  __hd auto operator[](const u32 (&idx)[NDIM]) -> T& {
    const auto offset = idx[0] * _strides[0] + idx[1] * _strides[1] + idx[2] * _strides[2];
    return _data[offset];
  }

 public:
  __hd auto contains(u32 i, u32 j, u32 k) const -> bool {
    return i < _shape[0] && j < _shape[1] && k < _shape[2];
  }

  __hd auto get(u32 i, u32 j, u32 k) const -> T {
    const auto offset = i * _strides[0] + j * _strides[1] + k * _strides[2];
    return _data[offset];
  }

  __hd void set(u32 i, u32 j, u32 k, const T& value) {
    const auto offset = i * _strides[0] + j * _strides[1] + k * _strides[2];
    _data[offset] = value;
  }

 public:
  void imap(auto&& f) const {
    for (auto i = 0U; i < _shape[0]; ++i) {
      const auto mat = (*this)[i];
      for (auto j = 0U; j < _shape[1]; ++j) {
        const auto row = mat[j];
        for (auto k = 0U; k < _shape[2]; ++k) {
          const auto& val = row[k];
          if constexpr (requires { f(i, j, k, val); }) {
            f(i, j, k, val);
          } else {
            f({i, j, k}, val);
          }
        }
      }
    }
  }

  void imap_mut(auto&& f) {
    for (auto i = 0U; i < _shape[0]; ++i) {
      auto mat = (*this)[i];
      for (auto j = 0U; j < _shape[1]; ++j) {
        auto row = mat[j];
        for (auto k = 0U; k < _shape[2]; ++k) {
          auto& val = row[k];
          if constexpr (requires { f(i, j, k, val); }) {
            f(i, j, k, val);
          } else {
            f({i, j, k}, val);
          }
        }
      }
    }
  }
};

template <class T>
struct NdSlice<T, 4> {
  static constexpr u32 NDIM = 4U;
  using Item = T;

  T* _data = nullptr;
  u32 _shape[NDIM] = {};
  u32 _strides[NDIM] = {};

 public:
  __hd NdSlice() noexcept = default;

  __hd NdSlice(T* data, const u32 (&shape)[NDIM], const u32 (&strides)[NDIM])
      : _data{data}
      , _shape{shape[0], shape[1], shape[2], shape[3]}
      , _strides{strides[0], strides[1], strides[2], strides[3]} {}

  __hd auto len() const -> u32 {
    return _shape[0];
  }

  __hd auto data() const -> const T* {
    return _data;
  }

  __hd auto numel() const -> u32 {
    return _shape[0] * _shape[1] * _shape[2] * _shape[3];
  }

  __hd auto operator[](u32 idx) const -> NdSlice<T, NDIM - 1> {
    const auto data = _data + idx * _strides[0];
    return {data, {_shape[1], _shape[2], _shape[3]}, {_strides[1], _strides[2], _strides[3]}};
  }

 public:
  __hd auto contains(u32 i, u32 j, u32 k, u32 l) const -> bool {
    return i < _shape[0] && j < _shape[1] && k < _shape[2] && l < _shape[3];
  }

  __hd auto get(u32 i, u32 j, u32 k, u32 l) const -> T {
    const auto offset = i * _strides[0] + j * _strides[1] + k * _strides[2] + l * _strides[3];
    return _data[offset];
  }

  __hd void set(u32 i, u32 j, u32 k, u32 l, const T& value) {
    const auto offset = i * _strides[0] + j * _strides[1] + k * _strides[2] + l * _strides[3];
    _data[offset] = value;
  }
};

}  // namespace sfc::math
