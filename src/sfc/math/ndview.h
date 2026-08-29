#pragma once

#include "sfc/math/vec.h"

namespace sfc::math {

template <class T, u32 N = 1>
struct NdView;

template <class T>
struct NdView<T, 1> {
  static constexpr u32 NDIM = 1U;
  using Item = T;

  T* _data = nullptr;
  u32 _shape[1] = {};
  u32 _strides[1] = {};

 public:
  __hd NdView() noexcept : _data{nullptr}, _shape{0}, _strides{0} {}

  __hd NdView(T* data, const u32 (&shape)[1], const u32 (&strides)[1])
      : _data{data}, _shape{shape[0]}, _strides{strides[0]} {}

  __hd NdView(T* data, const u32 (&shape)[1]) : _data{data}, _shape{shape[0]}, _strides{1} {}

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

 public:
  __hd auto contains(u32 i) const -> bool {
    return i < _shape[0];
  }

  __hd auto get(const u32 (&idx)[1]) const -> T {
    const auto offset = idx[0] * _strides[0];
    return _data[offset];
  }

  __hd void set(const u32 (&idx)[1], const T& value) {
    const auto offset = idx[0] * _strides[0];
    _data[offset] = value;
  }

  __hd auto operator[](const u32 (&idx)[1]) const -> const T& {
    const auto offset = idx[0] * _strides[0];
    return _data[offset];
  }

  __hd auto operator[](const u32 (&idx)[1]) -> T& {
    const auto offset = idx[0] * _strides[0];
    return _data[offset];
  }

  __hd auto operator[](u32 i) const -> T {
    return _data[i * _strides[0]];
  }

  __hd auto operator[](u32 i) -> T& {
    return _data[i * _strides[0]];
  }

 public:
#ifndef __CUDACC__
  auto is_contiguous() const -> bool {
    return _strides[0] == 1;
  }

  void for_each(auto&& f) const {
    for (auto i = 0U; i < _shape[0]; ++i) {
      const auto& e = _data[i * _strides[0]];
      f(e, i);
    }
  }

  void for_each_mut(auto&& f) {
    for (auto i = 0U; i < _shape[0]; ++i) {
      auto& e = _data[i * _strides[0]];
      f(e, i);
    }
  }

  void fill_with(auto&& f) {
    for (auto i = 0U; i < _shape[0]; ++i) {
      auto val = f(i);
      _data[i * _strides[0]] = val;
    }
  }

  void fmt(auto& f) const {
    auto self = *this;
    f.write_str("[");
    for (auto i = 0U; i < _shape[0]; ++i) {
      if (i != 0) f.write_str(", ");
      f.write_val(self[i]);
    }
    f.write_str("]");
  }
#endif
};

template <class T>
struct NdView<T, 2> {
  static constexpr u32 NDIM = 2U;
  using Item = T;

  T* _data = nullptr;
  u32 _shape[2] = {};
  u32 _strides[2] = {};

 public:
  __hd NdView() noexcept : _data{nullptr}, _shape{0, 0}, _strides{0, 0} {}

  __hd NdView(T* data, const u32 (&shape)[2], const u32 (&strides)[2])
      : _data{data}, _shape{shape[0], shape[1]}, _strides{strides[0], strides[1]} {}

  __hd NdView(T* data, const u32 (&shape)[2]) : _data{data}, _shape{shape[0], shape[1]}, _strides{shape[1], 1} {}

 public:
  __hd auto len() const -> u32 {
    return _shape[0];
  }

  __hd auto data() const -> T* {
    return _data;
  }

  __hd auto numel() const -> u32 {
    return _shape[0] * _shape[1];
  }

 public:
  __hd auto contains(const u32 (&idx)[2]) const -> bool {
    return idx[0] < _shape[0] && idx[1] < _shape[1];
  }

  __hd auto get(const u32 (&idx)[2]) const -> T {
    const auto offset = idx[0] * _strides[0] + idx[1] * _strides[1];
    return _data[offset];
  }

  __hd void set(const u32 (&idx)[2], const T& value) {
    const auto offset = idx[0] * _strides[0] + idx[1] * _strides[1];
    _data[offset] = value;
  }

  __hd auto operator[](const u32 (&idx)[2]) const -> const T& {
    const auto offset = idx[0] * _strides[0] + idx[1] * _strides[1];
    return _data[offset];
  }

  __hd auto operator[](const u32 (&idx)[2]) -> T& {
    const auto offset = idx[0] * _strides[0] + idx[1] * _strides[1];
    return _data[offset];
  }

  __hd auto operator[](u32 i) const -> NdView<T, 1> {
    return NdView<T, 1>{_data + i * _strides[0], {_shape[1]}, {_strides[1]}};
  }

 public:
#ifndef __CUDACC__
  auto is_contiguous() const -> bool {
    return _strides[0] == _shape[1] && _strides[1] == 1;
  }

  auto transpose() const -> NdView {
    const u32 shape[] = {_shape[1], _shape[0]};
    const u32 strides[] = {_strides[1], _strides[0]};
    return {_data, shape, strides};
  }

  void for_each(auto&& f) const {
    for (u32 i = 0U; i < _shape[0]; ++i) {
      const auto row = (*this)[i];
      for (u32 j = 0U; j < _shape[1]; ++j) {
        const auto& val = row[j];
        f(val, i, j);
      }
    }
  }

  void for_each_mut(auto&& f) {
    for (u32 i = 0U; i < _shape[0]; ++i) {
      auto row = (*this)[i];
      for (u32 j = 0U; j < _shape[1]; ++j) {
        auto& val = row[j];
        f(val, i, j);
      }
    }
  }

  void fill_with(auto&& f) {
    for (u32 i = 0U; i < _shape[0]; ++i) {
      auto row = (*this)[i];
      for (u32 j = 0U; j < _shape[1]; ++j) {
        auto val = f(i, j);
        row[j] = val;
      }
    }
  }

  auto fmt(auto& f) const -> void {
    f.write_str("[");
    for (u32 i = 0U; i < _shape[0]; ++i) {
      if (i != 0) f.write_str(",\n  ");
      const auto row = (*this)[i];
      f.write_val(row);
    }
    f.write_str("]");
  }
#endif
};

template <class T>
struct NdView<T, 3> {
  static constexpr u32 NDIM = 3U;
  using Item = T;

  T* _data = nullptr;
  u32 _shape[3] = {};
  u32 _strides[3] = {};

 public:
  __hd NdView() noexcept : _data{nullptr}, _shape{0, 0, 0}, _strides{0, 0, 0} {}

  __hd NdView(T* data, const u32 (&shape)[3], const u32 (&strides)[3])
      : _data{data}, _shape{shape[0], shape[1], shape[2]}, _strides{strides[0], strides[1], strides[2]} {}

  __hd NdView(T* data, const u32 (&shape)[3])
      : _data{data}, _shape{shape[0], shape[1], shape[2]}, _strides{shape[1] * shape[2], shape[2], 1} {}

 public:
  __hd auto len() const -> u32 {
    return _shape[0];
  }

  __hd auto data() const -> T* {
    return _data;
  }

  __hd auto numel() const -> u32 {
    return _shape[0] * _shape[1] * _shape[2];
  }

 public:
  __hd auto contains(const u32 (&idx)[3]) const -> bool {
    return idx[0] < _shape[0] && idx[1] < _shape[1] && idx[2] < _shape[2];
  }

  __hd auto get(const u32 (&idx)[3]) const -> T {
    const auto offset = idx[0] * _strides[0] + idx[1] * _strides[1] + idx[2] * _strides[2];
    return _data[offset];
  }

  __hd auto set(const u32 (&idx)[3], const T& value) {
    const auto offset = idx[0] * _strides[0] + idx[1] * _strides[1] + idx[2] * _strides[2];
    _data[offset] = value;
  }

  __hd auto operator[](const u32 (&idx)[3]) const -> const T& {
    const auto offset = idx[0] * _strides[0] + idx[1] * _strides[1] + idx[2] * _strides[2];
    return _data[offset];
  }

  __hd auto operator[](const u32 (&idx)[3]) -> T& {
    const auto offset = idx[0] * _strides[0] + idx[1] * _strides[1] + idx[2] * _strides[2];
    return _data[offset];
  }

  __hd auto operator[](u32 i) const -> NdView<T, 2> {
    return NdView<T, 2>{
        _data + i * _strides[0],
        {_shape[1], _shape[2]},
        {_strides[1], _strides[2]},
    };
  }

 public:
#ifndef __CUDACC__
  auto is_contiguous() const -> bool {
    return _strides[0] == _shape[1] * _shape[2] && _strides[1] == _shape[2] && _strides[2] == 1;
  }

  void for_each(auto&& f) const {
    for (auto i = 0U; i < _shape[0]; ++i) {
      const auto mat = (*this)[i];
      mat.for_each([&](const T& e, auto j, auto k) { f(e, i, j, k); });
    }
  }

  void for_each_mut(auto&& f) {
    for (auto i = 0U; i < _shape[0]; ++i) {
      auto mat = (*this)[i];
      mat.for_each_mut([&](T& e, auto j, auto k) { f(e, i, j, k); });
    }
  }

  void fill_with(auto&& f) {
    for (u32 i = 0U; i < _shape[0]; ++i) {
      auto mat = (*this)[i];
      mat.fill_with([&](auto j, auto k) { return f(i, j, k); });
    }
  }

  void fmt(auto& f) const {
    f.write_str("[");
    for (auto i = 0U; i < _shape[0]; ++i) {
      if (i != 0) f.write_str(",\n  ");
      const auto mat = (*this)[i];
      f.write_val(mat);
    }
    f.write_str("]");
  }
#endif
};

template <class T>
struct NdView<T, 4> {
  static constexpr u32 NDIM = 4U;
  using Item = T;

  T* _data = nullptr;
  u32 _shape[4] = {};
  u32 _strides[4] = {};

 public:
  __hd NdView() noexcept : _data{nullptr}, _shape{0, 0, 0, 0}, _strides{0, 0, 0, 0} {}

  __hd NdView(T* data, const u32 (&shape)[4], const u32 (&strides)[4])
      : _data{data}
      , _shape{shape[0], shape[1], shape[2], shape[3]}
      , _strides{strides[0], strides[1], strides[2], strides[3]} {}

  __hd NdView(T* data, const u32 (&shape)[4])
      : _data{data}
      , _shape{shape[0], shape[1], shape[2], shape[3]}
      , _strides{shape[1] * shape[2] * shape[3], shape[2] * shape[3], shape[3], 1} {}

 public:
  __hd auto len() const -> u32 {
    return _shape[0];
  }

  __hd auto data() const -> T* {
    return _data;
  }

  __hd auto numel() const -> u32 {
    return _shape[0] * _shape[1] * _shape[2] * _shape[3];
  }

 public:
  __hd auto contains(const u32 (&idx)[4]) const -> bool {
    return idx[0] < _shape[0] && idx[1] < _shape[1] && idx[2] < _shape[2] && idx[3] < _shape[3];
  }

  __hd auto get(const u32 (&idx)[4]) const -> T {
    const auto offset = idx[0] * _strides[0] + idx[1] * _strides[1] + idx[2] * _strides[2] + idx[3] * _strides[3];
    return _data[offset];
  }

  __hd auto set(const u32 (&idx)[4], const T& value) {
    const auto offset = idx[0] * _strides[0] + idx[1] * _strides[1] + idx[2] * _strides[2] + idx[3] * _strides[3];
    _data[offset] = value;
  }

  __hd auto operator[](const u32 (&idx)[4]) const -> const T& {
    const auto offset = idx[0] * _strides[0] + idx[1] * _strides[1] + idx[2] * _strides[2] + idx[3] * _strides[3];
    return _data[offset];
  }

  __hd auto operator[](const u32 (&idx)[4]) -> T& {
    const auto offset = idx[0] * _strides[0] + idx[1] * _strides[1] + idx[2] * _strides[2] + idx[3] * _strides[3];
    return _data[offset];
  }

  __hd auto operator[](u32 i) const -> NdView<T, 3> {
    return NdView<T, 3>{
        _data + i * _strides[0],
        {_shape[1], _shape[2], _shape[3]},
        {_strides[1], _strides[2], _strides[3]},
    };
  }
};

template <class T, u32 A, u32 B>
auto reshape(NdView<T, A> src, const u32 (&new_shape)[B]) -> NdView<T, B> {
  if (!src.is_contiguous()) return {};
  auto dst = NdView<T, B>{src._data, new_shape};
  if (new_shape[0] == 0) {
    dst._shape[0] = 1;                          // set to 1 temporarily
    dst._shape[0] = src.numel() / dst.numel();  // adjust shape[0]
  }
  if (src.numel() != dst.numel()) return {};
  return dst;
}

}  // namespace sfc::math
