#pragma once

#include "sfc/math/vec.h"

namespace sfc::math {

template <class T, int Dims>
struct NdView;

template <class T>
struct NdView<T, 1> {
  static constexpr u32 NDIM = 1U;
  using Shape = u32[NDIM];
  using Strides = u32[NDIM];
  using Index = u32[NDIM];

  T* _data;
  Shape _shape;
  Strides _strides;

 public:
  template <u32 NX>
  __hd NdView(T (&v)[NX]) : _data{v}, _shape{NX}, _strides{1} {}

  __hd NdView(T* data, const Shape& shape, const Strides& strides)
      : _data{data}, _shape{shape[0]}, _strides{strides[0]} {}

  __hd auto len() const -> u32 {
    return _shape[0];
  }

  __hd auto data() const -> const T* {
    return _data;
  }

  __hd auto shape() const -> const Shape& {
    return _shape;
  }

  __hd auto strides() const -> const Strides& {
    return _strides;
  }

  __hd auto numel() const -> u32 {
    return _shape[0];
  }

  __hd auto contains(const Index& indices) const -> bool {
    return indices[0] < _shape[0];
  }

  __hd auto operator[](u32 idx) const -> const T& {
    return _data[idx * _strides[0]];
  }

  __hd auto operator[](u32 idx) -> T& {
    return _data[idx * _strides[0]];
  }

  __hd auto operator[](const Index& indices) const -> const T& {
    return _data[indices[0] * _strides[0]];
  }

  __hd auto operator[](const Index& indices) -> T& {
    return _data[indices[0] * _strides[0]];
  }

  __hd auto is_contiguous() const -> bool {
    return _strides[0] == 1;
  }

 public:
  void imap(auto&& f) const {
    for (auto i = 0U; i < _shape[0]; ++i) {
      const auto& e = _data[i * _strides[0]];
      f(i, e);
    }
  }

  void imap_mut(auto&& f) {
    for (auto i = 0U; i < _shape[0]; ++i) {
      auto& e = _data[i * _strides[0]];
      f(i, e);
    }
  }

#ifndef __CUDACC__
  void fmt(auto& f) const {
    Slice{_data, _shape[0]}.fmt(f);
  }
#endif
};

template <class T>
struct NdView<T, 2> {
  static constexpr u32 NDIM = 2;
  using Shape = u32[NDIM];
  using Strides = u32[NDIM];
  using Index = u32[NDIM];

  T* _data;
  Shape _shape;
  Strides _strides;

 public:
  template <u32 NY, u32 NX>
  __hd NdView(T (&v)[NY][NX]) : _data{v[0]}, _shape{NY, NX}, _strides{NX, 1} {}

  __hd NdView(T* data, const Shape& shape, const Strides& strides)
      : _data{data}, _shape{shape[0], shape[1]}, _strides{strides[0], strides[1]} {}

  __hd auto len() const -> u32 {
    return _shape[0];
  }

  __hd auto data() const -> const T* {
    return _data;
  }

  __hd auto shape() const -> const Shape& {
    return _shape;
  }

  __hd auto strides() const -> const Strides& {
    return _strides;
  }

  __hd auto numel() const -> u32 {
    return _shape[0] * _shape[1];
  }

  __hd auto contains(const Index& indices) const -> bool {
    return indices[0] < _shape[0] && indices[1] < _shape[1];
  }

  __hd auto operator[](u32 x) const -> NdView<T, NDIM - 1> {
    const auto p = _data + x * _strides[0];
    return NdView<T, NDIM - 1>{p, {_shape[1]}, {_strides[1]}};
  }

  __hd auto operator[](const Index& indices) const -> const T& {
    const auto offset = indices[0] * _strides[0] + indices[1] * _strides[1];
    return _data[offset];
  }

  __hd auto operator[](const Index& indices) -> T& {
    const auto offset = indices[0] * _strides[0] + indices[1] * _strides[1];
    return _data[offset];
  }

 public:
  void imap(auto&& f) const {
    for (auto i = 0U; i < _shape[0]; ++i) {
      const auto row = (*this)[i];
      for (auto j = 0U; j < _shape[1]; ++j) {
        const auto& e = row[j];
        f(i, j, e);
      }
    }
  }

  void imap_mut(auto&& f) {
    for (auto i = 0U; i < _shape[0]; ++i) {
      auto row = (*this)[i];
      for (auto j = 0U; j < _shape[1]; ++j) {
        auto& e = row[j];
        f(i, j, e);
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
struct NdView<T, 3> {
  static constexpr u32 NDIM = 3U;
  using Shape = u32[NDIM];
  using Strides = u32[NDIM];
  using Index = u32[NDIM];

  T* _data;
  Shape _shape;
  Strides _strides;

 public:
  template <u32 NZ, u32 NY, u32 NX>
  __hd NdView(T (&v)[NZ][NY][NX]) : _data{v[0][0]}, _shape{NZ, NY, NX}, _strides{NY * NX, NX, 1} {}

  __hd NdView(T* data, const Shape& shape, const Strides& strides)
      : _data{data}, _shape{shape[0], shape[1], shape[2]}, _strides{strides[0], strides[1], strides[2]} {}

  __hd auto len() const -> u32 {
    return _shape[0];
  }

  __hd auto data() const -> const T* {
    return _data;
  }

  __hd auto shape() const -> const Shape& {
    return _shape;
  }

  __hd auto strides() const -> const Strides& {
    return _strides;
  }

  __hd auto numel() const -> u32 {
    return _shape[0] * _shape[1] * _shape[2];
  }

  __hd auto contains(const Index& indices) const -> bool {
    return indices[0] < _shape[0] && indices[1] < _shape[1] && indices[2] < _shape[2];
  }

  __hd auto operator[](u32 x) const -> NdView<T, NDIM - 1> {
    const T* data = _data + x * _strides[0];
    const u32 shape[] = {_shape[1], _shape[2]};
    const u32 strides[] = {_strides[1], _strides[2]};
    return {data, shape, strides};
  }

  __hd auto operator[](const Index& indices) const -> const T& {
    const auto offset = indices[0] * _strides[0] + indices[1] * _strides[1] + indices[2] * _strides[2];
    return _data[offset];
  }

  __hd auto operator[](const Index& indices) -> T& {
    const auto offset = indices[0] * _strides[0] + indices[1] * _strides[1] + indices[2] * _strides[2];
    return _data[offset];
  }

 public:
  void imap(auto&& f) const {
    for (auto i = 0U; i < _shape[0]; ++i) {
      const auto mat = (*this)[i];
      mat.imap([&](u32 j, u32 k, const T& e) { f(i, j, k, e); });
    }
  }

  void imap_mut(auto&& f) {
    for (auto i = 0U; i < _shape[0]; ++i) {
      auto mat = (*this)[i];
      mat.imap_mut([&](u32 j, u32 k, T& e) { f(i, j, k, e); });
    }
  }
};

template <class T>
struct NdView<T, 4> {
  static constexpr auto NDIM = 4;
  using Shape = u32[NDIM];
  using Strides = u32[NDIM];
  using Index = u32[NDIM];
  T* _data;
  Shape _shape;
  Strides _strides;

 public:
  template <u32 NW, u32 NZ, u32 NY, u32 NX>
  __hd NdView(T (&v)[NW][NZ][NY][NX])
      : _data{v[0][0][0]}, _shape{NW, NZ, NY, NX}, _strides{NZ * NY * NX, NY * NX, NX, 1} {}

  __hd NdView(T* data, const Shape& shape, const Strides& strides)
      : _data{data}
      , _shape{shape[0], shape[1], shape[2], shape[3]}
      , _strides{strides[0], strides[1], strides[2], strides[3]} {}

  __hd auto len() const -> u32 {
    return _shape[0];
  }

  __hd auto data() const -> const T* {
    return _data;
  }

  __hd auto shape() const -> const Shape& {
    return _shape;
  }

  __hd auto strides() const -> const Strides& {
    return _strides;
  }

  __hd auto numel() const -> u32 {
    return _shape[0] * _shape[1] * _shape[2] * _shape[3];
  }

  __hd auto contains(const Index& indices) const -> bool {
    return indices[0] < _shape[0] && indices[1] < _shape[1] && indices[2] < _shape[2] && indices[3] < _shape[3];
  }

  __hd auto operator[](u32 idx) const -> NdView<T, NDIM - 1> {
    const T* data = _data + idx * _strides[0];
    const u32 shape[] = {_shape[1], _shape[2], _shape[3]};
    const u32 strides[] = {_strides[1], _strides[2], _strides[3]};
    return {data, shape, strides};
  }

  __hd auto operator[](const Index& indices) const -> const T& {
    const auto offset = indices[0] * _strides[0] + indices[1] * _strides[1] + indices[2] * _strides[2] +
                        indices[3] * _strides[3];
    return _data[offset];
  }

  __hd auto operator[](const Index& indices) -> T& {
    const auto offset = indices[0] * _strides[0] + indices[1] * _strides[1] + indices[2] * _strides[2] +
                        indices[3] * _strides[3];
    return _data[offset];
  }
};

template <class T, int N0>
NdView(T (&)[N0]) -> NdView<T, 1>;

template <class T, int N0, int N1>
NdView(T (&)[N0][N1]) -> NdView<T, 2>;

}  // namespace sfc::math
