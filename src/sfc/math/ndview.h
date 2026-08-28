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
  u32 _shape[NDIM] = {};
  u32 _strides[NDIM] = {};

 public:
  __hd NdView() noexcept : _data{nullptr}, _shape{0}, _strides{0} {}

  __hd NdView(T* data, const u32 (&shape)[NDIM], const u32 (&strides)[NDIM])
      : _data{data}, _shape{shape[0]}, _strides{strides[0]} {}

  __hd NdView(T* data, const u32 (&shape)[NDIM]) : _data{data}, _shape{shape[0]}, _strides{1} {}

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
    return _data[i * _strides[0]];
  }

  __hd auto operator[](u32 i) -> T& {
    return _data[i * _strides[0]];
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

  __hd auto get(u32 i) const -> T {
    const auto offset = i * _strides[0];
    return _data[offset];
  }

  __hd void set(u32 i, const T& value) {
    const auto offset = i * _strides[0];
    _data[offset] = value;
  }

  __hd auto get_nearest(f32 x) const -> T {
    const auto ix = math::roundf(x);
    if (ix < 0 || ix > i32(_shape[0]) - 1) {
      return 0;
    }
    return this->get(u32(ix));
  }

  __hd auto get_interp(f32 x) const -> T {
    // caculate intgral of linear function in [x-0.5, x+0.5]
    const auto nx = _shape[0];
    const auto x0 = i32(math::floorf(x - 0.5f));
    const auto x1 = x0 + 1;

    if (x1 < 0 || x0 > nx - 1) {
      return 0;
    }

    if (x1 == 0) {
      const auto p = x + 0.5f;  // line[0, x+0.5]
      const auto t = (*this)[0];
      return p * t;
    }

    if (x0 == nx - 1) {
      const auto p = f32(nx) - (x - 0.5f);  // line[x-0.5 , nx]
      const auto t = (*this)[nx - 1];
      return p * t;
    }

    const auto t0 = (*this)[u32(x0)];
    const auto t1 = (*this)[u32(x1)];
    const auto p = x - 0.5f - f32(x0);
    const auto t = (1 - p) * t0 + p * t1;
    return t;
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
  u32 _shape[NDIM] = {};
  u32 _strides[NDIM] = {};

 public:
  __hd NdView() noexcept : _data{nullptr}, _shape{0, 0}, _strides{0, 0} {}

  __hd NdView(T* data, const u32 (&shape)[NDIM], const u32 (&strides)[NDIM])
      : _data{data}, _shape{shape[0], shape[1]}, _strides{strides[0], strides[1]} {}

  __hd NdView(T* data, const u32 (&shape)[NDIM]) : _data{data}, _shape{shape[0], shape[1]}, _strides{shape[1], 1} {}

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

  __hd auto operator[](u32 y) const -> NdView<T, 1> {
    T* data = _data + y * _strides[0];
    const u32 shape[] = {_shape[1]};
    const u32 strides[] = {_strides[1]};
    return {data, shape, strides};
  }

  __hd auto operator[](const u32 (&idx)[NDIM]) const -> const T& {
    const auto offset = idx[0] * _strides[0] + idx[1] * _strides[1];
    return _data[offset];
  }

  __hd auto operator[](const u32 (&idx)[NDIM]) -> T& {
    const auto offset = idx[0] * _strides[0] + idx[1] * _strides[1];
    return _data[offset];
  }

 public:
  __hd auto transpose() const -> NdView {
    const u32 shape[] = {_shape[1], _shape[0]};
    const u32 strides[] = {_strides[1], _strides[0]};
    return {_data, shape, strides};
  }

  __hd auto contains(const u32 (&idx)[NDIM]) const -> bool {
    return idx[0] < _shape[0] && idx[1] < _shape[1];
  }

  __hd auto get(const u32 (&idx)[NDIM]) const -> T {
    const auto offset = idx[0] * _strides[0] + idx[1] * _strides[1];
    return _data[offset];
  }

  __hd void set(const u32 (&idx)[NDIM], const T& value) {
    const auto offset = idx[0] * _strides[0] + idx[1] * _strides[1];
    _data[offset] = value;
  }

  __hd auto get_interp(const f32 (&pos)[NDIM]) const -> T {
    const auto [fy, fx] = pos;
    const auto [ny, nx] = _shape;

    const auto y0 = i32(math::floorf(fy - 0.5f));
    const auto y1 = y0 + 1;

    if (y1 < 0 || y0 > ny - 1) {
      return 0;
    }

    if (y1 == 0) {
      const auto p = fy + 0.5f;  // line[0, fy+0.5]
      const auto t = (*this)[0].get_interp(fx);
      return p * t;
    }

    if (y0 == ny - 1) {
      const auto p = f32(ny) - (fy - 0.5f);  // line[fy-0.5 , ny]
      const auto t = (*this)[ny - 1].get_interp(fx);
      return p * t;
    }

    const auto t0 = (*this)[u32(y0)].get_interp(fx);
    const auto t1 = (*this)[u32(y1)].get_interp(fx);
    const auto p = fy - 0.5f - f32(y0);
    const auto t = (1 - p) * t0 + p * t1;
    return t;
  }

 public:
#ifndef __CUDACC__
  auto is_contiguous() const -> bool {
    return _strides[0] == _shape[1] && _strides[1] == 1;
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
  u32 _shape[NDIM] = {};
  u32 _strides[NDIM] = {};

 public:
  __hd NdView() noexcept : _data{nullptr}, _shape{0, 0, 0}, _strides{0, 0, 0} {}

  __hd NdView(T* data, const u32 (&shape)[NDIM], const u32 (&strides)[NDIM])
      : _data{data}, _shape{shape[0], shape[1], shape[2]}, _strides{strides[0], strides[1], strides[2]} {}

  __hd NdView(T* data, const u32 (&shape)[NDIM])
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

  __hd auto operator[](u32 z) const -> NdView<T, 2> {
    T* data = _data + z * _strides[0];
    const u32 shape[] = {_shape[1], _shape[2]};
    const u32 strides[] = {_strides[1], _strides[2]};
    return {data, shape, strides};
  }

  __hd auto operator[](const u32 (&idx)[NDIM]) const -> const T& {
    const auto offset = idx[0] * _strides[0] + idx[1] * _strides[1] + idx[2] * _strides[2];
    return _data[offset];
  }

  __hd auto operator[](const u32 (&idx)[NDIM]) -> T& {
    const auto offset = idx[0] * _strides[0] + idx[1] * _strides[1] + idx[2] * _strides[2];
    return _data[offset];
  }

 public:
  __hd auto contains(const u32 (&idx)[NDIM]) const -> bool {
    return idx[0] < _shape[0] && idx[1] < _shape[1] && idx[2] < _shape[2];
  }

  __hd auto get(const u32 (&idx)[NDIM]) const -> T {
    const auto offset = idx[0] * _strides[0] + idx[1] * _strides[1] + idx[2] * _strides[2];
    return _data[offset];
  }

  __hd auto set(const u32 (&idx)[NDIM], const T& value) {
    const auto offset = idx[0] * _strides[0] + idx[1] * _strides[1] + idx[2] * _strides[2];
    _data[offset] = value;
  }

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
  u32 _shape[NDIM] = {};
  u32 _strides[NDIM] = {};

 public:
  __hd NdView() noexcept : _data{nullptr}, _shape{0, 0, 0, 0}, _strides{0, 0, 0, 0} {}

  __hd NdView(T* data, const u32 (&shape)[NDIM], const u32 (&strides)[NDIM])
      : _data{data}
      , _shape{shape[0], shape[1], shape[2], shape[3]}
      , _strides{strides[0], strides[1], strides[2], strides[3]} {}

  __hd NdView(T* data, const u32 (&shape)[NDIM])
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
  __hd auto operator[](u32 i) const -> NdView<T, 3> {
    T* data = _data + i * _strides[0];
    const u32 shape[] = {_shape[1], _shape[2], _shape[3]};
    const u32 strides[] = {_strides[1], _strides[2], _strides[3]};
    return {data, shape, strides};
  }

  __hd auto operator[](const u32 (&idx)[NDIM]) const -> const T& {
    const auto offset = idx[0] * _strides[0] + idx[1] * _strides[1] + idx[2] * _strides[2] + idx[3] * _strides[3];
    return _data[offset];
  }

  __hd auto operator[](const u32 (&idx)[NDIM]) -> T& {
    const auto offset = idx[0] * _strides[0] + idx[1] * _strides[1] + idx[2] * _strides[2] + idx[3] * _strides[3];
    return _data[offset];
  }

 public:
  __hd auto contains(const u32 (&idx)[NDIM]) const -> bool {
    return idx[0] < _shape[0] && idx[1] < _shape[1] && idx[2] < _shape[2] && idx[3] < _shape[3];
  }

  __hd auto get(const u32 (&idx)[NDIM]) const -> T {
    const auto offset = idx[0] * _strides[0] + idx[1] * _strides[1] + idx[2] * _strides[2] + idx[3] * _strides[3];
    return _data[offset];
  }

  __hd auto set(const u32 (&idx)[NDIM], const T& value) {
    const auto offset = idx[0] * _strides[0] + idx[1] * _strides[1] + idx[2] * _strides[2] + idx[3] * _strides[3];
    _data[offset] = value;
  }
};

template <class T, u32 A, u32 B>
auto reshape(NdView<T, A> view, const u32 (&shape)[B]) -> NdView<T, B> {
  auto res = NdView<T, B>{view.data(), shape};
  if (shape[0] == 0) {
    res._shape[0] = 1;                           // set to 1 temporarily
    res._shape[0] = view.numel() / res.numel();  // adjust shape[0]
  }
  return res;
}

}  // namespace sfc::math
