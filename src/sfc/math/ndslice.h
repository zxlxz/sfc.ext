#pragma once

#include "sfc/math/vec.h"

namespace sfc::math {

template <class T, int N = 1>
struct NdSlice;

template <class T>
struct NdSlice<T, 1> {
  static constexpr auto NDIM = 1U;
  using dims_t = math::vec<u32, 1>;
  using step_t = math::vec<u32, 1>;
  using idxs_t = math::vec<u32, 1>;

  T* _data;
  dims_t _dims;
  step_t _step;

 public:
  __hd auto shape() const -> const dims_t& {
    return _dims;
  }

  __hd auto strides() const -> const step_t& {
    return _step;
  }

  __hd auto numel() const -> u32 {
    return _dims.x;
  }

  __hd auto len() const -> u32 {
    return _dims.x;
  }

  __hd auto in_bounds(idxs_t idxs) const -> bool {
    return idxs.x < _dims.x;
  }

  __hd auto operator[](u32 idx) const -> const T& {
    return _data[idx];
  }

  __hd auto operator[](u32 idx) -> T& {
    return _data[idx];
  }

  __hd auto operator[](idxs_t idxs) const -> const T& {
    return _data[idxs.x];
  }

  __hd auto operator[](idxs_t idxs) -> T& {
    return _data[idxs.x];
  }

 public:
  void imap(auto&& f) {
    for (auto i = 0U; i < _dims.x; ++i) {
      auto& e = _data[i];
      f(i, e);
    }
  }

  // [0 ... shape]
  __hd auto load_interp(f32 x) const -> T {
    if (x < 0 || x >= _dims.x) return 0;

    const auto fx = x - 0.5f;
    const auto x0 = static_cast<i32>(fx);
    const auto x1 = x0 + 1;
    if (x0 < 0) return _data[0];
    if (x1 >= _dims.x) return _data[_dims.x - 1];

    const auto p1 = fx - static_cast<f32>(x0);
    const auto t0 = _data[x0];
    const auto t1 = _data[x1];
    return t0 * (1.0f - p1) + t1 * p1;
  }

  void fill(T val) {
    for (auto i = 0U; i < _dims.x; ++i) {
      _data[i] = val;
    }
  }

  void fmt(auto& f) const {
    for (auto i = 0U; i < _dims.x; ++i) {
      if (i > 0) f.write_str(", ");
      f.write_val(_data[i]);
    }
  }
};

template <class T>
struct NdSlice<T, 2> {
  static constexpr auto NDIM = 2U;
  using dims_t = vec2u;
  using step_t = vec2u;
  using idxs_t = vec2u;
  T* _data;
  dims_t _dims;
  step_t _step;

 public:
  __hd auto shape() const -> const dims_t& {
    return _dims;
  }

  __hd auto strides() const -> const step_t& {
    return _step;
  }

  __hd auto numel() const -> u32 {
    return _dims.x * _dims.y;
  }

  __hd auto len() const -> u32 {
    return _dims.y;
  }

  __hd auto operator[](u32 idx) const -> NdSlice<T, NDIM - 1> {
    const auto p = _data + idx * _step.y;
    return {p, {_dims.x}, {_step.x}};
  }

  __hd auto in_bounds(idxs_t idx) const -> bool {
    return idx.x < _dims.x && idx.y < _dims.y;
  }

  __hd auto operator[](idxs_t idx) const -> const T& {
    return _data[idx.x + idx.y * _step.y];
  }

  __hd auto operator[](idxs_t idx) -> T& {
    return _data[idx.x + idx.y * _step.y];
  }

 public:
  __hd auto load_interp_2d(vec2f p) const -> T {
    const auto [x, y] = p;
    if (y < 0 || y >= _dims.y) return 0;

    const auto fy = y - 0.5f;
    const auto y0 = static_cast<i32>(y);
    const auto y1 = y0 + 1;
    if (y0 < 0) {
      return (*this)[0].load_interp(x);
    }
    if (y1 >= _dims.y) {
      return (*this)[_dims.y - 1].load_interp(x);
    }

    const auto p1 = fy - static_cast<f32>(y0);
    const auto t0 = (*this)[y0].load_interp(x);
    const auto t1 = (*this)[y1].load_interp(x);
    return t0 * (1.0f - p1) + t1 * p1;
  }

  void imap(auto&& f) {
    for (auto j = 0U; j < _dims.y; ++j) {
      auto col = (*this)[j];
      col.imap([&](u32 i, T& e) { f(i, j, e); });
    }
  }

  void fill(T val) {
    for (auto k = 0U; k < _dims.y; ++k) {
      (*this)[k].fill(val);
    }
  }

  void fmt(auto& f) const {
    for (auto j = 0U; j < _dims.y; ++j) {
      const auto row = (*this)[j];
      f.write_fmt("{}\n", row);
    }
  }
};

template <class T>
struct NdSlice<T, 3> {
  static constexpr auto NDIM = 3U;
  using dims_t = vec3u;
  using step_t = vec3u;
  using idxs_t = vec3u;
  T* _data;
  dims_t _dims;
  step_t _step;

 public:
  __hd auto shape() const -> const dims_t& {
    return _dims;
  }

  __hd auto strides() const -> const step_t& {
    return _step;
  }

  __hd auto numel() const -> u32 {
    return _dims.x * _dims.y * _dims.z;
  }

  __hd auto len() const -> u32 {
    return _dims.z;
  }

  __hd auto operator[](u32 idx) const -> NdSlice<T, NDIM - 1> {
    const auto p = _data + idx * _step.z;
    return {p, {_dims.x, _dims.y}, {_step.x, _step.y}};
  }

  __hd auto in_bounds(idxs_t idxs) const -> bool {
    return idxs.x < _dims.x && idxs.y < _dims.y && idxs.z < _dims.z;
  }

  __hd auto operator[](idxs_t idxs) const -> const T& {
    return _data[idxs.x + idxs.y * _step.y + idxs.z * _step.z];
  }

  __hd auto operator[](idxs_t idxs) -> T& {
    return _data[idxs.x + idxs.y * _step.y + idxs.z * _step.z];
  }

 public:
  void imap(auto&& f) {
    for (auto z = 0U; z < _dims.z; ++z) {
      auto img = (*this)[z];
      img.imap([&](u32 i, u32 j, T& e) { f(i, j, z, e); });
    }
  }

  void fill(T val) {
    for (auto k = 0U; k < _dims.z; ++k) {
      (*this)[k].fill(val);
    }
  }

  auto slice(idxs_t idx, dims_t size) -> NdSlice {
    auto p = &(*this)[idx];
    return {p, size, _step};
  }
};

template <class T>
struct NdSlice<T, 4> {
  static constexpr auto NDIM = 4U;
  using dims_t = vec4u;
  using step_t = vec4u;
  using idxs_t = vec4u;
  T* _data;
  dims_t _dims;
  step_t _step;

 public:
  __hd auto shape() const -> const dims_t& {
    return _dims;
  }

  __hd auto strides() const -> const step_t& {
    return _step;
  }

  __hd auto numel() const -> u32 {
    return _dims.x * _dims.y * _dims.z * _dims.w;
  }

  __hd auto len() const -> u32 {
    return _dims.w;
  }

  __hd auto operator[](u32 idx) const -> NdSlice<T, NDIM - 1> {
    const auto p = _data + idx * _step.w;
    return {p, {_dims.x, _dims.y, _dims.z}, {_step.x, _step.y, _step.z}};
  }
};

}  // namespace sfc::math
