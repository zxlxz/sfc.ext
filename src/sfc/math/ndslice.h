#pragma once

#include "sfc/math/vec.h"

namespace sfc::math {

template <class T, int N = 1>
struct NdSlice;

template <class T>
struct NdSlice<T, 1> {
  static constexpr auto NDIM = 1U;
  using dims_t = math::vec<u32, NDIM>;
  using step_t = math::vec<u32, NDIM>;
  using idxs_t = math::vec<u32, NDIM>;
  T* _data;
  dims_t _dims;
  step_t _step;

 public:
  __hd auto len() const -> u32 {
    return _dims.x;
  }

  __hd auto data() const -> const T* {
    return _data;
  }

  __hd auto shape() const -> const dims_t& {
    return _dims;
  }

  __hd auto strides() const -> const step_t& {
    return _step;
  }

  __hd auto numel() const -> u32 {
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
  __hd auto load(f32 fx) const -> T {
    const auto check_x = fx >= 0 && fx < static_cast<f32>(_dims.x);
    if (!check_x) {
      return 0;
    }

    const auto ux = static_cast<u32>(fx);
    return _data[ux];
  }

  // [0 ... shape]
  __hd auto load_interp(f32 x) const -> T {
    const auto nx = static_cast<i32>(_dims.x);
    const auto x0 = static_cast<i32>(x);
    const auto x1 = x0 + 1;
    if (x0 < 0) {
      return x1 == 0 ? _data[x1] : 0;
    }
    if (x1 >= nx) {
      return x0 == nx - 1 ? _data[x0] : 0;
    }

    const auto cx = static_cast<f32>(x0) + 0.5f;
    const auto t0 = _data[static_cast<u32>(x0)];
    const auto t1 = _data[static_cast<u32>(x1)];
    return (x1 - cx) * t0 + (cx - x0) * t1;
  }

  void imap(auto&& f) const {
    for (auto i = 0U; i < _dims.x; ++i) {
      const auto& e = _data[i];
      f(i, e);
    }
  }

  void imap_mut(auto&& f) {
    for (auto i = 0U; i < _dims.x; ++i) {
      auto& e = _data[i];
      f(i, e);
    }
  }

  void fmt(auto& f) const {
    this->imap([&](u32 i, const T& e) {
      if (i > 0) f.write_str(", ");
      f.write_val(e);
    });
  }
};

template <class T>
struct NdSlice<T, 2> {
  static constexpr auto NDIM = 2U;
  using dims_t = math::vec<u32, NDIM>;
  using step_t = math::vec<u32, NDIM>;
  using idxs_t = math::vec<u32, NDIM>;
  T* _data;
  dims_t _dims;
  step_t _step;

 public:
  __hd auto len() const -> u32 {
    return _dims.y;
  }

  __hd auto data() const -> const T* {
    return _data;
  }

  __hd auto shape() const -> const dims_t& {
    return _dims;
  }

  __hd auto strides() const -> const step_t& {
    return _step;
  }

  __hd auto numel() const -> u32 {
    return _dims.x * _dims.y;
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
  __hd auto load(vec2f p) const -> T {
    const auto nx = static_cast<i32>(_dims.x);
    const auto ny = static_cast<i32>(_dims.y);
    const auto ix = static_cast<i32>(p.x);
    const auto iy = static_cast<i32>(p.y);
    if (ix < 0 || ix >= nx || iy < 0 || iy >= ny) {
      return 0;
    }

    const auto idx = vec2u{static_cast<u32>(p.x), static_cast<u32>(p.y)};
    const auto val = (*this)[idx];
    return val;
  }

  __hd auto load_interp_2d(vec2f p) const -> T {
    const auto ny = static_cast<i32>(_dims.y);
    const auto y0 = static_cast<i32>(p.y);
    const auto y1 = y0 + 1;

    if (y0 < 0) {
      if (y1 != 0) return 0;
      const auto col = (*this)[0];
      return col.load_interp(p.x);
    }

    if (y1 >= ny) {
      if (y0 != ny - 1) return 0;
      const auto col = (*this)[ny - 1];
      return col.load_interp(p.x);
    }

    const auto cy = static_cast<f32>(y0) + 0.5f;
    const auto col0 = (*this)[static_cast<u32>(y0)];
    const auto col1 = (*this)[static_cast<u32>(y1)];
    const auto val0 = col0.load_interp(p.x);
    const auto val1 = col1.load_interp(p.x);
    return (y1 - cy) * val0 + (cy - y0) * val1;
  }

  void imap(auto&& f) const {
    for (auto j = 0U; j < _dims.y; ++j) {
      const auto col = (*this)[j];
      for (auto i = 0U; i < _dims.x; ++i) {
        const auto& e = col[i];
        f(i, j, e);
      }
    }
  }

  void imap_mut(auto&& f) {
    for (auto j = 0U; j < _dims.y; ++j) {
      auto col = (*this)[j];
      for (auto i = 0U; i < _dims.x; ++i) {
        auto& e = col[i];
        f(i, j, e);
      }
    }
  }

  void fmt(auto& f) const {
    for (auto j = 0U; j < _dims.y; ++j) {
      const auto col = (*this)[j];
      f.write_fmt("{}\n", col);
    }
  }
};

template <class T>
struct NdSlice<T, 3> {
  static constexpr auto NDIM = 3;
  using dims_t = math::vec<u32, NDIM>;
  using step_t = math::vec<u32, NDIM>;
  using idxs_t = math::vec<u32, NDIM>;
  T* _data;
  dims_t _dims;
  step_t _step;

 public:
  __hd auto len() const -> u32 {
    return _dims.z;
  }

  __hd auto data() const -> const T* {
    return _data;
  }

  __hd auto shape() const -> const dims_t& {
    return _dims;
  }

  __hd auto strides() const -> const step_t& {
    return _step;
  }

  __hd auto numel() const -> u32 {
    return _dims.x * _dims.y * _dims.z;
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
  void imap(auto&& f) const {
    for (auto k = 0U; k < _dims.z; ++k) {
      const auto mat = (*this)[k];
      for (auto j = 0U; j < _dims.y; ++j) {
        const auto row = mat[j];
        for (auto i = 0U; i < _dims.x; ++i) {
          const auto& e = row[i];
          f(i, j, k, e);
        }
      }
    }
  }

  void imap_mut(auto&& f) {
    for (auto k = 0U; k < _dims.z; ++k) {
      auto mat = (*this)[k];
      for (auto j = 0U; j < _dims.y; ++j) {
        auto row = mat[j];
        for (auto i = 0U; i < _dims.x; ++i) {
          auto& e = row[i];
          f(i, j, k, e);
        }
      }
    }
  }
};

template <class T>
struct NdSlice<T, 4> {
  static constexpr auto NDIM = 4;
  using dims_t = math::vec<u32, NDIM>;
  using step_t = math::vec<u32, NDIM>;
  using idxs_t = math::vec<u32, NDIM>;
  T* _data;
  dims_t _dims;
  step_t _step;

 public:
  __hd auto len() const -> u32 {
    return _dims.w;
  }

  __hd auto data() const -> const T* {
    return _data;
  }

  __hd auto shape() const -> const dims_t& {
    return _dims;
  }

  __hd auto strides() const -> const step_t& {
    return _step;
  }

  __hd auto numel() const -> u32 {
    return _dims.x * _dims.y * _dims.z * _dims.w;
  }

  __hd auto operator[](u32 idx) const -> NdSlice<T, NDIM - 1> {
    const auto p = _data + idx * _step.w;
    return {p, {_dims.x, _dims.y, _dims.z}, {_step.x, _step.y, _step.z}};
  }

  __hd auto in_bounds(idxs_t idxs) const -> bool {
    return idxs.x < _dims.x && idxs.y < _dims.y && idxs.z < _dims.z && idxs.w < _dims.w;
  }

  __hd auto operator[](idxs_t idxs) const -> const T& {
    return _data[idxs.x + idxs.y * _step.y + idxs.z * _step.z + idxs.w * _step.w];
  }

  __hd auto operator[](idxs_t idxs) -> T& {
    return _data[idxs.x + idxs.y * _step.y + idxs.z * _step.z + idxs.w * _step.w];
  }

 public:
  void imap(auto&& f) const {
    for (auto w = 0U; w < _dims.w; ++w) {
      const auto img = (*this)[w];
      img.imap([&](u32 i, u32 j, u32 k, const T& e) { f(i, j, k, w, e); });
    }
  }

  void imap_mut(auto&& f) {
    for (auto w = 0U; w < _dims.w; ++w) {
      auto img = (*this)[w];
      img.imap_mut([&](u32 i, u32 j, u32 k, T& e) { f(i, j, k, w, e); });
    }
  }
};

}  // namespace sfc::math
