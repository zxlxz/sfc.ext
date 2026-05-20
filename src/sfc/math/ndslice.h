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
      const auto row = (*this)[j];
      f.write_fmt("{}\n", row);
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
