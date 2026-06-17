#pragma once

#include "sfc/math/vec.h"

namespace sfc::math {

template <class T, int N = 1>
struct NdSlice;

template <class T>
struct NdSlice<T, 1> {
  static constexpr auto NDIM = 1U;
  using item_t = T;
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

  __hd auto operator[](u32 idx) const -> const T& {
    return _data[idx * _step.x];
  }

  __hd auto operator[](u32 idx) -> T& {
    return _data[idx * _step.x];
  }

  __hd auto operator[](idxs_t idxs) const -> const T& {
    return _data[idxs.x * _step.x];
  }

  __hd auto operator[](idxs_t idxs) -> T& {
    return _data[idxs.x * _step.x];
  }

  __hd auto is_contiguous() const -> bool {
    return _step.x == 1;
  }

 public:
  void imap(auto&& f) const {
    for (auto i = 0U; i < _dims.x; ++i) {
      const auto& e = _data[i * _step.x];
      f(i, e);
    }
  }

  void imap_mut(auto&& f) {
    for (auto i = 0U; i < _dims.x; ++i) {
      auto& e = _data[i * _step.x];
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
  using item_t = T;
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
    return NdSlice<T, NDIM - 1>{p, {_dims.x}, {_step.x}};
  }

  __hd auto operator[](idxs_t idx) const -> const T& {
    return _data[idx.x * _step.x + idx.y * _step.y];
  }

  __hd auto operator[](idxs_t idx) -> T& {
    return _data[idx.x * _step.x + idx.y * _step.y];
  }

 public:
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
  using item_t = T;
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

  __hd auto operator[](idxs_t idxs) const -> const T& {
    return _data[idxs.x * _step.x + idxs.y * _step.y + idxs.z * _step.z];
  }

  __hd auto operator[](idxs_t idxs) -> T& {
    return _data[idxs.x * _step.x + idxs.y * _step.y + idxs.z * _step.z];
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
  using item_t = T;
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

  __hd auto operator[](idxs_t idxs) const -> const T& {
    return _data[idxs.x * _step.x + idxs.y * _step.y + idxs.z * _step.z + idxs.w * _step.w];
  }

  __hd auto operator[](idxs_t idxs) -> T& {
    return _data[idxs.x * _step.x + idxs.y * _step.y + idxs.z * _step.z + idxs.w * _step.w];
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
