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

  __hd static auto from_raw(T* data, const u32 (&shape)[NDIM]) -> NdView {
    u32 strides[NDIM] = {1};
    return NdView{data, shape, strides};
  }

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
      _data[i] = f(i);
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

template <class T, u32 N>
struct NdView {
  static constexpr u32 NDIM = N;
  using Item = T;
  T* _data = nullptr;
  u32 _shape[NDIM] = {};
  u32 _strides[NDIM] = {};

 public:
  __hd NdView() noexcept : _data{nullptr}, _shape{0, 0}, _strides{0, 0} {}

  template <u32... I>
  __hd NdView(T* data, const auto& shape, const auto& strides, ops::IdxSeq<I...>)
      : _data{data}, _shape{shape[I]...}, _strides{strides[I]...} {}

  __hd NdView(T* data, const u32 (&shape)[NDIM], const u32 (&strides)[NDIM])
      : NdView{data, shape, strides, ops::index_seq<NDIM>()} {}

  __hd static auto from_raw(T* data, const u32 (&shape)[NDIM]) -> NdView {
    u32 strides[NDIM] = {};
    strides[NDIM - 1] = 1;
    for (auto i = NDIM - 1; i > 0; --i) {
      strides[i - 1] = shape[i] * strides[i];
    }
    return NdView{data, shape, strides};
  }

 public:
  __hd auto len() const -> u32 {
    return _shape[0];
  }

  __hd auto data() const -> T* {
    return _data;
  }

  __hd auto operator[](u32 x) const -> NdView<T, NDIM - 1> {
    const auto data = _data + x * _strides[0];
    const auto& shape = *ptr::cast<const u32[NDIM - 1]>(_shape + 1);
    const auto& strides = *ptr::cast<const u32[NDIM - 1]>(_strides + 1);
    const auto ret = NdView<T, NDIM - 1>{data, shape, strides};
    return ret;
  }

  __hd auto operator[](const u32 (&idx)[NDIM]) const -> const T& {
    const auto offset = ops::index_seq<NDIM>::dot(idx, _strides);
    return _data[offset];
  }

  __hd auto operator[](const u32 (&idx)[NDIM]) -> T& {
    const auto offset = ops::index_seq<NDIM>::dot(idx, _strides);
    return _data[offset];
  }

 public:
  __hd auto offset(const u32 (&idx)[NDIM]) const -> u32 {
    return ops::index_seq<NDIM>::dot(idx, _strides);
  }

  __hd auto contains(const u32 (&idx)[NDIM]) const -> bool {
    return ops::index_seq<NDIM>::lt(idx, _shape);
  }

  __hd auto get(const u32 (&idx)[NDIM]) const -> T {
    const auto offset = ops::index_seq<NDIM>::dot(idx, _strides);
    return _data[offset];
  }

  __hd void set(const u32 (&idx)[NDIM], const T& value) {
    const auto offset = ops::index_seq<NDIM>::dot(idx, _strides);
    _data[offset] = value;
  }

 public:
#ifndef __CUDACC__
  __hd auto numel() const -> u32 {
    return ops::index_seq<NDIM>::prod(_shape);
  }

  auto is_contiguous() const -> bool {
    const auto tmp = NdView::from_raw(_data, _shape);
    return ops::index_seq<NDIM>::eq(tmp._strides, _strides);
  }

  void for_each(auto&& f) const {
    for (auto i = 0U; i < _shape[0]; ++i) {
      const auto row = (*this)[i];
      row.for_each([&](const T& e, auto... j) { f(e, i, j...); });
    }
  }

  void for_each_mut(auto&& f) {
    for (auto i = 0U; i < _shape[0]; ++i) {
      auto row = (*this)[i];
      row.for_each_mut([&](T& e, auto... j) { f(e, i, j...); });
    }
  }

  void fill_with(auto&& f) {
    for (u32 i = 0U; i < _shape[0]; ++i) {
      auto row = (*this)[i];
      row.fill_with([&](auto... j) { return f(i, j...); });
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
