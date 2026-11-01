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
  u32 _shape[1] = {};
  u32 _strides[1] = {};

 public:
  __hd NdSlice() noexcept : _data{nullptr}, _shape{0}, _strides{0} {}

  __hd NdSlice(T* data, const u32 (&shape)[1], const u32 (&strides)[1])
      : _data{data}, _shape{shape[0]}, _strides{strides[0]} {}

  static auto with_shape(T* data, const u32 (&shape)[1]) -> NdSlice {
    const u32 strides[1] = {1};
    return NdSlice{data, shape, strides};
  }

 public:
  __hd auto len() const -> u32 {
    return _shape[0];
  }

  __hd auto ptr() const -> T* {
    return _data;
  }

  __hd auto numel() const -> u32 {
    return _shape[0];
  }

  __hd auto shape(u32 idim) const -> u32 {
    return idim < NDIM ? _shape[idim] : 0;
  }

 public:
  __hd auto contains(u32 i) const -> bool {
    return i < _shape[0];
  }

  __hd auto operator[](u32 i) const -> const T& {
    const auto offset = i * _strides[0];
    return _data[offset];
  }

  __hd auto operator[](u32 i) -> T& {
    const auto offset = i * _strides[0];
    return _data[offset];
  }

  __hd auto operator()(u32 i) const -> const T& {
    const auto offset = i * _strides[0];
    return _data[offset];
  }

  __hd auto operator()(u32 i) -> T& {
    const auto offset = i * _strides[0];
    return _data[offset];
  }

 public:
  auto is_contiguous() const -> bool {
    return _strides[0] == 1;
  }

  void for_each(auto&& f) const {
    for (auto i = 0U; i < _shape[0]; ++i) {
      const auto& e = (*this)[i];
      f(i, e);
    }
  }

  void for_each_mut(auto&& f) {
    for (auto i = 0U; i < _shape[0]; ++i) {
      auto& e = (*this)[i];
      f(i, e);
    }
  }

 public:
  void fmt(auto& f) const {
    auto imp = f.debug_list();
    this->for_each([&]([[maybe_unused]] u32 i, const T& e) { imp.entry(e); });
    return;
  }

  void fill_with(auto&& f) {
    this->for_each_mut([&](u32 i, T& e) {
      const auto val = f(i);
      e = val;
    });
  }
};

template <class T>
struct NdSlice<T, 2> {
  static constexpr u32 NDIM = 2U;
  using Item = T;

  T* _data = nullptr;
  u32 _shape[2] = {};
  u32 _strides[2] = {};

 public:
  __hd NdSlice() noexcept : _data{nullptr}, _shape{0, 0}, _strides{0, 0} {}

  __hd NdSlice(T* data, const u32 (&shape)[2], const u32 (&strides)[2])
      : _data{data}, _shape{shape[0], shape[1]}, _strides{strides[0], strides[1]} {}

  static auto with_shape(T* data, const u32 (&shape)[2]) -> NdSlice {
    const u32 strides[2] = {shape[1], 1};
    return NdSlice{data, shape, strides};
  }

 public:
  __hd auto len() const -> u32 {
    return _shape[0];
  }

  __hd auto ptr() const -> T* {
    return _data;
  }

  __hd auto numel() const -> u32 {
    return _shape[0] * _shape[1];
  }

  __hd auto shape(u32 dim) const -> u32 {
    return dim < NDIM ? _shape[dim] : 0;
  }

 public:
  __hd auto operator[](u32 i) const -> NdSlice<T, 1> {
    const auto data = _data + i * _strides[0];
    const u32 shape[] = {_shape[1]};
    const u32 strides[] = {_strides[1]};
    return NdSlice<T, 1>{data, shape, strides};
  }

  __hd auto contains(u32 i, u32 j) const -> bool {
    return i < _shape[0] && j < _shape[1];
  }

  __hd auto operator()(u32 i, u32 j) const -> const T& {
    const auto offset = i * _strides[0] + j * _strides[1];
    return _data[offset];
  }

  __hd auto operator()(u32 i, u32 j) -> T& {
    const auto offset = i * _strides[0] + j * _strides[1];
    return _data[offset];
  }

  __hd auto get(u32 i, u32 j) const -> T {
    const auto offset = i * _strides[0] + j * _strides[1];
    return _data[offset];
  }

  __hd void set(u32 i, u32 j, T val) {
    const auto offset = i * _strides[0] + j * _strides[1];
    _data[offset] = val;
  }

 public:
  auto is_contiguous() const -> bool {
    const auto s = NdSlice::with_shape(_data, _shape);
    return _strides[0] == s._strides[0] && _strides[1] == s._strides[1];
  }

  auto transpose() const -> NdSlice {
    const u32 shape[] = {_shape[1], _shape[0]};
    const u32 strides[] = {_strides[1], _strides[0]};
    return {_data, shape, strides};
  }

  template <u32 I = 1, u32 J = 0>
  auto permute() const -> NdSlice {
    static_assert(I < 2 && J < 2);
    static_assert(I != J);

    const u32 shape[] = {_shape[I], _shape[J]};
    const u32 strides[] = {_strides[I], _strides[J]};
    return {_data, shape, strides};
  }

  void for_each(auto&& f) const {
    for (u32 i = 0U; i < _shape[0]; ++i) {
      const auto row = (*this)[i];
      for (u32 j = 0U; j < _shape[1]; ++j) {
        const auto& e = row[j];
        f(i, j, e);
      }
    }
  }

  void for_each_mut(auto&& f) {
    for (u32 i = 0U; i < _shape[0]; ++i) {
      auto row = (*this)[i];
      for (u32 j = 0U; j < _shape[1]; ++j) {
        auto& e = row[j];
        f(i, j, e);
      }
    }
  }

 public:
  auto fmt(auto& f) const -> void {
    f.set_max_depth(f.depth() + 1);
    auto imp = f.debug_list();
    for (u32 i = 0U; i < _shape[0]; ++i) {
      imp.entry((*this)[i]);
    }
  }

  void fill_with(auto&& f) {
    this->for_each_mut([&](u32 i, u32 j, T& e) {
      const auto val = f(i, j);
      e = val;
    });
  }
};

template <class T>
struct NdSlice<T, 3> {
  static constexpr u32 NDIM = 3U;
  using Item = T;

  T* _data = nullptr;
  u32 _shape[3] = {};
  u32 _strides[3] = {};

 public:
  __hd NdSlice() noexcept : _data{nullptr}, _shape{0, 0, 0}, _strides{0, 0, 0} {}

  __hd NdSlice(T* data, const u32 (&shape)[3], const u32 (&strides)[3])
      : _data{data}, _shape{shape[0], shape[1], shape[2]}, _strides{strides[0], strides[1], strides[2]} {}

  static auto with_shape(T* data, const u32 (&shape)[3]) -> NdSlice {
    const u32 strides[3] = {shape[1] * shape[2], shape[2], 1};
    return NdSlice{data, shape, strides};
  }

 public:
  __hd auto len() const -> u32 {
    return _shape[0];
  }

  __hd auto ptr() const -> T* {
    return _data;
  }

  __hd auto numel() const -> u32 {
    return _shape[0] * _shape[1] * _shape[2];
  }

  __hd auto shape(u32 dim) const -> u32 {
    return dim < NDIM ? _shape[dim] : 0;
  }

 public:
  __hd auto operator[](u32 i) const -> NdSlice<T, 2> {
    const auto data = _data + i * _strides[0];
    const u32 shape[] = {_shape[1], _shape[2]};
    const u32 strides[] = {_strides[1], _strides[2]};
    return NdSlice<T, 2>{data, shape, strides};
  }

  __hd auto contains(u32 i, u32 j, u32 k) const -> bool {
    return i < _shape[0] && j < _shape[1] && k < _shape[2];
  }

  __hd auto operator()(u32 i, u32 j, u32 k) const -> const T& {
    const auto offset = i * _strides[0] + j * _strides[1] + k * _strides[2];
    return _data[offset];
  }

  __hd auto operator()(u32 i, u32 j, u32 k) -> T& {
    const auto offset = i * _strides[0] + j * _strides[1] + k * _strides[2];
    return _data[offset];
  }

 public:
  auto is_contiguous() const -> bool {
    const auto s = NdSlice::with_shape(_data, _shape);
    return _strides[0] == s._strides[0] && _strides[1] == s._strides[1] && _strides[2] == s._strides[2];
  }

  template <u32 I, u32 J, u32 K>
  auto permute() const -> NdSlice<T, 3> {
    static_assert(I < 3 && J < 3 && K < 3);
    static_assert(I != J && I != K && J != K);

    const u32 shape[] = {_shape[I], _shape[J], _shape[K]};
    const u32 strides[] = {_strides[I], _strides[J], _strides[K]};
    return NdSlice<T, 3>{_data, shape, strides};
  }

  void for_each(auto&& f) const {
    for (auto i = 0U; i < _shape[0]; ++i) {
      const auto mat = (*this)[i];
      for (auto j = 0U; j < _shape[1]; ++j) {
        const auto col = mat[j];
        for (auto k = 0U; k < _shape[2]; ++k) {
          const auto& e = col[k];
          f(i, j, k, e);
        }
      }
    }
  }

  void for_each_mut(auto&& f) {
    for (auto i = 0U; i < _shape[0]; ++i) {
      auto mat = (*this)[i];
      for (auto j = 0U; j < _shape[1]; ++j) {
        auto col = mat[j];
        for (auto k = 0U; k < _shape[2]; ++k) {
          auto& e = col[k];
          f(i, j, k, e);
        }
      }
    }
  }

 public:
  void fmt(auto& f) const {
    auto imp = f.debug_list();
    for (u32 i = 0U; i < _shape[0]; ++i) {
      imp.entry((*this)[i]);
    }
  }

  void fill_with(auto&& f) {
    this->for_each_mut([&](u32 i, u32 j, u32 k, T& e) {
      const auto val = f(i, j, k);
      e = val;
    });
  }
};

template <class T>
struct NdSlice<T, 4> {
  static constexpr u32 NDIM = 4U;
  using Item = T;

  T* _data = nullptr;
  u32 _shape[4] = {};
  u32 _strides[4] = {};

 public:
  __hd NdSlice() noexcept : _data{nullptr}, _shape{0, 0, 0, 0}, _strides{0, 0, 0, 0} {}

  __hd NdSlice(T* data, const u32 (&shape)[4], const u32 (&strides)[4])
      : _data{data}
      , _shape{shape[0], shape[1], shape[2], shape[3]}
      , _strides{strides[0], strides[1], strides[2], strides[3]} {}

  static auto with_shape(T* data, const u32 (&shape)[4]) -> NdSlice {
    const u32 strides[4] = {shape[1] * shape[2] * shape[3], shape[2] * shape[3], shape[3], 1};
    return NdSlice{data, shape, strides};
  }

 public:
  __hd auto len() const -> u32 {
    return _shape[0];
  }

  __hd auto ptr() const -> T* {
    return _data;
  }

  __hd auto numel() const -> u32 {
    return _shape[0] * _shape[1] * _shape[2] * _shape[3];
  }

  __hd auto shape(u32 idim) const -> u32 {
    return idim < NDIM ? _shape[idim] : 0;
  }

 public:
  __hd auto operator[](u32 i) const -> NdSlice<T, 3> {
    const auto data = _data + i * _strides[0];
    const u32 shape[] = {_shape[1], _shape[2], _shape[3]};
    const u32 strides[] = {_strides[1], _strides[2], _strides[3]};
    return NdSlice<T, 3>{data, shape, strides};
  }

  __hd auto contains(u32 i, u32 j, u32 k, u32 l) const -> bool {
    return i < _shape[0] && j < _shape[1] && k < _shape[2] && l < _shape[3];
  }

  __hd auto operator()(u32 i, u32 j, u32 k, u32 l) const -> const T& {
    const auto offset = i * _strides[0] + j * _strides[1] + k * _strides[2] + l * _strides[3];
    return _data[offset];
  }

  __hd auto operator()(u32 i, u32 j, u32 k, u32 l) -> T& {
    const auto offset = i * _strides[0] + j * _strides[1] + k * _strides[2] + l * _strides[3];
    return _data[offset];
  }

 public:
  auto is_contiguous() const -> bool {
    const auto s = NdSlice::with_shape(_data, _shape);
    return _strides[0] == s._strides[0] && _strides[1] == s._strides[1] && _strides[2] == s._strides[2] &&
           _strides[3] == s._strides[3];
  }

  template <u32 I, u32 J, u32 K, u32 L>
  auto permute() -> NdSlice<T, 4> {
    static_assert(I < 4 && J < 4 && K < 4 && L < 4);
    static_assert(I != J && I != K && I != L && J != K && J != L && K != L);

    const u32 shape[4] = {_shape[I], _shape[J], _shape[K], _shape[L]};
    const u32 strides[4] = {_strides[I], _strides[J], _strides[K], _strides[L]};
    return NdSlice<T, 4>{_data, shape, strides};
  }

  void for_each(auto&& func) const {
    for (auto i = 0U; i < _shape[0]; ++i) {
      const auto vol = (*this)[i];
      vol.for_each([&](u32 j, u32 k, u32 l, const T& e) { func(i, j, k, l, e); });
    }
  }

  void for_each_mut(auto&& func) {
    for (auto i = 0U; i < _shape[0]; ++i) {
      auto vol = (*this)[i];
      vol.for_each_mut([&](u32 j, u32 k, u32 l, T& e) { func(i, j, k, l, e); });
    }
  }

 public:
  void fmt(auto& f) const {
    auto imp = f.debug_list();
    for (u32 i = 0U; i < _shape[0]; ++i) {
      imp.entry((*this)[i]);
    }
  }

  void fill_with(auto&& f) {
    this->for_each_mut([&](u32 i, u32 j, u32 k, u32 l, T& e) {
      const auto val = f(i, j, k, l);
      e = val;
    });
  }
};

template <class T, u32 A, u32 B>
auto reshape(NdSlice<T, A> src, const u32 (&new_shape)[B]) -> NdSlice<T, B> {
  auto dst = NdSlice<T, B>::with_shape(src._data, new_shape);
  if (!src.is_contiguous() || src.numel() != dst.numel()) {
    return {};
  }
  return dst;
}

}  // namespace sfc::math
