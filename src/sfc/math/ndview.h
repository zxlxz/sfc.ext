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

  static auto with_shape(T* data, const u32 (&shape)[1]) -> NdView {
    const u32 strides[1] = {1};
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
      auto& e = _data[i * _strides[0]];
      f(i, e);
    }
  }

  void fill_with(auto&& f) {
    this->for_each_mut([&f](u32 i, T& e) { e = f(i); });
  }

  void fmt(auto& f) const {
    auto imp = f.debug_list();
    for (auto i = 0U; i < _shape[0]; ++i) {
      imp.entry((*this)[i]);
    }
  }
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

  static auto with_shape(T* data, const u32 (&shape)[2]) -> NdView {
    const u32 strides[2] = {shape[1], 1};
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
    return _shape[0] * _shape[1];
  }

 public:
  __hd auto contains(u32 i, u32 j) const -> bool {
    return i < _shape[0] && j < _shape[1];
  }

  __hd auto operator[](u32 i, u32 j) const -> const T& {
    const auto offset = i * _strides[0] + j * _strides[1];
    return _data[offset];
  }

  __hd auto operator[](u32 i, u32 j) -> T& {
    const auto offset = i * _strides[0] + j * _strides[1];
    return _data[offset];
  }

  __hd auto operator[](u32 i) const -> NdView<T, 1> {
    const auto data = _data + i * _strides[0];
    return NdView<T, 1>{data, {_shape[1]}, {_strides[1]}};
  }

 public:
  auto is_contiguous() const -> bool {
    const auto s = NdView::with_shape(_data, _shape);
    return _strides[0] == s._strides[0] && _strides[1] == s._strides[1];
  }

  template <u32 I = 1, u32 J = 0>
  auto transpose() const -> NdView {
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

  void fill_with(auto&& f) {
    this->for_each_mut([&f](u32 i, u32 j, T& e) { e = f(i, j); });
  }

  auto fmt(auto& f) const -> void {
    f.set_max_depth(f.depth() + 1);
    auto imp = f.debug_list();
    for (u32 i = 0U; i < _shape[0]; ++i) {
      imp.entry((*this)[i]);
    }
  }
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

  static auto with_shape(T* data, const u32 (&shape)[3]) -> NdView {
    const u32 strides[3] = {shape[1] * shape[2], shape[2], 1};
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
    return _shape[0] * _shape[1] * _shape[2];
  }

 public:
  __hd auto contains(u32 i, u32 j, u32 k) const -> bool {
    return i < _shape[0] && j < _shape[1] && k < _shape[2];
  }

  __hd auto operator[](u32 i, u32 j, u32 k) const -> const T& {
    const auto offset = i * _strides[0] + j * _strides[1] + k * _strides[2];
    return _data[offset];
  }

  __hd auto operator[](u32 i, u32 j, u32 k) -> T& {
    const auto offset = i * _strides[0] + j * _strides[1] + k * _strides[2];
    return _data[offset];
  }

  __hd auto operator[](u32 i) const -> NdView<T, 2> {
    const auto data = _data + i * _strides[0];
    return NdView<T, 2>{data, {_shape[1], _shape[2]}, {_strides[1], _strides[2]}};
  }

 public:
  auto is_contiguous() const -> bool {
    const auto s = NdView::with_shape(_data, _shape);
    return _strides[0] == s._strides[0] && _strides[1] == s._strides[1] && _strides[2] == s._strides[2];
  }

  template <u32 I, u32 J, u32 K>
  auto transpose() const -> NdView<T, 3> {
    static_assert(I < 3 && J < 3 && K < 3);
    static_assert(I != J && I != K && J != K);

    const u32 shape[] = {_shape[I], _shape[J], _shape[K]};
    const u32 strides[] = {_strides[I], _strides[J], _strides[K]};
    return NdView<T, 3>{_data, shape, strides};
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

  void fill_with(auto&& f) {
    this->for_each_mut([&](u32 i, u32 j, u32 k, T& e) { e = f(i, j, k); });
  }

  void fmt(auto& f) const {
    auto imp = f.debug_list();
    for (u32 i = 0U; i < _shape[0]; ++i) {
      imp.entry((*this)[i]);
    }
  }
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

  static auto with_shape(T* data, const u32 (&shape)[4]) -> NdView {
    const u32 strides[4] = {shape[1] * shape[2] * shape[3], shape[2] * shape[3], shape[3], 1};
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
    return _shape[0] * _shape[1] * _shape[2] * _shape[3];
  }

 public:
  __hd auto contains(u32 i, u32 j, u32 k, u32 l) const -> bool {
    return i < _shape[0] && j < _shape[1] && k < _shape[2] && l < _shape[3];
  }

  __hd auto operator[](u32 i, u32 j, u32 k, u32 l) const -> const T& {
    const auto offset = i * _strides[0] + j * _strides[1] + k * _strides[2] + l * _strides[3];
    return _data[offset];
  }

  __hd auto operator[](u32 i, u32 j, u32 k, u32 l) -> T& {
    const auto offset = i * _strides[0] + j * _strides[1] + k * _strides[2] + l * _strides[3];
    return _data[offset];
  }

  __hd auto operator[](u32 i) const -> NdView<T, 3> {
    const auto data = _data + i * _strides[0];
    return NdView<T, 3>{data, {_shape[1], _shape[2], _shape[3]}, {_strides[1], _strides[2], _strides[3]}};
  }

 public:
  auto is_contiguous() const -> bool {
    const auto s = NdView::with_shape(_data, _shape);
    return _strides[0] == s._strides[0] && _strides[1] == s._strides[1] && _strides[2] == s._strides[2] &&
           _strides[3] == s._strides[3];
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

  void fill_with(auto&& func) {
    for (auto i = 0U; i < _shape[0]; ++i) {
      auto vol = (*this)[i];
      vol.fill_with([&](u32 j, u32 k, u32 l) { return func(i, j, k, l); });
    }
  }

  void fmt(auto& f) const {
    auto imp = f.debug_list();
    for (u32 i = 0U; i < _shape[0]; ++i) {
      imp.entry((*this)[i]);
    }
  }
};

template <class T, u32 A, u32 B>
auto reshape(NdView<T, A> src, const u32 (&new_shape)[B]) -> NdView<T, B> {
  auto dst = NdView<T, B>::with_shape(src._data, new_shape);
  if (!src.is_contiguous() || src.numel() != dst.numel()) {
    return {};
  }
  return dst;
}

}  // namespace sfc::math
