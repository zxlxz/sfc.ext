#pragma once

#include "sfc/math/vec.h"

namespace sfc::math {

template <int N>
struct Shape {
  u32 _data[N];

 public:
  __hd auto operator[](int idx) const -> u32 {
    return _data[idx];
  }

 public:
  auto numel() const -> u32 {
    auto f = [&](auto... I) { return ((_data[I] * ...)); };
    return ops::fold_seq<N>(f);
  }

  auto operator==(const Shape& other) const -> bool {
    auto f = [&](auto... I) { return (((_data[I] == other._data[I]) && ...)); };
    return ops::fold_seq<N>(f);
  }

  void fmt(auto& f) const {
    return f.write_fmt("{}", Slice{_data});
  }
};

template <int N>
struct Strides {
  u32 _data[N];

 public:
  static auto from_shape(const Shape<N>& shape) -> Strides {
    auto res = Strides{};
    for (auto i = N - 1; i >= 0; --i) {
      res._data[i] = i == N - 1 ? 1 : res[i + 1] * shape[i + 1];
    }
    return res;
  }

  __hd auto operator[](int idx) const -> u32 {
    return _data[idx];
  }

  auto operator==(const Strides& other) const -> bool {
    auto f = [&](auto... I) { return (((_data[I] == other._data[I]) && ...)); };
    return ops::fold_seq<N>(f);
  }

  void fmt(auto& f) const {
    return f.write_fmt("{}", Slice{_data});
  }
};

template <int N>
struct NdIdx {
  u32 _data[N];

 public:
  __hd auto operator<(const Shape<N>& shape) const -> bool {
    auto f = [&](auto... I) { return ((_data[I] < shape._data[I]) && ...); };
    return ops::fold_seq<N>(f);
  }

  __hd auto operator*(const Strides<N>& strides) const -> u32 {
    auto f = [&](auto... I) { return ((_data[I] * strides._data[I]) + ...); };
    return ops::fold_seq<N>(f);
  }

  void fmt(auto& f) const {
    return f.write_fmt("{}", Slice{_data});
  }
};

template <class T, int Dims = 1>
struct NdView;

template <class T>
struct NdView<T, 1> {
  static constexpr auto NDIM = 1U;
  using Shape = math::Shape<NDIM>;
  using Strides = math::Strides<NDIM>;
  using Index = math::NdIdx<NDIM>;

  T* _data;
  Shape _shape;
  Strides _strides;

 public:
  __hd auto len() const -> u32 {
    return _shape._data[0];
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
    return _shape.numel();
  }

  __hd auto operator[](u32 idx) const -> const T& {
    return _data[idx * _strides._data[0]];
  }

  __hd auto operator[](u32 idx) -> T& {
    return _data[idx * _strides._data[0]];
  }

  __hd auto operator[](Index indices) const -> const T& {
    return _data[indices * _strides];
  }

  __hd auto operator[](Index indices) -> T& {
    return _data[indices * _strides];
  }

  __hd auto is_contiguous() const -> bool {
    return _strides._data[0] == 1;
  }

 public:
  void imap(auto&& f) const {
    for (auto i = 0U; i < _shape._data[0]; ++i) {
      const auto& e = _data[i * _strides._data[0]];
      f(i, e);
    }
  }

  void imap_mut(auto&& f) {
    for (auto i = 0U; i < _shape._data[0]; ++i) {
      auto& e = _data[i * _strides._data[0]];
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
struct NdView<T, 2> {
  static constexpr auto NDIM = 2U;
  using Shape = math::Shape<NDIM>;
  using Strides = math::Strides<NDIM>;
  using Index = math::NdIdx<NDIM>;

  T* _data;
  Shape _shape;
  Strides _strides;

 public:
  __hd auto len() const -> u32 {
    return _shape._data[0];
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
    return _shape.numel();
  }

  __hd auto operator[](u32 x) const -> NdView<T, NDIM - 1> {
    const auto p = _data + x * _strides._data[0];
    return NdView<T, NDIM - 1>{p, {_shape._data[1]}, {_strides._data[1]}};
  }

  __hd auto operator[](Index indices) const -> const T& {
    return _data[indices * _strides];
  }

  __hd auto operator[](Index indices) -> T& {
    return _data[indices * _strides];
  }

 public:
  void imap(auto&& f) const {
    for (auto i = 0U; i < _shape._data[0]; ++i) {
      const auto row = (*this)[i];
      for (auto j = 0U; j < _shape._data[1]; ++j) {
        const auto& e = row[j];
        f(i, j, e);
      }
    }
  }

  void imap_mut(auto&& f) {
    for (auto i = 0U; i < _shape._data[0]; ++i) {
      auto row = (*this)[i];
      for (auto j = 0U; j < _shape._data[1]; ++j) {
        auto& e = row[j];
        f(i, j, e);
      }
    }
  }

  void fmt(auto& f) const {
    for (auto i = 0U; i < _shape._data[0]; ++i) {
      const auto row = (*this)[i];
      f.write_fmt("{}\n", row);
    }
  }
};

template <class T>
struct NdView<T, 3> {
  static constexpr auto NDIM = 3;
  using Shape = math::Shape<NDIM>;
  using Strides = math::Strides<NDIM>;
  using Index = math::NdIdx<NDIM>;

  T* _data;
  Shape _shape;
  Strides _strides;

 public:
  __hd auto len() const -> u32 {
    return _shape._data[0];
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
    return _shape.numel();
  }

  __hd auto operator[](u32 x) const -> NdView<T, NDIM - 1> {
    const auto data = _data + x * _strides._data[0];
    const auto shape = math::Shape<NDIM - 1>{_shape._data[1], _shape._data[2]};
    const auto strides = math::Strides<NDIM - 1>{_strides._data[1], _strides._data[2]};
    return {data, shape, strides};
  }

  __hd auto operator[](Index indices) const -> const T& {
    return _data[indices * _strides];
  }

  __hd auto operator[](Index indices) -> T& {
    return _data[indices * _strides];
  }

 public:
  void imap(auto&& f) const {
    for (auto i = 0U; i < _shape._data[0]; ++i) {
      const auto mat = (*this)[i];
      mat.imap([&](u32 j, u32 k, const T& e) { f(i, j, k, e); });
    }
  }

  void imap_mut(auto&& f) {
    for (auto i = 0U; i < _shape._data[0]; ++i) {
      auto mat = (*this)[i];
      mat.imap_mut([&](u32 j, u32 k, T& e) { f(i, j, k, e); });
    }
  }
};

template <class T>
struct NdView<T, 4> {
  static constexpr auto NDIM = 4;
  using Shape = math::Shape<NDIM>;
  using Strides = math::Strides<NDIM>;
  using Index = math::NdIdx<NDIM>;
  T* _data;
  Shape _shape;
  Strides _strides;

 public:
  __hd auto len() const -> u32 {
    return _shape._data[0];
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
    return _shape.numel();
  }

  __hd auto operator[](u32 idx) const -> NdView<T, NDIM - 1> {
    const auto data = _data + idx * _strides._data[0];
    const auto shape = math::Shape<NDIM - 1>{_shape._data[1], _shape._data[2], _shape._data[3]};
    const auto strides = math::Strides<NDIM - 1>{_strides._data[1], _strides._data[2], _strides._data[3]};
    return {data, shape, strides};
  }

  __hd auto operator[](Index indices) const -> const T& {
    return _data[indices * _strides];
  }

  __hd auto operator[](Index idxs) -> T& {
    return _data[idxs * _strides];
  }

 public:
  void imap(auto&& f) const {
    for (auto w = 0U; w < _shape._data[0]; ++w) {
      const auto cube = (*this)[w];
      cube.imap([&](u32 i, u32 j, u32 k, const T& e) { f(i, j, k, w, e); });
    }
  }

  void imap_mut(auto&& f) {
    for (auto w = 0U; w < _shape._data[0]; ++w) {
      auto cube = (*this)[w];
      cube.imap_mut([&](u32 i, u32 j, u32 k, T& e) { f(i, j, k, w, e); });
    }
  }
};

}  // namespace sfc::math
