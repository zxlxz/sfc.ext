#pragma once

#include "sfc/math/vec.h"

namespace sfc::math {

template <class T, u32 N>
struct NdSlice;

template <class T>
struct NdSlice<T, 1> {
  T* _data = nullptr;
  u32 _shape[1] = {};
  u32 _strides[1] = {};

 public:
  __hd NdSlice() noexcept {}

  template <u32... I>
  __hd NdSlice([[maybe_unused]] ops::Idxs<I...> dims, T* data, const u32 shape[], const u32 strides[])
      : _data{data}, _shape{shape[I]...}, _strides{strides[I]...} {}

  __hd NdSlice(T* data, const u32 (&shape)[1], const u32 (&strides)[1])
      : _data{data}, _shape{shape[0]}, _strides{strides[0]} {}

  static auto with_shape(T* data, const u32 (&shape)[1]) -> NdSlice {
    const u32 strides[] = {1};
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

 public:
  __hd auto operator[](u32 i) const -> const T& {
    return _data[i * _strides[0]];
  }

  __hd auto operator[](u32 i) -> T& {
    return _data[i * _strides[0]];
  }

  __hd auto contains(const u32 (&idx)[1]) const -> bool {
    return idx[0] < _shape[0];
  }

  __hd auto get(const u32 (&idx)[1]) const -> T {
    const auto offset = idx[0] * _strides[0];
    return _data[offset];
  }

  __hd void set(const u32 (&idx)[1], T val) {
    const auto offset = idx[0] * _strides[0];
    _data[offset] = val;
  }

 public:
  auto is_contiguous() const -> bool {
    return _strides[0] == 1;
  }

  void for_each(auto&& f) const {
    for (u32 i = 0U; i < _shape[0]; ++i) {
      const auto& val = (*this)[i];
      f(val, i);
    }
  }

  void for_each_mut(auto&& f) {
    for (u32 i = 0U; i < _shape[0]; ++i) {
      auto& val = (*this)[i];
      f(val, i);
    }
  }

 public:
  void fmt(auto& f) const {
    f.write_str("[");
    for (u32 i = 0U; i < _shape[0]; ++i) {
      if (i > 0) f.write_str(", ");
      f.write_val((*this)[i]);
    }
    f.write_str("]");
  }

  void fill_with(auto&& f) {
    this->for_each_mut([&](T& dst, u32 idx) {
      const auto val = f(idx);
      dst = val;
    });
  }
};

template <class T, u32 N>
struct NdSlice {
  static constexpr u32 NDIM = N;
  using InxSeqFor = ops::IdxsFor<N>;
  T* _data = nullptr;
  u32 _shape[N] = {};
  u32 _strides[N] = {};

 public:
  __hd NdSlice() noexcept {}

  template <u32... I>
  __hd NdSlice([[maybe_unused]] ops::Idxs<I...> dims, T* data, const u32 shape[], const u32 strides[])
      : _data{data}, _shape{shape[I]...}, _strides{strides[I]...} {}

  __hd NdSlice(T* data, const u32 (&shape)[N], const u32 (&strides)[N])
      : NdSlice{ops::IdxsFor<N>{}, data, shape, strides} {}

  static auto with_shape(T* data, const u32 (&shape)[N]) -> NdSlice {
    u32 strides[N] = {};
    strides[N - 1] = 1;
    for (u32 i = N - 1; i > 0; --i) {
      strides[i - 1] = strides[i] * shape[i];
    }
    return NdSlice{data, shape, strides};
  }

 public:
  __hd auto ptr() const -> T* {
    return _data;
  }

  __hd auto len() const -> u32 {
    return _shape[0];
  }

  __hd auto numel() const -> u32 {
    const auto f = [&]<u32... I>(ops::Idxs<I...>) { return (... * _shape[I]); };
    return f(ops::IdxsFor<N>{});
  }

 public:
  __hd auto operator[](u32 i) const -> NdSlice<T, N - 1> {
    const auto data = _data + i * _strides[0];
    return {ops::IdxsFor<N - 1>{}, data, _shape + 1, _strides + 1};
  }

  __hd auto contains(const u32 (&idx)[N]) const -> bool {
    const auto f = [&]<u32... I>(ops::Idxs<I...>) { return ((idx[I] < _shape[I]) && ...); };
    return f(ops::IdxsFor<N>{});
  }

  __hd auto get(const u32 (&idx)[N]) const -> T {
    const auto f = [&]<u32... I>(ops::Idxs<I...>) { return ((idx[I] * _strides[I]) + ...); };
    return _data[f(ops::IdxsFor<N>{})];
  }

  __hd void set(const u32 (&idx)[N], T val) {
    const auto f = [&]<u32... I>(ops::Idxs<I...>) { return ((idx[I] * _strides[I]) + ...); };
    _data[f(ops::IdxsFor<N>{})] = val;
  }

 public:
  auto is_contiguous() const -> bool {
    const auto s = NdSlice::with_shape(_data, _shape);
    const auto f = [&]<u32... I>(ops::Idxs<I...>) { return ((_strides[I] == s._strides[I]) && ...); };
    return f(ops::IdxsFor<N>{});
  }

  auto transpose() const -> NdSlice {
    static_assert(N == 2, "transpose is only supported for 2D slices");
    return this->permute(ops::Idxs<1, 0>{});
  }

  template <u32... I>
  auto permute(ops::Idxs<I...> dims = {}) const -> NdSlice {
    static_assert(sizeof...(I) == N);
    static_assert(((I < N) && ...));
    return NdSlice{dims, _data, _shape, _strides};
  }

  void for_each(auto&& f) const {
    for (u32 i = 0U; i < _shape[0]; ++i) {
      const auto v = (*this)[i];
      v.for_each([&](const T& val, auto... j) { f(val, i, j...); });
    }
  }

  void for_each_mut(auto&& f) {
    for (u32 i = 0U; i < _shape[0]; ++i) {
      auto v = (*this)[i];
      v.for_each_mut([&](T& val, auto... j) { f(val, i, j...); });
    }
  }

 public:
  void fmt(auto& f) const {
    auto imp = f.debug_list();
    for (u32 i = 0U; i < _shape[0]; ++i) {
      const auto v = (*this)[i];
      imp.entry(v);
    }
    imp.finish();
  }

  void fill_with(auto&& f) {
    this->for_each_mut([&](T& dst, auto... idx) {
      const auto val = f(idx...);
      dst = val;
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
