#pragma once

#include "sfc/alloc.h"
#include "sfc/math/ndslice.h"
#include "sfc/math/alloc.h"

namespace sfc::math {

template <class T, u32 N = 1, class A = alloc::Global>
class [[nodiscard]] NdArray {
  using Buf = alloc::RawBuf<A>;
  using Inn = NdSlice<T, N>;
  Buf _buf{};
  Inn _inn{};

 public:
  NdArray() noexcept : _buf{}, _inn{nullptr, {}, {}} {}
  ~NdArray() {}

  NdArray(NdArray&& other) noexcept = default;
  NdArray& operator=(NdArray&& other) noexcept = default;

  static auto from_buf(Buf buf, const u32 (&shape)[N]) -> NdArray {
    u32 strides[N] = {};
    for (auto i = N; i != 0; --i) {
      strides[i - 1] = i == N ? 1 : shape[i] * strides[i];
    }

    const auto ptr = ptr::cast<T>(buf.ptr());
    auto res = NdArray{};
    res._buf = mem::move(buf);
    res._inn = Inn{ptr, shape, strides};
    return res;
  }

  static auto new_(const u32 (&shape)[N], A alloc = {}) -> NdArray {
    const auto numel = Inn{nullptr, shape, {}}.numel();
    auto buf = Buf::with_capacity(numel * sizeof(T), alloc);
    return NdArray::from_buf(mem::move(buf), shape);
  }

  static auto new_zeroed(const u32 (&shape)[N], A alloc = {}) -> NdArray {
    const auto numel = Inn{nullptr, shape, {}}.numel();
    auto buf = Buf::with_capacity_zeroed(numel * sizeof(T), alloc);
    return NdArray::from_buf(mem::move(buf), shape);
  }

  auto as_ptr() const -> const T* {
    return _inn._data;
  }

  auto as_mut_ptr() -> T* {
    return _inn._data;
  }

  auto numel() const -> u32 {
    return _inn.numel();
  }

  using shape_t = u32[N];
  auto shape() const -> const shape_t& {
    return _inn._shape;
  }

  using strides_t = u32[N];
  auto strides() const -> const u32 (&)[N] {
    return _inn._strides;
  }

  auto buf() const -> const Buf& {
    return _buf;
  }

  auto buf_mut() -> Buf& {
    return _buf;
  }

 public:
  operator Inn() const {
    return _inn;
  }

  auto operator*() const -> Inn {
    return _inn;
  }

  auto operator[](u32 idx) const -> decltype(auto) {
    return _inn[idx];
  }

  auto operator[](u32 idx) -> decltype(auto) {
    return _inn[idx];
  }

  auto operator[](const u32 (&idx)[N]) const -> T {
    return _inn[idx];
  }

  auto operator[](const u32 (&idx)[N]) -> T& {
    return _inn[idx];
  }

  auto get(const u32 (&idx)[N]) const -> T {
    return _inn[idx];
  }

  void set(const u32 (&idx)[N], T value) {
    _inn[idx] = value;
  }

 public:
  void copy_from(const NdArray& other) {
    _buf.copy_from(other._buf);
  }

  auto clone() const -> NdArray {
    auto res = NdArray::new_(_inn._shape, _buf.mem_location());
    res.copy_from(*this);
    return res;
  }

 public:
  void for_each(auto&& f) const {
    _inn.for_each(f);
  }

  void for_each_mut(auto&& f) {
    _inn.for_each_mut(f);
  }

  void fmt(auto& f) const {
    _inn.fmt(f);
  }
};

template <class T, u32 N = 1>
auto array(const u32 (&shape)[N]) -> NdArray<T, N> {
  return NdArray<T, N>::new_(shape);
}

template <class T, u32 N = 1>
auto zero(const u32 (&shape)[N]) -> NdArray<T, N> {
  auto a = NdArray<T, N>::new_zeroed(shape);
  return a;
}

template <class T, u32 N>
auto array_like(const NdArray<T, N>& array) -> NdArray<T, N> {
  const auto& shape = array.shape();
  return NdArray<T, N>::new_(shape);
}

}  // namespace sfc::math
