#pragma once

#include "sfc/math/ndslice.h"
#include "sfc/math/alloc.h"

namespace sfc::math {

class RawBuf {
  using A = PoolAllocator;
  u8* _ptr{nullptr};
  usize _size{0};
  MemLocation _location{MemKind::CPU, 0};
  [[no_unique_address]] A _alloc{};

 public:
  explicit RawBuf() noexcept;
  ~RawBuf();

  RawBuf(RawBuf&& other) noexcept;
  RawBuf& operator=(RawBuf&& other) noexcept;

  static auto new_(usize size, MemLocation location) -> RawBuf;

 public:
  auto ptr() const -> u8*;
  auto capacity() const -> usize;
  auto mem_location() const -> MemLocation;

  auto as_bytes() const -> slice::Slice<const u8>;
  auto as_mut_bytes() -> slice::Slice<u8>;

 public:
  void bzero();
  void sync(MemLocation location);
  void copy_from(const RawBuf& other);
};

template <class T, u32 N = 1>
class [[nodiscard]] NdArray {
  using Buf = RawBuf;
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

  static auto new_(const u32 (&shape)[N], MemLocation loc = {}) -> NdArray {
    const auto numel = Inn{nullptr, shape, {}}.numel();
    auto buf = Buf::new_(numel * sizeof(T), loc);
    return NdArray::from_buf(mem::move(buf), shape);
  }

  auto data() const -> T* {
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

  auto mem_location() const -> MemLocation {
    return _buf.mem_location();
  }

  auto as_bytes() const -> slice::Slice<const u8> {
    return _buf.as_bytes();
  }

  auto as_mut_bytes() -> slice::Slice<u8> {
    return _buf.as_mut_bytes();
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
  void bzero() {
    _buf.bzero();
  }

  void sync(MemLocation location) {
    _buf.sync(location);
    _inn._data = ptr::cast<T>(_buf.ptr());
  }

  void for_each(auto&& f) {
    _inn.for_each(f);
  }

  void fmt(auto& f) const {
    _inn.fmt(f);
  }
};

template <class T = f32, u32 N = 1>
auto array(const u32 (&shape)[N], MemLocation location = {}) -> NdArray<T, N> {
  return NdArray<T, N>::new_(shape, location);
}

template <class T = f32, u32 N = 1>
auto zero(const u32 (&shape)[N], MemLocation location = {}) -> NdArray<T, N> {
  auto a = NdArray<T, N>::new_(shape, location);
  a.bzero();
  return a;
}

}  // namespace sfc::math
