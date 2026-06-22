#pragma once

#include "sfc/cuda/memory.h"

namespace sfc::math {

enum class MemType {
  CPU,
  GPU,
  UVA,
};

struct Alloc {
  MemType mtype;

 public:
  auto alloc(usize size) -> void*;
  void dealloc(void* ptr, usize size);
};

struct PoolAlloc {
  MemType mtype;

 public:
  auto alloc(usize size) -> void*;
  void dealloc(void* ptr, usize size);
};

template <class T, class A = Alloc>
class RawBuf {
  T* _ptr;
  usize _cap;
  A _a;

 public:
  RawBuf() : _ptr{nullptr}, _cap{0}, _a{} {}

  ~RawBuf() {
    if (!_ptr) return;
    _a.dealloc(_ptr, _cap * sizeof(T));
  }

  RawBuf(RawBuf&& other) noexcept : _ptr{mem::take(other._ptr)}, _cap{mem::take(other._cap)}, _a{mem::move(other._a)} {}

  RawBuf& operator=(RawBuf&& other) noexcept {
    if (this == &other) return *this;
    this->swap(other);
    return *this;
  }

  void swap(RawBuf& other) noexcept {
    if (this == &other) return;
    mem::swap(_ptr, other._ptr);
    mem::swap(_cap, other._cap);
    mem::swap(_a, other._a);
  }

  static auto with_capacity(usize cap, A a = {}) -> RawBuf {
    auto buf = RawBuf{};
    buf._ptr = ptr::cast<T>(a.alloc(cap * sizeof(T)));
    buf._cap = cap;
    buf._a = mem::move(a);

    return buf;
  }

 public:
  auto ptr() const -> T* {
    return _ptr;
  }

  auto cap() const -> usize {
    return _cap;
  }

  auto mtype() const -> MemType {
    return _a.mtype;
  }

 public:
#ifndef __CUDACC__
  auto as_slice() const -> Slice<const T> {
    return Slice{_ptr, _cap};
  }

  auto as_mut_slice() -> Slice<T> {
    return Slice{_ptr, _cap};
  }

  auto as_bytes() const -> Slice<const u8> {
    return this->as_slice().as_bytes();
  }

  auto as_mut_bytes() -> Slice<u8> {
    return this->as_mut_slice().as_mut_bytes();
  }
#endif

 public:
  void bzero() {
    cuda::fill_bytes(_ptr, 0, _cap * sizeof(T));
  }

  auto clone(MemType mtype) const -> RawBuf {
    auto res = RawBuf::with_capacity(_cap, mtype);
    cuda::copy_bytes(_ptr, res._ptr, _cap * sizeof(T));
    return res;
  }
};

}  // namespace sfc::math
