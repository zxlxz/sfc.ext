#pragma once

#include "sfc/math/mod.h"
#include "sfc/core/slice.h"
#include "sfc/cuda/memory.h"

namespace sfc::math {

template <class T>
class RawBuf {
  T* _ptr;
  usize _cap;
  cuda::Alloc _a;

 public:
  RawBuf() : _ptr{nullptr}, _cap{0}, _a{} {}

  ~RawBuf() {
    if (!_ptr) return;
    _a.dealloc(_ptr);
  }

  RawBuf(RawBuf&& other) noexcept : _ptr{other._ptr}, _cap{other._cap}, _a{other._a} {
    other._ptr = nullptr;
    other._cap = 0;
    other._a = {};
  }

  RawBuf& operator=(RawBuf&& other) noexcept {
    if (this == &other) return *this;
    // to keep this operator noexcept
    // we just swap, and let the destructor of other to do the cleanup
    this->swap(other);
    return *this;
  }

  void swap(RawBuf& other) noexcept {
    mem::swap(_ptr, other._ptr);
    mem::swap(_cap, other._cap);
    mem::swap(_a, other._a);
  }

  static auto with_capacity(usize cap, cuda::MemType type = {}) -> RawBuf {
    auto buf = RawBuf{};
    buf._a = cuda::Alloc{type};
    buf._ptr = static_cast<T*>(buf._a.alloc(cap * sizeof(T)));
    buf._cap = cap;

    return buf;
  }

  auto ptr() const -> T* {
    return _ptr;
  }

  auto cap() const -> usize {
    return _cap;
  }

  auto mtype() const -> cuda::MemType {
    return _a.type;
  }

  auto as_slice() const -> slice::Slice<const T> {
    return {_ptr, _cap};
  }

  auto as_mut_slice() -> slice::Slice<T> {
    return {_ptr, _cap};
  }

  auto as_bytes() const -> slice::Slice<const u8> {
    return this->as_slice().as_bytes();
  }

  auto as_mut_bytes() -> slice::Slice<u8> {
    return this->as_mut_slice().as_mut_bytes();
  }

  void copy_from(const RawBuf& src) {
    const auto cnt = src._cap < _cap ? src._cap : _cap;
    cuda::copy_bytes(src._ptr, _ptr, cnt * sizeof(T));
  }

  void zero() {
    cuda::fill_bytes(_ptr, 0, _cap * sizeof(T));
  }

  void sync(cuda::MemType mtype) {
    if (_a.type == mtype) return;
    auto new_buf = RawBuf::with_capacity(_cap, mtype);
    this->swap(new_buf);
  }
};

}  // namespace sfc::math
