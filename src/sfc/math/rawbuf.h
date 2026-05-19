#pragma once

#include "sfc/core.h"
#include "sfc/cuda/memory.h"

namespace sfc::math {

using cuda::Alloc;
using cuda::MemType;

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

 public:
  auto ptr() const -> T* {
    return _ptr;
  }

  auto cap() const -> usize {
    return _cap;
  }

  auto mtype() const -> cuda::MemType {
    return _a.mtype;
  }

 public:
#ifndef __CUDACC__
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
#endif

 public:
  void bzero() {
    const auto blk = cuda::MemBlock{_ptr, _cap * sizeof(T), _a.mtype};
    cuda::fill_bytes(blk, 0);
  }

  void copy_from(const RawBuf& src) {
    const auto dst_blk = cuda::MemBlock{_ptr, _cap * sizeof(T), _a.mtype};
    const auto src_blk = cuda::MemBlock{src._ptr, src._cap * sizeof(T), src._a.mtype};
    cuda::copy_bytes(src_blk, dst_blk);
  }

  auto clone(cuda::MemType mtype) const -> RawBuf {
    auto res = RawBuf::with_capacity(_cap, mtype);
    res.copy_from(*this);
    return res;
  }

  void sync(cuda::MemType mtype) {
    if (_a.mtype == mtype) {
      return;
    }
    auto new_buf = this->clone(mtype);
    this->swap(new_buf);
  }
};

}  // namespace sfc::math
