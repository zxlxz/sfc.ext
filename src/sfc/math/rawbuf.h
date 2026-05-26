#pragma once

#include "sfc/core/mem.h"
#include "sfc/cuda/memory.h"

#ifndef __CUDACC__
#include "sfc/core/slice.h"
#endif

namespace sfc::math {

using cuda::Alloc;
using cuda::MemType;

template <class T>
class RawBuf {
  T* _ptr;
  usize _cap;
  Alloc _a;

 public:
  RawBuf() : _ptr{nullptr}, _cap{0}, _a{} {}

  ~RawBuf() {
    if (!_ptr) return;
    _a.dealloc(_ptr);
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

  static auto with_capacity(usize cap, MemType type = {}) -> RawBuf {
    auto buf = RawBuf{};
    buf._a = Alloc{type};
    buf._ptr = ptr::cast_mut<T>(buf._a.alloc(cap * sizeof(T)));
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

  auto mtype() const -> MemType {
    return _a.mtype;
  }

  auto mblk() const -> cuda::MemBlock {
    return cuda::MemBlock{_ptr, _cap * sizeof(T), _a.mtype};
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
    const auto blk = this->mblk();
    cuda::fill_bytes(blk, 0);
  }

  void copy_from(const RawBuf& src) {
    const auto src_blk = src.mblk();
    const auto dst_blk = this->mblk();
    cuda::copy_bytes(src_blk, dst_blk);
  }

  auto clone(MemType mtype) const -> RawBuf {
    auto res = RawBuf::with_capacity(_cap, mtype);
    res.copy_from(*this);
    return res;
  }

  void sync(MemType mtype) {
    if (_a.mtype == mtype) {
      return;
    }
    auto new_buf = this->clone(mtype);
    this->swap(new_buf);
  }
};

}  // namespace sfc::math
