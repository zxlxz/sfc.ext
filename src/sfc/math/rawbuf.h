#pragma once

#include "sfc/cuda/memory.h"

namespace sfc::math {

enum class MemType {
  CPU,
  GPU,
  UVA,
};

class Allocator {
 protected:
  Allocator();
  ~Allocator();

  Allocator(const Allocator&) = delete;
  Allocator& operator=(const Allocator&) = delete;

 public:
  virtual void* allocate(usize size, MemType mtype) = 0;
  virtual void deallocate(void* ptr, usize size, MemType mtype) = 0;
};

class DefaultAlloc : public Allocator {
  DefaultAlloc();
  ~DefaultAlloc();

 public:
  static auto instance() -> DefaultAlloc&;

  void* allocate(usize size, MemType mtype) override;
  void deallocate(void* ptr, usize size, MemType mtype) override;
};

class PoolAlloc : public Allocator {
  PoolAlloc();
  ~PoolAlloc();

 public:
  static auto instance() -> PoolAlloc&;

  void* allocate(usize size, MemType mtype) override;
  void deallocate(void* ptr, usize size, MemType mtype) override;
};

template <class T>
class RawBuf {
  T* _ptr;
  usize _cap;
  MemType _mtype;
  Allocator* _a = &DefaultAlloc::instance();

 public:
  RawBuf() : _ptr{nullptr}, _cap{0}, _a{} {}

  ~RawBuf() {
    if (!_ptr) return;
    _a->deallocate(_ptr, _cap * sizeof(T), _mtype);
  }

  RawBuf(RawBuf&& other) noexcept : _ptr{mem::take(other._ptr)}, _cap{mem::take(other._cap)}, _a{mem::move(other._a)} {}

  RawBuf& operator=(RawBuf&& other) noexcept {
    this->swap(other);
    return *this;
  }

  void swap(RawBuf& other) noexcept {
    if (this == &other) return;
    mem::swap(_ptr, other._ptr);
    mem::swap(_cap, other._cap);
    mem::swap(_a, other._a);
  }

  static auto with_capacity(usize capacity, MemType mtype, Allocator& alloc = DefaultAlloc::instance()) -> RawBuf {
    auto buf = RawBuf{};
    buf._ptr = ptr::cast<T>(alloc.allocate(capacity * sizeof(T), mtype));
    buf._cap = capacity;
    buf._mtype = mtype;
    buf._a = &alloc;

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
    return _mtype;
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
    auto res = RawBuf::with_capacity(_cap, mtype, *_a);
    cuda::copy_bytes(_ptr, res._ptr, _cap * sizeof(T));
    return res;
  }
};

}  // namespace sfc::math
