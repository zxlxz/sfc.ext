#pragma once

#include "sfc/alloc.h"

namespace sfc::math {

class Allocator {
 public:
  static auto global() -> Allocator&;

 public:
  virtual void* allocate(Layout layout) = 0;
  virtual void* allocate_zeroed(Layout layout) = 0;
  virtual void deallocate(void* ptr, Layout layout) = 0;
};

template <class T>
class RawBuf {
  ptr::Unique<T> _ptr{};
  usize _cap{0};
  Allocator* _a{nullptr};

 public:
  static auto with_capacity(usize capacity, Allocator& alloc) -> RawBuf {
    auto res = RawBuf{};
    res._ptr = (T*)alloc.allocate(Layout::array<T>(capacity));
    res._cap = capacity;
    res._a = &alloc;
    return res;
  }

  static auto with_capacity_zeroed(usize capacity, Allocator& alloc) -> RawBuf {
    auto res = RawBuf{};
    res._ptr = (T*)alloc.allocate_zeroed(Layout::array<T>(capacity));
    res._cap = capacity;
    res._a = &alloc;
    return res;
  }

 public:
  auto ptr() const noexcept -> T* {
    return _ptr._ptr;
  }

  auto cap() const noexcept -> usize {
    return _cap;
  }

  auto as_slice() const -> Slice<const T> {
    return {_ptr._ptr, _cap};
  }

  auto as_mut_slice() -> Slice<T> {
    return {_ptr._ptr, _cap};
  }
};

}  // namespace sfc::math
