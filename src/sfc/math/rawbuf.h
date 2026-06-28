#pragma once

#include "sfc/alloc.h"

namespace sfc::math {

enum class MemType {
  CPU,
  GPU,
  UVA,
};

struct Allocator {
  static auto allocate(usize size, MemType mtype) -> void*;
  static void deallocate(void* ptr, usize size, MemType mtype);
  static void bzero(void* ptr, usize size, MemType mtype);
};

struct PoolAllocator {
  static auto allocate(usize size, MemType mtype) -> void*;
  static void deallocate(void* ptr, usize size, MemType mtype);
};

template <class A = Allocator>
class RawBuf {
  void* _ptr{nullptr};
  usize _size{0};
  MemType _mtype{MemType::CPU};
  [[no_unique_address]] A _a{};

 public:
  explicit RawBuf() noexcept {}

  ~RawBuf() {
    if (_ptr == nullptr) return;
    _a.deallocate(_ptr, _size, _mtype);
  }

  RawBuf(RawBuf&& other) noexcept
      : _ptr{mem::take(other._ptr)}, _size{mem::take(other._size)}, _mtype{mem::take(other._mtype)} {}

  RawBuf& operator=(RawBuf&& other) noexcept {
    if (this != &other) {
      mem::swap(_ptr, other._ptr);
      mem::swap(_size, other._size);
      mem::swap(_mtype, other._mtype);
    }
    return *this;
  }

  static auto with_capacity(usize capacity, MemType mtype, A a = {}) -> RawBuf {
    auto res = RawBuf{};
    res._ptr = a.allocate(capacity, mtype);
    res._size = capacity;
    res._mtype = mtype;
    res._a = a;
    return res;
  }

 public:
  auto ptr() const -> void* {
    return _ptr;
  }

  auto size() const -> usize {
    return _size;
  }

  auto mtype() const -> MemType {
    return _mtype;
  }

  void bzero() {
    if (_ptr == nullptr || _size == 0) return;
    Allocator::bzero(_ptr, _size, _mtype);
  }
};

}  // namespace sfc::math
