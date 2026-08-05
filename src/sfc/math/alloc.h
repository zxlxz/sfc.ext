#pragma once

#include "sfc/core.h"
#include "sfc/cuda/memory.h"

namespace sfc::math {

using cuda::MemKind;
using cuda::MemLocation;

template <class A>
class MemPool {
  class Inn;
  Box<Inn> _inn;

 public:
  MemPool();
  ~MemPool();
  MemPool(MemPool&& other) noexcept;
  MemPool& operator=(MemPool&& other) noexcept;

  static auto new_(A alloc = {}) -> MemPool;

 public:
  auto allocate(usize size) -> void*;
  void deallocate(void* ptr, usize size);
};

struct PoolAllocator {
  static auto pool(MemLocation location) -> MemPool&;

  static void* allocate(usize size, MemLocation location);
  static void deallocate(void* ptr, usize size, MemLocation location);
};

}  // namespace sfc::math
