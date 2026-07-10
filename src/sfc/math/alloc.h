#pragma once

#include "sfc/core.h"
#include "sfc/cuda/memory.h"

namespace sfc::math {

using cuda::MemKind;
using cuda::MemLocation;

class MemPool;

struct PoolAllocator {
  static auto pool(MemLocation location) -> MemPool&;

  static void* allocate(usize size, MemLocation location);
  static void deallocate(void* ptr, usize size, MemLocation location);
};

}  // namespace sfc::math
