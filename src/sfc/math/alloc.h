#pragma once

#include "sfc/alloc.h"

namespace sfc::math {

enum class MemKind {
  CPU,
  GPU,
  UVA,
};

struct SysAllocator {
  static void* allocate(usize size, MemKind kind);
  static void deallocate(void* ptr, usize size, MemKind kind);
};

struct PoolAllocator {
  static void* allocate(usize size, MemKind kind);
  static void deallocate(void* ptr, usize size, MemKind kind);
};

}  // namespace sfc::math
