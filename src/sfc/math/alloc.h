#pragma once

#include "sfc/core.h"

namespace sfc::math {

enum class MemKind {
  CPU,
  GPU,
  UVA,
};

struct MemLocation {
  MemKind kind = MemKind::CPU;
  u32 device = 0;

 public:
  MemLocation(MemKind kind = {}, u32 device = {}) : kind{kind}, device{device} {}
};

class MemPool;

struct SysAllocator {
  static void* allocate(usize size, MemLocation location);
  static void deallocate(void* ptr, usize size, MemLocation location);
};

struct PoolAllocator {
  static auto pool(MemLocation location) -> MemPool&;

  static void* allocate(usize size, MemLocation location);
  static void deallocate(void* ptr, usize size, MemLocation location);
};

}  // namespace sfc::math
