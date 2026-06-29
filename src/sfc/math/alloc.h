#pragma once

#include "sfc/alloc.h"

namespace sfc::math {

enum class MemType {
  CPU,
  GPU,
  UVA,
};

struct SysAllocator {
  static void* allocate(usize size, MemType mtype);
  static void deallocate(void* ptr, usize size, MemType mtype);
};

struct PoolAllocator {
  static void* allocate(usize size, MemType mtype);
  static void deallocate(void* ptr, usize size, MemType mtype);
};

}  // namespace sfc::math
