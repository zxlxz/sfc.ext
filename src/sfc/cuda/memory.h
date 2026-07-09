#pragma once

#include "sfc/cuda/mod.h"

namespace sfc::cuda {

struct HeapAllocator {
  static auto allocate(usize size) -> void*;
  static void deallocate(void* ptr);
};

struct HostAllocator {
  static auto allocate(usize size) -> void*;
  static void deallocate(void* ptr);
};

struct DeviceAllocator {
  static auto allocate(usize size) -> void*;
  static void deallocate(void* ptr);
};

struct ManagedAllocator {
  static auto allocate(usize size) -> void*;
  static void deallocate(void* ptr);
};

auto prefetch_cpu(void* ptr, usize size) -> Result<>;
auto prefetch_gpu(void* ptr, usize size) -> Result<>;

auto fill_bytes(void* ptr, u8 val, usize size) -> Result<>;
auto copy_bytes(const void* src, void* dst, usize size) -> Result<>;

}  // namespace sfc::cuda
