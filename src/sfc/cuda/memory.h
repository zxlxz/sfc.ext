#pragma once

#include "sfc/cuda/mod.h"

namespace sfc::cuda {

auto heap_alloc(usize size) -> void*;
void heap_free(void* ptr);

auto host_alloc(usize size) -> void*;
void host_free(void* ptr);

auto device_alloc(usize size) -> void*;
void device_free(void* ptr);

auto managed_alloc(usize size) -> void*;
void managed_free(void* ptr);

void prefetch_cpu(void* ptr, usize size);
void prefetch_gpu(void* ptr, usize size);

void copy_bytes(const void* src, void* dst, usize size);
void fill_bytes(void* ptr, u8 val, usize cnt);

enum class MemType {
  Heap,    // heap
  Host,    // pinned host
  Device,  // device
  UVA,     // managed
};

struct Alloc {
  MemType type = MemType::Heap;

 public:
  auto alloc(usize size) -> void*;
  void dealloc(void* ptr);
};

}  // namespace sfc::cuda
