#pragma once

#include "sfc/cuda/mod.h"

namespace sfc::cuda {

enum class MemType {
  Heap,    // heap
  Host,    // pinned host
  Device,  // device
  UVA,     // managed
};

struct MemBlock {
  void* ptr;
  usize size;
  MemType mtype;
};

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

void copy_bytes(MemBlock src, MemBlock dst);
void fill_bytes(MemBlock block, u8 val);

struct Alloc {
  MemType mtype = MemType::Heap;

 public:
  auto alloc(usize size) -> void*;
  void dealloc(void* ptr);
};

}  // namespace sfc::cuda
