#pragma once

#include "sfc/core.h"

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

void fill_bytes(void* ptr, u8 val, usize size);
void copy_bytes(const void* src, void* dst, usize size);

}  // namespace sfc::cuda
