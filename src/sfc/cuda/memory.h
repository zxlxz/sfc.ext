#pragma once

#include "sfc/cuda/mod.h"

namespace sfc::cuda {

enum MemKind {
  CPU = 0,
  RAM = 1,
  GPU = 2,
  UVA = 3,
};
auto to_str(MemKind kind) -> Str;

struct MemLocation {
  MemKind kind{};
  u32 device{0};

 public:
  MemLocation(MemKind kind = {}, u32 device = {}) noexcept : kind{kind}, device{device} {}

 public:
  void fmt(fmt::Formatter& f) const;
};

auto allocate(usize size, MemLocation loc) -> void*;
void deallocate(void* ptr, usize size, MemLocation loc);
auto location(void* ptr) -> MemLocation;

auto prefetch_cpu(void* ptr, usize size) -> Result<>;
auto prefetch_gpu(void* ptr, usize size) -> Result<>;

auto fill_bytes(void* ptr, u8 val, usize size) -> Result<>;
auto copy_bytes(const void* src, void* dst, usize size) -> Result<>;

}  // namespace sfc::cuda
