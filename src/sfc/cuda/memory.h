#pragma once

#include "sfc/cuda/mod.h"

namespace sfc::cuda {

enum MemKind {
  CPU = 0,
  RAM = 1,
  GPU = 2,
  UVA = 3,
};
auto to_str(MemKind kind) -> str::Str;

struct MemLocation {
  MemKind kind{};
  u32 device{0};

 public:
  MemLocation(MemKind kind = {}, u32 device = {}) noexcept : kind{kind}, device{device} {}

 public:
  void fmt(fmt::Formatter& f) const;
};

auto mem_allocate(usize size, MemLocation loc) -> void*;
void mem_deallocate(void* ptr, MemLocation location);
auto mem_location(void* ptr) -> MemLocation;
auto mem_prefetch(void* ptr, usize size, MemLocation loc) -> Result<>;

auto mem_fill(void* ptr, u8 val, usize size) -> Result<>;
auto mem_copy(const void* src, void* dst, usize size) -> Result<>;

}  // namespace sfc::cuda
