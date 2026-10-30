#pragma once

#include "sfc/cuda/mod.h"
#include "sfc/math/ndarray.h"

namespace sfc::cuda {

enum class MemKind {
  Heap = 0,
  Host = 1,
  Device = 2,
};
auto to_str(MemKind kind) -> str::Str;

struct MemLocation {
  MemKind kind{MemKind::Heap};
  u32 device{0};

 public:
  static auto Heap() -> MemLocation;
  static auto Host() -> MemLocation;
  static auto Device(u32 device = 0) -> MemLocation;

 public:
  auto on_host() const -> bool;
  auto on_device() const -> bool;
  auto allocator() const -> math::Allocator;

 public:
  void fmt(fmt::Formatter& f) const;
};

struct MemBlock {
  void* ptr;
  usize size;
  MemLocation loc;

 public:
  auto fill_bytes(u8 val) -> Result<>;
  auto copy_from(MemBlock src) -> Result<>;
};

using math::NdArray;

template <class T, u32 N = 1>
auto empty(const u32 (&shape)[N], MemLocation loc = {}) -> NdArray<T, N> {
  return NdArray<T, N>::new_(shape, loc.allocator());
}

template <class T, u32 N = 1>
auto zero(const u32 (&shape)[N], MemLocation loc = {}) -> NdArray<T, N> {
  auto res = NdArray<T, N>::new_(shape, loc.allocator());
  auto blk = MemBlock{res.as_mut_ptr(), res.numel() * sizeof(T), loc};
  blk.fill_bytes(0).unwrap();
  return res;
}

template <class T, u32 N>
auto empty_like(const NdArray<T, N>& array) -> NdArray<T, N> {
  const auto& shape = array.shape();
  return NdArray<T, N>::new_(shape, array.allocator());
}

}  // namespace sfc::cuda
