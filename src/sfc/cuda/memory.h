#pragma once

#include "sfc/cuda/mod.h"
#include "sfc/math/tensor.h"

namespace sfc::cuda {

using mem::Layout;
using math::Tensor;

enum class MemKind {
  Heap = 0,
  Host = 1,
  Device = 2,
};
auto to_str(MemKind kind) -> str::Str;

struct MemLocation {
  static const u32 kMaxDeviceCnt = 8U;
  MemKind kind{MemKind::Heap};
  u32 device{0};

 public:
  static auto Heap() -> MemLocation;
  static auto Host() -> MemLocation;
  static auto Device(u32 device = 0) -> MemLocation;

 public:
  auto on_host() const -> bool;
  auto on_device() const -> bool;

  auto allocate(Layout layout) -> void*;
  void deallocate(void* ptr, Layout layout);

  auto pool() const -> mem_pool::Pool&;

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

template <class T, u32 N = 1>
auto empty(const u32 (&shape)[N], MemLocation loc = {}) -> Tensor<T, N> {
  return Tensor<T, N>::new_(shape, loc.pool());
}

template <class T, u32 N = 1>
auto zero(const u32 (&shape)[N], MemLocation loc = {}) -> Tensor<T, N> {
  auto res = Tensor<T, N>::new_(shape, loc.pool());
  auto blk = MemBlock{res.as_mut_ptr(), res.numel() * sizeof(T), loc};
  blk.fill_bytes(0).unwrap();
  return res;
}

template <class T, u32 N>
auto empty_like(const Tensor<T, N>& array) -> Tensor<T, N> {
  const auto& shape = array.shape();
  return Tensor<T, N>::new_(shape, array.allocator());
}

}  // namespace sfc::cuda
