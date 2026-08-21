#pragma once

#include "sfc/cuda/mod.h"
#include "sfc/math/tensor.h"

namespace sfc::cuda {

using mem::Layout;

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
  auto pool() const -> mem_pool::Pool&;

  auto allocate(Layout layout) -> void*;
  void deallocate(void* ptr, Layout layout);

 public:
  void fmt(fmt::Formatter& f) const;
};

auto fill_bytes(void* ptr, u8 val, usize size) -> Result<>;
auto copy_bytes(const void* src, void* dst, usize size) -> Result<>;

template <class T, u32 N = 1>
auto empty(const u32 (&shape)[N], MemLocation loc = {}) -> math::Tensor<T, N> {
  if (loc.kind == MemKind::Heap) {
    return math::empty<T>(shape);
  }
  return math::Tensor<T, N>::new_(shape, loc.pool());
}

template <class T, u32 N = 1>
auto zero(const u32 (&shape)[N], MemLocation loc = {}) -> math::Tensor<T, N> {
  if (loc.kind == MemKind::Heap) {
    return math::zero<T>(shape);
  }
  auto res = math::Tensor<T, N>::new_(shape, loc.pool());
  cuda::fill_bytes(res.as_mut_ptr(), 0, res.numel() * sizeof(T)).unwrap();
  return res;
}

template <class T, u32 N>
auto empty_like(const math::Tensor<T, N>& array) -> math::Tensor<T, N> {
  const auto& shape = array.shape();
  return math::Tensor<T, N>::new_(shape, array.allocator());
}

}  // namespace sfc::cuda
