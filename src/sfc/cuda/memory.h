#pragma once

#include "sfc/cuda/mod.h"
#include "sfc/math/ndarray.h"

namespace sfc::cuda {

enum class MemKind {
  Heap = 0,
  Host = 1,
  Device = 2,
};

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
  auto allocator() const -> math::Allocator&;

 public:
  void fmt(fmt::Formatter& f) const;
};

auto copy_uva(const void* src, void* dst, usize size) -> Result<>;
auto copy_h2h(const void* src, void* dst, usize size) -> Result<>;
auto copy_h2d(const void* src, void* dst, usize size) -> Result<>;
auto copy_d2h(const void* src, void* dst, usize size) -> Result<>;
auto copy_d2d(const void* src, void* dst, usize size) -> Result<>;

auto bzero(void* dst, usize size) -> Result<>;

template <class T, u32 N = 1>
auto empty(const u32 (&shape)[N], MemLocation loc = {}) -> math::NdArray<T, N> {
  return math::NdArray<T, N>::new_(shape, loc.allocator());
}

template <class T, u32 N = 1>
auto zero(const u32 (&shape)[N], MemLocation loc = {}) -> math::NdArray<T, N> {
  return math::NdArray<T, N>::new_zeroed(shape, loc.allocator());
}

template <class T, u32 N>
auto empty_like(const math::NdArray<T, N>& array) -> math::NdArray<T, N> {
  const auto& shape = array.shape();
  return math::NdArray<T, N>::new_(shape, array.allocator());
}

}  // namespace sfc::cuda
