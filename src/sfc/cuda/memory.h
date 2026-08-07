#pragma once

#include "sfc/cuda/mod.h"
#include "sfc/math/ndarray.h"

namespace sfc::cuda {

using mem::Layout;

enum MemKind {
  CPU = 0,
  RAM = 1,
  GPU = 2,
  UVA = 3,
};
auto to_str(MemKind kind) -> str::Str;

struct Allocator {
  MemKind kind{};
  u32 device{0};

 public:
  void* allocate(Layout layout);
  void deallocate(void* ptr, Layout layout);

 public:
  void fmt(fmt::Formatter& f) const;
};

template <class T, u32 N>
using NdArray = math::NdArray<T, N, mem_pool::Allocator<Allocator>>;

template <class T, u32 N = 1>
auto array(const u32 (&shape)[N], Allocator a = {}) -> NdArray<T, N> {
  return NdArray<T, N>::new_(shape, {a});
}

template <class T, u32 N = 1>
auto zero(const u32 (&shape)[N], Allocator a = {}) -> NdArray<T, N> {
  return NdArray<T, N>::new_zeroed(shape, {a});
}

template <class T, u32 N>
auto array_like(const NdArray<T, N>& array) -> NdArray<T, N> {
  const auto& shape = array.shape();
  return NdArray<T, N>::new_(shape, array.allocator());
}

auto fill_bytes(void* ptr, u8 val, usize size) -> Result<>;
auto copy_bytes(const void* src, void* dst, usize size) -> Result<>;

}  // namespace sfc::cuda
