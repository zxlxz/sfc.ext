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

struct MemLocation {
  static const u32 kMaxDeviceCnt = 8U;

  MemKind kind{MemKind::CPU};
  u32 device{0};

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
auto array(const u32 (&shape)[N], MemLocation loc = {}) -> math::NdArray<T, N> {
  return math::NdArray<T, N>::new_(shape, loc.pool());
}

template <class T, u32 N = 1>
auto zero(const u32 (&shape)[N], MemLocation loc = {}) -> math::NdArray<T, N> {
  auto res = math::NdArray<T, N>::new_(shape, loc.pool());
  auto& buf = res.buf();
  cuda::fill_bytes(buf.ptr(), 0, buf.len() * sizeof(T));
  return res;
}

template <class T, u32 N>
auto array_like(const math::NdArray<T, N>& array) -> math::NdArray<T, N> {
  const auto& shape = array.shape();
  return math::NdArray<T, N>::new_(shape, array.allocator());
}

}  // namespace sfc::cuda
