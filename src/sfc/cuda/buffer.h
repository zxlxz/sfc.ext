#pragma once

#include "sfc/core.h"

struct CUarray_st;

namespace sfc::math {
template <class T, int N>
struct vec;
}

namespace sfc::cuda {

using buf_t = struct ::CUarray_st*;
using tex_t = unsigned long long;

struct Extent {
  u32 width;
  u32 height;  //  1D:    height=0
  u32 depth;   //  1D/2D: depth=0

  template <int N>
  static auto from(math::vec<u32, N> dims) -> Extent {
    static_assert(N >= 1 && N <= 3, "invalid dimension");
    if constexpr (N == 1) return {dims.x, 0, 0};
    if constexpr (N == 2) return {dims.x, dims.y, 0};
    if constexpr (N == 3) return {dims.x, dims.y, dims.z};
  }
};

template <class T>
class Buffer {
  buf_t _arr = nullptr;

 public:
  Buffer() noexcept;
  ~Buffer();
  Buffer(Buffer&& other) noexcept;
  Buffer& operator=(Buffer&& other) noexcept;

  static auto xnew(Extent ext) -> Buffer;

 public:
  auto as_ptr() const -> buf_t;
  void set_data(const T* src);
};

}  // namespace sfc::cuda
