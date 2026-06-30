#pragma once

#include <cuda_runtime_api.h>
#include "sfc/core.h"

namespace sfc::math {
template <class T, int N>
struct vec;
}

namespace sfc::cuda {

using buf_t = cudaArray_t;
using tex_t = cudaTextureObject_t;

struct Extent {
  u32 width;
  u32 height;  //  1D:    height=0
  u32 depth;   //  1D/2D: depth=0

  template <int N>
  static auto from(const u32 (&shape)[N]) -> Extent {
    static_assert(N >= 1 && N <= 3, "invalid dimension");
    if constexpr (N == 1) return {shape[0], 0, 0};
    if constexpr (N == 2) return {shape[0], shape[1], 0};
    if constexpr (N == 3) return {shape[0], shape[1], shape[2]};
  }
};

template <class T>
class Buffer {
  buf_t _arr = nullptr;
  Extent _ext = {};

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
