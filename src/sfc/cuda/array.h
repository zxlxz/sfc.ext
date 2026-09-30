#pragma once

#include "sfc/cuda/mod.h"
#include "sfc/math/ndview.h"

namespace sfc::cuda {

struct Extent {
  usize x = 0;
  usize y = 0;
  usize z = 0;

 public:
  template <u32 N>
  static auto from_shape(const u32 (&shape)[N]) -> Extent {
    static_assert(N > 0 && N <= 3, "cuda::Extent: only support 1D, 2D, and 3D");
    const auto nx = N > 0 ? shape[0] : 0;
    const auto ny = N > 1 ? shape[1] : 0;
    const auto nz = N > 2 ? shape[2] : 0;
    return Extent{nx, ny, nz};
  }

 public:
  auto operator==(const Extent& other) const -> bool {
    return x == other.x && y == other.y && z == other.z;
  }
};

template <class T>
class Array {
  using arr_t = struct CUarray_st*;
  using ext_t = Extent;
  arr_t _arr = nullptr;

 public:
  Array() noexcept;
  ~Array();
  Array(Array&& other) noexcept;
  Array& operator=(Array&& other) noexcept;

  static auto new_(Extent ext) -> Array;
  static auto new_layered(Extent ext) -> Array;

 public:
  auto as_ptr() const -> arr_t;
  auto extent() const -> Extent;

  template <u32 N>
  auto set_data(math::NdView<T, N> view) -> Result<>;
};

}  // namespace sfc::cuda
