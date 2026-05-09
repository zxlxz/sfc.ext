#pragma once

#include "sfc/cuda/mod.h"
#include "sfc/math/vec.h"

struct CUarray_st;

namespace sfc::math {
template <class T, int N>
struct vec;
}

namespace sfc::cuda {

using buf_t = struct ::CUarray_st*;
using tex_t = unsigned long long;

struct BufFmt {
  enum Kind { Unknown = 0, SInt = 0x01, UInt = 0x02, Float = 0x10 };
  Kind kind;
  u32 size;

  template <class T>
  static auto of() -> BufFmt;
};

struct BufExt {
  u32 width;
  u32 height = 0;  //  1D:    height=0
  u32 depth = 0;   //  1D/2D: depth=0

  template <int N>
  static auto from(const math::vec<u32, N>& dims) -> BufExt {
    static_assert(N >= 1 && N <= 3, "BufExt only supports up 1D/2D/3D");
    if constexpr (N == 1) {
      return BufExt{dims.x, 0, 0};
    } else if constexpr (N == 2) {
      return BufExt{dims.x, dims.y, 0};
    } else {
      return BufExt{dims.x, dims.y, dims.z};
    }
  }

  static auto from(math::vec<u32, 2> dims) -> BufExt {
    return BufExt{dims.x, dims.y, 0};
  }

  static auto from(math::vec<u32, 3> dims) -> BufExt {
    return BufExt{dims.x, dims.y, dims.z};
  }
};

auto buffer_new(BufFmt fmt, BufExt ext, bool is_layered = false) -> buf_t;
void buffer_del(buf_t arr);
void buffer_set(buf_t arr, const void* src);

template <class T>
class Buffer {
  buf_t _arr = nullptr;

 public:
  Buffer() noexcept : _arr{nullptr} {}

  ~Buffer() {
    cuda::buffer_del(_arr);
  }

  Buffer(Buffer&& other) noexcept : _arr{other._arr} {
    other._arr = nullptr;
  }

  Buffer& operator=(Buffer&& other) noexcept {
    if (this == &other) return *this;
    cuda::buffer_del(_arr);
    _arr = other._arr;
    other._arr = nullptr;
    return *this;
  }

  static auto with_shape(BufExt ext) -> Buffer {
    const auto fmt = BufFmt::of<T>();
    auto res = Buffer{};
    res._arr = cuda::buffer_new(fmt, ext, false);
    return res;
  }

  auto as_ptr() const -> buf_t {
    return _arr;
  }

  void set_data(const T* src) {
    cuda::buffer_set(_arr, src);
  }
};

}  // namespace sfc::cuda
