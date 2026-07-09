#pragma once

#include "sfc/cuda/tex.h"
#include "sfc/math/ndslice.h"

namespace sfc::cuda {

enum class TexFilt {
  Point = 0,
  Linear = 1,
};

enum class TexAddr {
  Wrap = 0,    // 0
  Clamp = 1,   // 1
  Mirror = 2,  // 2
  Border = 3   // 3
};

struct Extent {
  u32 x = 0;
  u32 y = 0;
  u32 z = 0;
};

template <class T>
class Buffer {
  using buf_t = struct cudaArray*;
  buf_t _arr = nullptr;

 public:
  Buffer() noexcept;
  ~Buffer();
  Buffer(Buffer&& other) noexcept;
  Buffer& operator=(Buffer&& other) noexcept;

  static auto xnew(Extent ext) -> Buffer;
  static auto xnew_layered(Extent ext) -> Buffer;

 public:
  auto as_ptr() const -> buf_t;
  auto set_data(const T* src) -> Result<>;
};

template <class T, int N = 3>
class Texture {
  using Tex = cuda::Tex<T, N>;
  using Buf = cuda::Buffer<T>;
  u64 _tex = {};
  Buf _buf = {};

 public:
  Texture() noexcept;
  ~Texture() noexcept;
  Texture(Texture&& other) noexcept;
  Texture& operator=(Texture&& other) noexcept;

  static auto xnew(const u32 (&shape)[N], TexFilt filt = TexFilt::Point, TexAddr addr = TexAddr::Clamp) -> Texture;

 public:
  operator Tex() const {
    return {_tex};
  }

  auto operator*() const -> Tex {
    return {_tex};
  }

  auto set_data(math::NdSlice<T, N> src) -> Result<>;
};

template <class T, int N = 3>
class LTexture {
  using Tex = cuda::LTex<T, N>;
  using Buf = cuda::Buffer<T>;
  u64 _tex = {};
  Buf _buf = {};

 public:
  LTexture() noexcept;
  ~LTexture() noexcept;
  LTexture(LTexture&& other) noexcept;
  LTexture& operator=(LTexture&& other) noexcept;

  static auto xnew(const u32 (&shape)[N], TexFilt filt = TexFilt::Point, TexAddr addr = TexAddr::Clamp) -> LTexture;

 public:
  operator Tex() const {
    return {_tex};
  }

  auto operator*() const -> Tex {
    return {_tex};
  }

  auto set_data(math::NdSlice<T, N> src) -> Result<>;
};

}  // namespace sfc::cuda
