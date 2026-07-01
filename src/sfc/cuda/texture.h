#pragma once

#include "sfc/cuda/tex.h"
#include "sfc/cuda/buffer.h"
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

auto texture_new(buf_t arr, TexFilt filt_mode, TexAddr addr_mode) -> tex_t;
void texture_del(tex_t obj);

template <class T, int N = 3>
class Texture {
  using Buf = cuda::Buffer<T>;
  using Tex = cuda::Tex<T, N>;
  using Shape = u32[N];

  Buf _buf = {};
  Tex _tex = {};

 public:
  Texture() noexcept;
  ~Texture() noexcept;
  Texture(Texture&& other) noexcept;
  Texture& operator=(Texture&& other) noexcept;

  static auto xnew(const Shape& shape, TexFilt filt = TexFilt::Point, TexAddr addr = TexAddr::Clamp) -> Texture;

 public:
  auto operator*() const -> Tex {
    return _tex;
  }

  void set_data(math::NdSlice<T, N> src);
};

template <class T, int N = 3>
class LTexture {
  using Buf = cuda::Buffer<T>;
  using Tex = cuda::LTex<T, N>;
  using Shape = u32[N];

  Buf _buf = {};
  Tex _tex = {};

 public:
  LTexture() noexcept;
  ~LTexture() noexcept;
  LTexture(LTexture&& other) noexcept;
  LTexture& operator=(LTexture&& other) noexcept;

  static auto xnew(const Shape& shape, TexFilt filt = TexFilt::Point, TexAddr addr = TexAddr::Clamp) -> LTexture;

 public:
  auto operator*() const -> Tex {
    return _tex;
  }

  void set_data(math::NdSlice<T, N> src);
};

}  // namespace sfc::cuda
