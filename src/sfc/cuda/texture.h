#pragma once

#include "sfc/cuda/tex.h"
#include "sfc/cuda/buffer.h"
#include "sfc/math/ndslice.h"

namespace sfc::cuda {

using tex_t = unsigned long long;

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
  using Dim = math::vec<u32, N>;

  Buf _buf = {};
  Tex _tex = {};

 public:
  Texture() noexcept;
  ~Texture() noexcept;
  Texture(Texture&& other) noexcept;
  Texture& operator=(Texture&& other) noexcept;

  static auto with_shape(Dim dims,
                         TexFilt filt_mode = TexFilt::Point,
                         TexAddr addr_mode = TexAddr::Clamp) -> Texture;

 public:
  operator Tex() const;
  auto operator*() const -> Tex;

  void set_data(math::NdSlice<T, N> src);
};

template <class T, int N = 3>
class LTexture {
  using Buf = cuda::Buffer<T>;
  using Tex = cuda::LTex<T, N>;
  using Dim = math::vec<u32, N>;

  Buf _buf = {};
  Tex _tex = {};

 public:
  LTexture() noexcept;
  ~LTexture() noexcept;
  LTexture(LTexture&& other) noexcept;
  LTexture& operator=(LTexture&& other) noexcept;

  static auto with_shape(Dim dims,
                         TexFilt filt_mode = TexFilt::Point,
                         TexAddr addr_mode = TexAddr::Clamp) -> LTexture;

 public:
  operator Tex() const;
  auto operator*() const -> Tex;

  void set_data(math::NdSlice<T, N> src);
};

}  // namespace sfc::cuda
