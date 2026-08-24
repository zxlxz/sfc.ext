#pragma once

#include "sfc/cuda/tex.h"
#include "sfc/math/ndview.h"

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
class Array {
  using arr_t = struct CUarray_st*;
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
  auto set_data(const T* src) -> Result<>;
};

template <class T, int N = 3>
class Texture {
  using Tex = cuda::Tex<T, N>;
  using Arr = cuda::Array<T>;
  u64 _tex = {};
  Arr _arr = {};

 public:
  Texture() noexcept;
  ~Texture() noexcept;
  Texture(Texture&& other) noexcept;
  Texture& operator=(Texture&& other) noexcept;

  static auto new_(const u32 (&shape)[N], TexFilt filt = TexFilt::Point, TexAddr addr = TexAddr::Clamp) -> Texture;

 public:
  operator Tex() const {
    return {_tex};
  }

  auto operator*() const -> Tex {
    return {_tex};
  }

  auto set_data(math::NdView<T, N> src) -> Result<>;
};

template <class T, int N = 3>
class LTexture {
  using Tex = cuda::LTex<T, N>;
  using Arr = cuda::Array<T>;
  u64 _tex = {};
  Arr _arr = {};

 public:
  LTexture() noexcept;
  ~LTexture() noexcept;
  LTexture(LTexture&& other) noexcept;
  LTexture& operator=(LTexture&& other) noexcept;

  static auto new_(const u32 (&shape)[N], TexFilt filt = TexFilt::Point, TexAddr addr = TexAddr::Clamp) -> LTexture;

 public:
  operator Tex() const {
    return {_tex};
  }

  auto operator*() const -> Tex {
    return {_tex};
  }

  auto set_data(math::NdView<T, N> src) -> Result<>;
};

}  // namespace sfc::cuda
