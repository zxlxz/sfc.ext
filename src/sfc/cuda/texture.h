#pragma once

#include "sfc/cuda/array.h"

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

template <class T, u32 N = 3>
struct Tex;

template <class T, u32 N = 3>
struct LTex;

template <class T>
struct Tex<T, 2> {
  using Item = T;
  u64 _tex;

 public:
#ifdef __CUDACC__
  __dev auto load(math::vec2f pos) const -> T {
    auto res = T{0};
    ::tex2D(&res, _tex, pos.x, pos.y);
    return res;
  }
#endif
};

template <class T>
struct Tex<T, 3> {
  using Item = T;
  u64 _tex;

 public:
#ifdef __CUDACC__
  __dev auto load(math::vec3f pos) const -> T {
    auto res = T{0};
    ::tex3D(&res, _tex, pos.x, pos.y, pos.z);
    return res;
  }
#endif
};

template <class T>
struct LTex<T, 3> {
  using Item = T;
  u64 _tex;

 public:
#ifdef __CUDACC__
  __dev auto load(math::vec2f pos, int layer) const -> T {
    auto res = T{0};
    ::tex2DLayered(&res, _tex, pos.x, pos.y, layer);
    return res;
  }
#endif

 public:
  struct Layer {
    u64 _tex;
    int _layer;

#ifdef __CUDACC__
    __dev auto load(math::vec2f pos) const -> T {
      auto res = T{0};
      ::tex2DLayered(&res, _tex, pos.x, pos.y, _layer);
      return res;
    }
#endif
  };

  __dev auto operator[](int k) const -> Layer {
    return Layer{_tex, k};
  }
};

template <class T, u32 N = 3>
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
    return Tex{_tex};
  }

  auto as_tex() const -> Tex {
    return Tex{_tex};
  }

 public:
  auto set_data(math::NdView<T, N> src) -> Result<>;
};

template <class T, u32 N = 3>
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
  auto operator*() const -> Tex {
    return Tex{_tex};
  }

  auto as_tex() const -> Tex {
    return Tex{_tex};
  }

 public:
  auto set_data(math::NdView<T, N> src) -> Result<>;
};

}  // namespace sfc::cuda
