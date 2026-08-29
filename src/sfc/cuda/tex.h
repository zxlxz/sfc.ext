#pragma once

#include "sfc/cuda/mod.h"
#include "sfc/math/vec.h"

namespace sfc::cuda {

template <class T, int N = 3>
struct Tex;

template <class T, int N = 3>
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

}  // namespace sfc::cuda
