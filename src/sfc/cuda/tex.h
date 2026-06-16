#pragma once

#include "sfc/cuda/exec.h"

#ifndef __CUDACC__
extern void tex2D(void* res, unsigned long long tex, float x, float y);
extern void tex3D(void* res, unsigned long long tex, float x, float y, float z);
extern void tex2DLayered(void* res, unsigned long long tex, float x, float y, int layer);
#endif

namespace sfc::cuda {

using tex_t = unsigned long long;

template <class T, int N = 3>
struct Tex;

template <class T, int N = 3>
struct LTex;

template <class T>
struct Tex<T, 2> {
  using Item = T;
  tex_t _tex;

 public:
  __dev auto load(math::vec2f pos) const -> T {
    auto res = T{0};
    ::tex2D(&res, _tex, pos.x, pos.y);
    return res;
  }
};

template <class T>
struct Tex<T, 3> {
  using Item = T;
  tex_t _tex;

 public:
  __dev auto load(math::vec3f pos) const -> T {
    auto res = T{0};
    ::tex3D(&res, _tex, pos.x, pos.y, pos.z);
    return res;
  }
};

template <class T>
struct LTex<T, 3> {
  using Item = T;
  tex_t _tex;

 public:
  __dev auto load(math::vec2f pos, int layer) const -> T {
    auto res = T{0};
    ::tex2DLayered(&res, _tex, pos.x, pos.y, layer);
    return res;
  }

 public:
  struct Layer {
    tex_t _tex;
    int _layer;

    __dev auto load(math::vec2f pos) const -> T {
      auto res = T{0};
      ::tex2DLayered(&res, _tex, pos.x, pos.y, _layer);
      return res;
    }
  };

  __dev auto operator[](int k) const -> Layer {
    return Layer{_tex, k};
  }
};

}  // namespace sfc::cuda
