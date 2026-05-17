#pragma once

#include "sfc/cuda/mod.h"

namespace sfc::cuda {

using tex_t = unsigned long long;

template <class T, int N = 3>
struct Tex;

template <class T, int N = 3>
struct LTex;

template <class T>
struct Tex<T, 2> {
  tex_t _tex = 0;

 public:
  __dev auto get(auto pos) const -> T {
    auto res = T{0};
#ifdef __CUDACC__
    ::tex2D(&res, _tex, pos.x, pos.y);
#endif
    return res;
  }
};

template <class T>
struct Tex<T, 3> {
  tex_t _tex = 0;

 public:
  __dev auto get(auto pos) const -> T {
    auto res = T{0};
#ifdef __CUDACC__
    ::tex3D(&res, _tex, pos.x, pos.y, pos.z);
#endif
    return res;
  }
};

template <class T>
struct LTex<T, 3> {
  tex_t _tex = 0;

 public:
  __dev auto get(auto pos, int k) const -> T {
    auto res = T{0};
#ifdef __CUDACC__
    ::tex2DLayered(&res, _tex, pos.x, pos.y, k);
#endif
    return res;
  }

 public:
  struct Item {
    tex_t _tex;
    int _layer;

    __dev auto get(auto pos) const -> T {
      auto res = T{0};
#ifdef __CUDACC__
      ::tex2DLayered(&res, _tex, pos.x, pos.y, _layer);
#endif
      return res;
    }
  };

  __dev auto operator[](int k) const -> Item {
    return {_tex, k};
  }
};

}  // namespace sfc::cuda
