#pragma once

#include "sfc/cuda/exec.h"

namespace sfc::cuda {

using tex_t = unsigned long long;

template <class T, int N = 3>
struct Tex;

template <class T, int N = 3>
struct LTex;

template <class T>
struct Tex<T, 2> {
  tex_t _tex;

 public:
  __dev auto get(auto pos) const -> T;
};

template <class T>
struct Tex<T, 3> {
  tex_t _tex;

 public:
  __dev auto get(auto pos) const -> T;
};

template <class T>
struct LTex<T, 3> {
  tex_t _tex;

 public:
  __dev auto get(auto pos, int k) const -> T;

 public:
  struct Item {
    tex_t _tex;
    int _layer;

    __dev auto get(auto pos) const -> T {
      return LTex{_tex}.get(pos, _layer);
    }
  };

  __dev auto operator[](int k) const -> Item {
    return {_tex, k};
  }
};

#ifdef __CUDACC__
template <class T>
__dev auto Tex<T, 2>::get(auto pos) const -> T {
  auto res = T{0};
  ::tex2D(&res, _tex, pos.x, pos.y);
  return res;
}

template <class T>
__dev auto Tex<T, 3>::get(auto pos) const -> T {
  auto res = T{0};
  ::tex3D(&res, _tex, pos.x, pos.y, pos.z);
  return res;
}

template <class T>
__dev auto LTex<T, 3>::get(auto pos, int k) const -> T {
  auto res = T{0};
  ::tex2DLayered(&res, _tex, pos.x, pos.y, k);
  return res;
}
#endif

}  // namespace sfc::cuda
