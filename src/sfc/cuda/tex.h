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
  using Item = T;
  tex_t _tex;

 public:
  __dev auto load(auto pos) const -> T;
};

template <class T>
struct Tex<T, 3> {
  using Item = T;
  tex_t _tex;

 public:
  __dev auto load(auto pos) const -> T;
};

template <class T>
struct LTex<T, 3> {
  using Item = T;
  tex_t _tex;

 public:
  __dev auto load(auto pos, int k) const -> T;

 public:
  struct Layer {
    tex_t _tex;
    int _sid;
    __dev auto load(auto pos) const -> T;
  };

  __dev auto operator[](int k) const -> Layer { return Layer{_tex, k}; }
};

#ifdef __CUDACC__
template <class T>
__dev auto Tex<T, 2>::load(auto pos) const -> T {
  auto res = T{0};
  ::tex2D(&res, _tex, pos.x, pos.y);
  return res;
}

template <class T>
__dev auto Tex<T, 3>::load(auto pos) const -> T {
  auto res = T{0};
  ::tex3D(&res, _tex, pos.x, pos.y, pos.z);
  return res;
}

template <class T>
__dev auto LTex<T, 3>::load(auto pos, int k) const -> T {
  auto res = T{0};
  ::tex2DLayered(&res, _tex, pos.x, pos.y, k);
  return res;
}

template <class T>
__dev auto LTex<T, 3>::Layer::load(auto pos) const -> T {
  auto res = T{0};
  ::tex2DLayered(&res, _tex, pos.x, pos.y, _sid);
  return res;
}
#endif

}  // namespace sfc::cuda
