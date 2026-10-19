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

template <class T, u32 N>
struct Tex {
  using Item = T;
  u64 _tex;

 public:
  __dev auto load(math::vec<f32, N> pos) const -> T;
};

template <class T, u32 N>
struct LTex {
  using Item = T;
  u64 _tex;

 public:
  __dev auto load(math::vec<f32, N - 1> pos, int layer) const -> T;
};

class TexObject {
  u64 _tex;

 public:
  TexObject() noexcept;
  ~TexObject() noexcept;

  TexObject(TexObject&& other) noexcept;
  TexObject& operator=(TexObject&& other) noexcept;

  static auto new_(CUarray_st* arr, TexFilt filt, TexAddr addr) -> TexObject;

 public:
  auto as_tex() const -> u64 {
    return _tex;
  }
};

template <class T, u32 N = 3>
class Texture {
  Array<T> _arr = {};
  TexObject _obj = {};

 public:
  static auto new_(const u32 (&shape)[N], TexFilt filt = TexFilt::Point, TexAddr addr = TexAddr::Clamp) -> Texture {
    auto arr = Array<T>::new_(Extent::from_shape(shape));
    auto obj = TexObject::new_(arr.as_ptr(), filt, addr);

    auto res = Texture{};
    res._arr = mem::move(arr);
    res._obj = mem::move(obj);
    return res;
  }

 public:
  auto operator*() const -> Tex<T, N> {
    return {_obj.as_tex()};
  }

  auto as_tex() const -> Tex<T, N> {
    return {_obj.as_tex()};
  }

 public:
  auto set_data(math::NdView<T, N> src) -> Result<> {
    return _arr.set_data(src);
  }
};

template <class T, u32 N = 3>
class LTexture {
  Array<T> _arr = {};
  TexObject _obj = {};

 public:
  static auto new_(const u32 (&shape)[N], TexFilt filt = TexFilt::Point, TexAddr addr = TexAddr::Clamp) -> LTexture {
    auto arr = Array<T>::new_layered(Extent::from_shape(shape));
    auto obj = TexObject::new_(arr.as_ptr(), filt, addr);

    auto res = LTexture{};
    res._arr = mem::move(arr);
    res._obj = mem::move(obj);
    return res;
  }

 public:
  auto operator*() const -> LTex<T, N> {
    return {_obj.as_tex()};
  }

  auto as_tex() const -> LTex<T, N> {
    return {_obj.as_tex()};
  }

 public:
  auto set_data(math::NdView<T, N> src) -> Result<> {
    return _arr.set_data(src);
  }
};

#ifdef __CUDACC__
template <class T, u32 N>
__dev auto Tex<T, N>::load(math::vec<f32, N> pos) const -> T {
  auto res = T{0};
  if constexpr (N == 1) {
    ::tex1D(&res, _tex, pos.x);
  } else if constexpr (N == 2) {
    ::tex2D(&res, _tex, pos.x, pos.y);
  } else if constexpr (N == 3) {
    ::tex3D(&res, _tex, pos.x, pos.y, pos.z);
  }
  return res;
}

template <class T, u32 N>
__dev auto LTex<T, N>::load(math::vec<f32, N - 1> pos, int layer) const -> T {
  auto res = T{0};
  if constexpr (N == 2) {
    ::tex2DLayered(&res, _tex, pos.x, layer);
  } else if constexpr (N == 3) {
    ::tex3DLayered(&res, _tex, pos.x, pos.y, layer);
  }
  return res;
}
#endif

}  // namespace sfc::cuda
