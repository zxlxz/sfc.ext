#pragma once

#include "sfc/math/vec.h"
#include "sfc/cuda/tex.h"
#include "sfc/cuda/buffer.h"

namespace sfc::math {
template <class T, int N>
struct NdSlice;
}

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

template <class T, int N>
class Texture {
  using Buf = cuda::Buffer<T>;
  using Inn = cuda::Tex<T, N>;

  Buf _buf = {};
  Inn _tex = {};

 public:
  Texture() noexcept {}

  ~Texture() noexcept {
    cuda::texture_del(_tex._tex);
  }

  Texture(Texture&& other) noexcept : _buf{static_cast<Buf&&>(other._buf)}, _tex{other._tex} {
    other._tex = {};
  }

  static auto with_shape(math::vec<u32, N> dims,
                         TexFilt filt_mode = TexFilt::Point,
                         TexAddr addr_mode = TexAddr::Clamp) -> Texture {
    auto res = Texture{};
    res._buf = Buf::xnew(BufExt::from(dims));
    res._tex = {cuda::texture_new(res._buf.as_ptr(), filt_mode, addr_mode)};
    return res;
  }

 public:
  operator Inn() const {
    return _tex;
  }

  auto operator*() const -> Inn {
    return _tex;
  }

 public:
  void set_data(const math::NdSlice<T, N>& src) {
    _buf.set_data(src._data);
  }
};

template <class T, int N>
class LTexture {
  using Buf = cuda::Buffer<T>;
  using Tex = cuda::LTex<T, N>;

  Buf _buf = {};
  Tex _tex = {};

 public:
  LTexture() noexcept : _buf{}, _tex{} {}

  ~LTexture() noexcept {
    cuda::texture_del(_tex._tex);
  }

  LTexture(LTexture&& other) noexcept : _buf{static_cast<Buf&&>(other._buf)}, _tex{other._tex} {
    other._tex = {};
  }

  static auto with_shape(math::vec<u32, N> dims,
                         TexFilt filt_mode = TexFilt::Point,
                         TexAddr addr_mode = TexAddr::Clamp) -> LTexture {
    auto res = LTexture{};
    res._buf = Buf::xnew(BufExt::from(dims));
    res._tex = {cuda::texture_new(res._buf.as_ptr(), filt_mode, addr_mode)};
    return res;
  }

 public:
  operator Tex() const {
    return _tex;
  }

  auto operator*() const -> Tex {
    return _tex;
  }

 public:
  void set_data(const math::NdSlice<T, N>& src) {
    _buf.set_data(src._data);
  }
};

}  // namespace sfc::cuda
