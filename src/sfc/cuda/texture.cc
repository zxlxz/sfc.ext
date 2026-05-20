#include <cuda.h>

#include "sfc/core.h"
#include "sfc/cuda/texture.h"
#include "sfc/cuda/error.h"

namespace sfc::cuda {

auto texture_new(buf_t arr, TexFilt filt_mode, TexAddr addr_mode) -> tex_t {
  sfc::expect(arr != nullptr, "texture_new: arr is null");

  const auto cu_filt = static_cast<CUfilter_mode>(filt_mode);
  const auto cu_addr = static_cast<CUaddress_mode>(addr_mode);

  const auto res_desc = CUDA_RESOURCE_DESC_st{
      .resType = CU_RESOURCE_TYPE_ARRAY,
      .res = {.array = {.hArray = arr}},
  };

  const auto tex_desc = CUDA_TEXTURE_DESC_st{
      .addressMode = {cu_addr, cu_addr, cu_addr},
      .filterMode = cu_filt,
      .flags = 0,
  };

  const auto view_desc = nullptr;

  auto tex_obj = tex_t{0};
  if (auto e = ::cuTexObjectCreate(&tex_obj, &res_desc, &tex_desc, view_desc)) {
    panic::panic_fmt("cuTexObjectCreate failed, err={}", Error{e});
  }

  return tex_obj;
}

void texture_del(tex_t obj) {
  if (obj == -1) return;

  if (auto e = ::cuTexObjectDestroy(obj)) {
    panic::panic_fmt("cuTexObjectDestroy failed, err={}", Error{e});
  }
}

template <class T, int N>
Texture<T, N>::Texture() noexcept {}

template <class T, int N>
Texture<T, N>::~Texture() noexcept {
  cuda::texture_del(_tex._tex);
}

template <class T, int N>
Texture<T, N>::Texture(Texture&& other) noexcept
    : _buf{static_cast<Buf&&>(other._buf)}, _tex{other._tex} {
  other._tex = {};
}

template <class T, int N>
auto Texture<T, N>::operator=(Texture&& other) noexcept -> Texture& {
  if (this == &other) return *this;
  mem::swap(_buf, other._buf);
  mem::swap(_tex, other._tex);
  return *this;
}

template <class T, int N>
auto Texture<T, N>::with_shape(Dim dims, TexFilt filt_mode, TexAddr addr_mode) -> Texture {
  const auto ext = Extent::from(dims);
  auto res = Texture{};
  res._buf = Buf::xnew(ext);
  res._tex = {cuda::texture_new(res._buf.as_ptr(), filt_mode, addr_mode)};
  return res;
}

template <class T, int N>
void Texture<T, N>::set_data(math::NdSlice<T, N> src) {
  _buf.set_data(src._data);
}

template <class T, int N>
Texture<T, N>::operator Tex() const {
  return _tex;
}

template <class T, int N>
auto Texture<T, N>::operator*() const -> Tex {
  return _tex;
}

template <class T, int N>
LTexture<T, N>::LTexture() noexcept {}

template <class T, int N>
LTexture<T, N>::~LTexture() noexcept {
  cuda::texture_del(_tex._tex);
}

template <class T, int N>
LTexture<T, N>::LTexture(LTexture&& other) noexcept
    : _buf{static_cast<Buf&&>(other._buf)}, _tex{other._tex} {
  other._tex = {};
}

template <class T, int N>
auto LTexture<T, N>::operator=(LTexture&& other) noexcept -> LTexture& {
  if (this == &other) return *this;
  mem::swap(_buf, other._buf);
  mem::swap(_tex, other._tex);
  return *this;
}

template <class T, int N>
auto LTexture<T, N>::with_shape(Dim dims, TexFilt filt_mode, TexAddr addr_mode) -> LTexture {
  const auto ext = Extent::from(dims);
  auto res = LTexture{};
  res._buf = Buf::xnew(ext);
  res._tex = {cuda::texture_new(res._buf.as_ptr(), filt_mode, addr_mode)};
  return res;
}

template <class T, int N>
void LTexture<T, N>::set_data(math::NdSlice<T, N> src) {
  _buf.set_data(src._data);
}

template <class T, int N>
LTexture<T, N>::operator Tex() const {
  return _tex;
}

template <class T, int N>
auto LTexture<T, N>::operator*() const -> Tex {
  return _tex;
}

#define IMPL_TEXTURE(T)          \
  template class Texture<T, 2>;  \
  template class Texture<T, 3>;  \
  template class LTexture<T, 3>
IMPL_TEXTURE(u8);
IMPL_TEXTURE(u16);
IMPL_TEXTURE(u32);

IMPL_TEXTURE(i8);
IMPL_TEXTURE(i16);
IMPL_TEXTURE(i32);

IMPL_TEXTURE(f32);
#undef IMPL_TEXTURE
}  // namespace sfc::cuda
