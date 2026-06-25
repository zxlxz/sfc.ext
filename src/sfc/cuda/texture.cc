#include "sfc/math.h"
#include "sfc/cuda/mod.inl"
#include "sfc/cuda/texture.h"

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic ignored "-Wmissing-designated-field-initializers"
#endif

namespace sfc::cuda {

static constexpr auto kInvalidTex = num::Int<tex_t>::MAX;

auto texture_new(buf_t arr, TexFilt filt_mode, TexAddr addr_mode) -> tex_t {
  const auto cu_filt = cudaTextureFilterMode(filt_mode);
  const auto cu_addr = cudaTextureAddressMode(addr_mode);

  const auto res_desc = cudaResourceDesc{
      .resType = cudaResourceTypeArray,
      .res = {.array = {.array = arr}},
  };

  const auto tex_desc = cudaTextureDesc{
      .addressMode = {cu_addr, cu_addr, cu_addr},
      .filterMode = cu_filt,
  };

  auto tex_obj = tex_t{0};
  CHECK_RET(cudaCreateTextureObject, &tex_obj, &res_desc, &tex_desc, nullptr);

  return tex_obj;
}

void texture_del(tex_t obj) {
  if (obj == kInvalidTex) return;

  CHECK_RET(cudaDestroyTextureObject, obj);
}

template <class T, int N>
Texture<T, N>::Texture() noexcept {}

template <class T, int N>
Texture<T, N>::~Texture() noexcept {
  cuda::texture_del(_tex._tex);
}

template <class T, int N>
Texture<T, N>::Texture(Texture&& other) noexcept : _buf{mem::move(other._buf)}, _tex{other._tex} {
  other._tex = Tex{kInvalidTex};
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
LTexture<T, N>::LTexture() noexcept {}

template <class T, int N>
LTexture<T, N>::~LTexture() noexcept {
  cuda::texture_del(_tex._tex);
}

template <class T, int N>
LTexture<T, N>::LTexture(LTexture&& other) noexcept : _buf{mem::move(other._buf)}, _tex{other._tex} {
  other._tex = Tex{kInvalidTex};
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

#define IMPL_TEXTURE(T)         \
  template class Texture<T, 2>; \
  template class Texture<T, 3>; \
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
