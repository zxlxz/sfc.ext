#include <cuda.h>

#include "sfc/cuda/mod.h"
#include "sfc/cuda/stream.h"
#include "sfc/cuda/texture.h"

namespace sfc::cuda {

namespace detail {

using arr_t = CUarray;

auto texture_new(arr_t arr, TexFilt tex_filt, TexAddr tex_addr) -> Result<u64> {
  auto res_desc = CUDA_RESOURCE_DESC{};
  res_desc.resType = CU_RESOURCE_TYPE_ARRAY;
  res_desc.res.array.hArray = arr;
  auto tex_desc = CUDA_TEXTURE_DESC{};
  tex_desc.addressMode[0] = CUaddress_mode(tex_addr);
  tex_desc.addressMode[1] = CUaddress_mode(tex_addr);
  tex_desc.addressMode[2] = CUaddress_mode(tex_addr);
  tex_desc.filterMode = CUfilter_mode(tex_filt);
  auto tex = CUtexObject{};
  if (auto err = cuTexObjectCreate(&tex, &res_desc, &tex_desc, nullptr)) {
    return Error(err);
  }

  return u64{tex};
}

auto texture_del(u64 tex) -> Result<> {
  if (auto err = cuTexObjectDestroy(CUtexObject(tex))) {
    return Error(err);
  }
  return Ok{};
}

template <class T, u32 N>
auto check_strides(const math::NdView<T, N>& view) -> bool {
  u32 strides[N] = {1};
  for (auto i = 1U; i < N; ++i) {
    strides[i] = strides[i - 1] * view.shape(i - 1);
  }

  for (auto i = 0U; i < N; ++i) {
    if (strides[i] != view._strides[i]) {
      return false;
    }
  }
  return true;
}

}  // namespace detail

template <class T, u32 N>
Texture<T, N>::Texture() noexcept {}

template <class T, u32 N>
Texture<T, N>::~Texture() noexcept {
  if (_tex == 0) {
    return;
  }

  detail::texture_del(_tex).unwrap();
  _tex = {};
}

template <class T, u32 N>
Texture<T, N>::Texture(Texture&& other) noexcept : _tex{mem::take(other._tex)}, _arr{mem::move(other._arr)} {}

template <class T, u32 N>
auto Texture<T, N>::operator=(Texture&& other) noexcept -> Texture& {
  if (this != &other) {
    mem::swap(_tex, other._tex);
    mem::swap(_arr, other._arr);
  }
  return *this;
}

template <class T, u32 N>
auto Texture<T, N>::new_(const u32 (&shape)[N], TexFilt filt, TexAddr addr) -> Texture {
  const auto ext = Extent::from_shape(shape);
  auto arr = Arr::new_(ext);
  auto tex = detail::texture_new(arr.as_ptr(), filt, addr).unwrap();

  auto res = Texture{};
  res._arr = mem::move(arr);
  res._tex = tex;
  return res;
}

template <class T, u32 N>
LTexture<T, N>::LTexture() noexcept {}

template <class T, u32 N>
LTexture<T, N>::~LTexture() noexcept {
  if (_tex == 0) {
    return;
  }
  detail::texture_del(_tex).unwrap();
}

template <class T, u32 N>
LTexture<T, N>::LTexture(LTexture&& other) noexcept : _tex{mem::take(other._tex)}, _arr{mem::move(other._arr)} {}

template <class T, u32 N>
auto LTexture<T, N>::operator=(LTexture&& other) noexcept -> LTexture& {
  if (this != &other) {
    mem::swap(_tex, other._tex);
    mem::swap(_arr, other._arr);
  }
  return *this;
}

template <class T, u32 N>
auto LTexture<T, N>::new_(const u32 (&shape)[N], TexFilt filt, TexAddr addr) -> LTexture {
  const auto ext = Extent::from_shape(shape);
  auto arr = Arr::new_layered(ext);
  auto tex = detail::texture_new(arr.as_ptr(), filt, addr).unwrap();

  auto res = LTexture{};
  res._arr = mem::move(arr);
  res._tex = tex;
  return res;
}

template <class T, u32 N>
auto LTexture<T, N>::set_data(math::NdView<T, N> view) -> Result<> {
  if (!detail::check_strides(view)) {
    return Error(CUDA_ERROR_INVALID_VALUE);
  }

  const auto dst_extent = _arr.extent();
  const auto src_extent = Extent::from_shape(view._shape);
  if (dst_extent != src_extent) {
    return Error(CUDA_ERROR_INVALID_VALUE);
  }

  return detail::array_set(_arr.as_ptr(), view._ptr, cuda::stream_current());
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
