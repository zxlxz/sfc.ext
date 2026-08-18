#include <cuda.h>

#include "sfc/cuda/mod.h"
#include "sfc/cuda/stream.h"
#include "sfc/cuda/texture.h"

namespace sfc::cuda {

using buffer_t = CUarray;

template <class T>
static auto buffer_format() -> CUarray_format {
  if constexpr (trait::uint_<T>) {
    if constexpr (sizeof(T) == 1) return CU_AD_FORMAT_UNSIGNED_INT8;
    if constexpr (sizeof(T) == 2) return CU_AD_FORMAT_UNSIGNED_INT16;
    return CU_AD_FORMAT_UNSIGNED_INT32;
  } else if constexpr (trait::sint_<T>) {
    if constexpr (sizeof(T) == 1) return CU_AD_FORMAT_SIGNED_INT8;
    if constexpr (sizeof(T) == 2) return CU_AD_FORMAT_SIGNED_INT16;
    return CU_AD_FORMAT_SIGNED_INT32;
  } else if constexpr (trait::float_<T>) {
    if constexpr (sizeof(T) == 4) return CU_AD_FORMAT_FLOAT;
    static_assert(sizeof(T) == 4, "unsupported floating-point texture type");
  } else {
    static_assert(false, "unsupported type");
  }
}

template <class T>
static auto array_new(Extent ext, u32 flags) -> Result<buffer_t> {
  const auto desc = CUDA_ARRAY3D_DESCRIPTOR{
      .Width = ext.x,
      .Height = ext.y,
      .Depth = ext.z,
      .Format = cuda::buffer_format<T>(),
      .NumChannels = 1,
      .Flags = flags,
  };
  auto res = buffer_t{nullptr};
  if (auto err = cuArray3DCreate_v2(&res, &desc); err != CUDA_SUCCESS) {
    return Error(err);
  }

  return Ok{res};
}

static auto buffer_del(buffer_t arr) -> Result<> {
  if (arr == nullptr) {
    return Ok{};
  }

  if (auto err = cuArrayDestroy(arr); err != CUDA_SUCCESS) {
    return Error(err);
  }

  return Ok{};
}

static auto buffer_ext(buffer_t arr) -> Result<CUDA_ARRAY3D_DESCRIPTOR> {
  if (arr == nullptr) {
    return Error(CUDA_ERROR_INVALID_VALUE);
  }

  auto desc = CUDA_ARRAY3D_DESCRIPTOR{};
  if (auto err = cuArray3DGetDescriptor_v2(&desc, arr); err != CUDA_SUCCESS) {
    return Error(err);
  }
  return Ok{desc};
}

template <class T>
static auto buffer_set(buffer_t arr, const T* src) -> Result<> {
  if (arr == nullptr || src == nullptr) {
    return Error(CUDA_ERROR_INVALID_VALUE);
  }

  const auto desc = _TRY(buffer_ext(arr));
  auto copy_params = CUDA_MEMCPY3D{};
  copy_params.srcMemoryType = CU_MEMORYTYPE_HOST;
  copy_params.srcHost = src;
  copy_params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
  copy_params.dstArray = arr;
  copy_params.WidthInBytes = desc.Width * sizeof(T);
  copy_params.Height = desc.Height ? desc.Height : 1;
  copy_params.Depth = desc.Depth ? desc.Depth : 1;

  const auto stream = cuda::stream_current();
  if (auto err = cuMemcpy3DAsync_v2(&copy_params, stream); err != CUDA_SUCCESS) {
    return Error(err);
  }

  return Ok{};
}

static auto texture_new(buffer_t arr, TexFilt tex_filt, TexAddr tex_addr) -> Result<u64> {
  auto res_desc = CUDA_RESOURCE_DESC{};
  res_desc.resType = CU_RESOURCE_TYPE_ARRAY;
  res_desc.res.array.hArray = arr;
  auto tex_desc = CUDA_TEXTURE_DESC{};
  tex_desc.addressMode[0] = CUaddress_mode(tex_addr);
  tex_desc.addressMode[1] = CUaddress_mode(tex_addr);
  tex_desc.addressMode[2] = CUaddress_mode(tex_addr);
  tex_desc.filterMode = CUfilter_mode(tex_filt);
  auto tex = CUtexObject{};
  if (auto err = cuTexObjectCreate(&tex, &res_desc, &tex_desc, nullptr); err != CUDA_SUCCESS) {
    return Error(err);
  }

  return u64{tex};
}

static auto texture_del(u64 tex) -> Result<> {
  if (auto err = cuTexObjectDestroy(CUtexObject(tex)); err != CUDA_SUCCESS) {
    return Error(err);
  }
  return Ok{};
}

template <class T>
Array<T>::Array() noexcept : _arr{nullptr} {}

template <class T>
Array<T>::~Array() {
  if (_arr == nullptr) {
    return;
  }

  cuda::buffer_del(_arr).unwrap();
}

template <class T>
Array<T>::Array(Array&& other) noexcept : _arr{other._arr} {
  other._arr = nullptr;
}

template <class T>
auto Array<T>::operator=(Array&& other) noexcept -> Array& {
  if (this == &other) return *this;
  mem::swap(_arr, other._arr);
  return *this;
}

template <class T>
auto Array<T>::new_(Extent ext) -> Array {
  auto buf = cuda::array_new<T>(ext, 0).unwrap();
  auto res = Array{};
  res._arr = buf;
  return res;
}

template <class T>
auto Array<T>::new_layered(Extent ext) -> Array {
  auto buf = cuda::array_new<T>(ext, CUDA_ARRAY3D_LAYERED).unwrap();
  auto res = Array{};
  res._arr = buf;
  return res;
}

template <class T>
auto Array<T>::as_ptr() const -> arr_t {
  return _arr;
}

template <class T>
auto Array<T>::set_data(const T* src) -> Result<> {
  return cuda::buffer_set(_arr, src);
}

template <class T, int N>
Texture<T, N>::Texture() noexcept {}

template <class T, int N>
Texture<T, N>::~Texture() noexcept {
  if (_tex == 0) {
    return;
  }

  cuda::texture_del(_tex).unwrap();
  _tex = {};
}

template <class T, int N>
Texture<T, N>::Texture(Texture&& other) noexcept : _tex{mem::take(other._tex)}, _buf{mem::move(other._buf)} {}

template <class T, int N>
auto Texture<T, N>::operator=(Texture&& other) noexcept -> Texture& {
  if (this != &other) {
    mem::swap(_tex, other._tex);
    mem::swap(_buf, other._buf);
  }
  return *this;
}

template <class T, int N>
auto Texture<T, N>::new_(const u32 (&shape)[N], TexFilt filt, TexAddr addr) -> Texture {
  const auto ext = Extent{
      N > 0 ? shape[0] : 0,
      N > 1 ? shape[1] : 0,
      N > 2 ? shape[2] : 0,
  };
  auto buf = Buf::new_(ext);
  auto tex = cuda::texture_new(buf.as_ptr(), filt, addr).unwrap();

  auto res = Texture{};
  res._buf = mem::move(buf);
  res._tex = tex;
  return res;
}

template <class T, int N>
auto Texture<T, N>::set_data(math::NdView<T, N> src) -> Result<> {
  return _buf.set_data(src._data);
}

template <class T, int N>
LTexture<T, N>::LTexture() noexcept {}

template <class T, int N>
LTexture<T, N>::~LTexture() noexcept {
  if (_tex == 0) {
    return;
  }
  cuda::texture_del(_tex).unwrap();
}

template <class T, int N>
LTexture<T, N>::LTexture(LTexture&& other) noexcept : _tex{mem::take(other._tex)}, _buf{mem::move(other._buf)} {}

template <class T, int N>
auto LTexture<T, N>::operator=(LTexture&& other) noexcept -> LTexture& {
  if (this != &other) {
    mem::swap(_tex, other._tex);
    mem::swap(_buf, other._buf);
  }
  return *this;
}

template <class T, int N>
auto LTexture<T, N>::new_(const u32 (&shape)[N], TexFilt filt, TexAddr addr) -> LTexture {
  const auto ext = Extent{
      N > 0 ? shape[0] : 0,
      N > 1 ? shape[1] : 0,
      N > 2 ? shape[2] : 0,
  };
  auto res = LTexture{};
  res._buf = Buf::new_layered(ext);
  res._tex = cuda::texture_new(res._buf.as_ptr(), filt, addr).unwrap();
  return res;
}

template <class T, int N>
auto LTexture<T, N>::set_data(math::NdView<T, N> src) -> Result<> {
  return _buf.set_data(src._data);
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
