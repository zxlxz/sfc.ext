#include <cuda_runtime_api.h>

#include "sfc/core.h"
#include "sfc/cuda/mod.h"
#include "sfc/cuda/stream.h"
#include "sfc/cuda/texture.h"

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic ignored "-Wmissing-designated-field-initializers"
#endif

namespace sfc::cuda {

using buffer_t = cudaArray_t;

template <class T>
static auto pitched_ptr(const T* p, cudaExtent extent) -> cudaPitchedPtr {
  const auto pitch = extent.width * sizeof(T);
  const auto xsize = extent.width;
  const auto ysize = extent.height ? extent.height : 1;
  return cudaPitchedPtr{ptr::cast_mut(p), pitch, xsize, ysize};
}

template <class T>
static auto buffer_format() -> cudaChannelFormatDesc {
  static constexpr auto NBITS = sizeof(T) * 8;
  if constexpr (trait::uint_<T>) {
    return {NBITS, 0, 0, 0, cudaChannelFormatKindUnsigned};
  } else if constexpr (trait::sint_<T>) {
    return {NBITS, 0, 0, 0, cudaChannelFormatKindSigned};
  } else if constexpr (trait::float_<T>) {
    return {NBITS, 0, 0, 0, cudaChannelFormatKindFloat};
  } else {
    static_assert(false, "unsupported type");
  }
}

template <class T>
static auto buffer_new(cudaExtent ext, u32 flags) -> Result<buffer_t> {
  const auto desc = cuda::buffer_format<T>();
  auto res = buffer_t{nullptr};
  if (auto err = cudaMalloc3DArray(&res, &desc, ext, flags); err != cudaSuccess) {
    return Error(err);
  }

  return Ok{res};
}

static auto buffer_del(buffer_t arr) -> Result<> {
  if (arr == nullptr) {
    return Ok{};
  }

  if (auto err = cudaFreeArray(arr); err != cudaSuccess) {
    return Error(err);
  }

  return Ok{};
}

static auto buffer_ext(buffer_t arr) -> Result<cudaExtent> {
  if (arr == nullptr) {
    return Error(cudaErrorInvalidValue);
  }

  auto desc = cudaChannelFormatDesc{};
  auto ext = cudaExtent{};
  auto flags = 0U;
  if (auto err = cudaArrayGetInfo(&desc, &ext, &flags, arr); err != cudaSuccess) {
    return Error(err);
  }
  return Ok{ext};
}

template <class T>
static auto buffer_set(buffer_t arr, const T* src) -> Result<> {
  if (arr == nullptr || src == nullptr) {
    return Error(cudaErrorInvalidValue);
  }

  const auto ext = _TRY(buffer_ext(arr));

  auto copy_params = cudaMemcpy3DParms{};
  copy_params.srcPtr = cuda::pitched_ptr(src, ext);
  copy_params.dstArray = arr;
  copy_params.extent = ext;  // array element count
  copy_params.kind = cudaMemcpyHostToDevice;

  const auto stream = cuda::stream_current();
  const auto err_code = stream  //
                            ? cudaMemcpy3DAsync(&copy_params, stream)
                            : cudaMemcpy3D(&copy_params);

  if (err_code != cudaSuccess) {
    return Error(err_code);
  }

  return Ok{};
}

static auto texture_new(buffer_t arr, TexFilt tex_filt, TexAddr tex_addr) -> Result<u64> {
  const auto filt_mode = cudaTextureFilterMode(tex_filt);
  const auto addr_mode = cudaTextureAddressMode(tex_addr);

  const auto res_desc = cudaResourceDesc{
      .resType = cudaResourceTypeArray,
      .res = {.array = {.array = arr}},
  };

  const auto tex_desc = cudaTextureDesc{
      .addressMode = {addr_mode, addr_mode, addr_mode},
      .filterMode = filt_mode,
  };

  auto tex = cudaTextureObject_t{};
  if (auto err = cudaCreateTextureObject(&tex, &res_desc, &tex_desc, nullptr); err != cudaSuccess) {
    return Error(err);
  }

  return Ok{tex};
}

static auto texture_del(u64 tex) -> Result<> {
  if (auto err = cudaDestroyTextureObject(tex); err != cudaSuccess) {
    return Error(err);
  }
  return Ok{};
}

template <class T>
Buffer<T>::Buffer() noexcept : _arr{nullptr} {}

template <class T>
Buffer<T>::~Buffer() {
  if (_arr == nullptr) {
    return;
  }

  cuda::buffer_del(_arr).unwrap();
}

template <class T>
Buffer<T>::Buffer(Buffer&& other) noexcept : _arr{other._arr} {
  other._arr = nullptr;
}

template <class T>
auto Buffer<T>::operator=(Buffer&& other) noexcept -> Buffer& {
  if (this == &other) return *this;
  mem::swap(_arr, other._arr);
  return *this;
}

template <class T>
auto Buffer<T>::new_(Extent ext) -> Buffer {
  const auto cu_ext = cudaExtent{ext.x, ext.y, ext.z};

  auto buf = cuda::buffer_new<T>(cu_ext, cudaArrayDefault).unwrap();
  auto res = Buffer{};
  res._arr = buf;
  return res;
}

template <class T>
auto Buffer<T>::new_layered(Extent ext) -> Buffer {
  const auto cu_ext = cudaExtent{ext.x, ext.y, ext.z};

  auto buf = cuda::buffer_new<T>(cu_ext, cudaArrayLayered).unwrap();
  auto res = Buffer{};
  res._arr = buf;
  return res;
}

template <class T>
auto Buffer<T>::as_ptr() const -> buf_t {
  return _arr;
}

template <class T>
auto Buffer<T>::set_data(const T* src) -> Result<> {
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
auto Texture<T, N>::set_data(math::NdSlice<T, N> src) -> Result<> {
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
auto LTexture<T, N>::set_data(math::NdSlice<T, N> src) -> Result<> {
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
