#include "sfc/cuda/mod.inl"
#include "sfc/cuda/buffer.h"
#include "sfc/cuda/stream.h"

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic ignored "-Wmissing-designated-field-initializers"
#endif

namespace sfc::cuda {

template <class T>
static auto array_format() -> cudaChannelFormatDesc {
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
static auto pitched_ptr(const T* p, Extent ext) -> cudaPitchedPtr {
  const auto pitch = ext.width * sizeof(T);
  const auto xsize = ext.width;
  const auto ysize = ext.height ? ext.height : 1;
  return cudaPitchedPtr{ptr::cast_mut(p), pitch, xsize, ysize};
}

template <class T>
static auto buffer_new(Extent ext, bool is_layered) -> buf_t {
  if (ext.width == 0) {
    return nullptr;
  }

  const auto desc = cuda::array_format<T>();
  const auto flags = is_layered ? cudaArrayLayered : 0U;
  const auto extent = cudaExtent{ext.width, ext.height, ext.depth};

  auto res = buf_t{nullptr};
  CHECK_RET(cudaMalloc3DArray, &res, &desc, extent, flags);
  return res;
}

static void buffer_del(buf_t arr) {
  if (arr == nullptr) return;
  CHECK_RET(cudaFreeArray, arr);
}

template <class T>
static void buffer_set(buf_t arr, Extent ext, const T* src) {
  if (arr == nullptr || src == nullptr) {
    return;
  }

  const auto copy_ext = cudaExtent{
      ext.width,
      ext.height ? ext.height : 1,
      ext.depth ? ext.depth : 1,
  };

  auto copy_params = cudaMemcpy3DParms{};
  copy_params.srcPos = cudaPos{0, 0, 0};
  copy_params.srcPtr = cuda::pitched_ptr(src, ext);
  copy_params.dstPos = cudaPos{0, 0, 0};
  copy_params.dstArray = arr;
  copy_params.extent = copy_ext;  // array element count
  copy_params.kind = cudaMemcpyHostToDevice;

  const auto stream = cuda::stream_get();
  if (stream) {
    CHECK_RET(cudaMemcpy3DAsync, &copy_params, stream);
  } else {
    CHECK_RET(cudaMemcpy3D, &copy_params);
  }
}

template <class T>
Buffer<T>::Buffer() noexcept : _arr{nullptr}, _ext{} {}

template <class T>
Buffer<T>::~Buffer() {
  cuda::buffer_del(_arr);
}

template <class T>
Buffer<T>::Buffer(Buffer&& other) noexcept : _arr{other._arr}, _ext{other._ext} {
  other._arr = nullptr;
  other._ext = {};
}

template <class T>
auto Buffer<T>::operator=(Buffer&& other) noexcept -> Buffer& {
  if (this == &other) return *this;
  mem::swap(_arr, other._arr);
  mem::swap(_ext, other._ext);
  return *this;
}

template <class T>
auto Buffer<T>::xnew(Extent ext) -> Buffer {
  auto res = Buffer{};
  res._arr = cuda::buffer_new<T>(ext, false);
  res._ext = ext;
  return res;
}

template <class T>
auto Buffer<T>::as_ptr() const -> buf_t {
  return _arr;
}

template <class T>
void Buffer<T>::set_data(const T* src) {
  cuda::buffer_set(_arr, _ext, src);
}

template class Buffer<u8>;
template class Buffer<u16>;
template class Buffer<u32>;

template class Buffer<i8>;
template class Buffer<i16>;
template class Buffer<i32>;

template class Buffer<f32>;
}  // namespace sfc::cuda
