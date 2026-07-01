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
static auto pitched_ptr(const T* p, cudaExtent extent) -> cudaPitchedPtr {
  const auto pitch = extent.width * sizeof(T);
  const auto xsize = extent.width;
  const auto ysize = extent.height ? extent.height : 1;
  return cudaPitchedPtr{ptr::cast_mut(p), pitch, xsize, ysize};
}

template <class T>
static auto array_new(const u32 (&shape)[3], bool is_layered) -> buf_t {
  const auto [width, height, depth] = shape;

  if (shape[0] == 0) {
    return nullptr;
  }

  const auto desc = cuda::array_format<T>();
  const auto flags = is_layered ? u32{cudaArrayLayered} : u32{cudaArrayDefault};
  const auto extent = cudaExtent{width, height, depth};

  auto res = buf_t{nullptr};
  CHECK_RET(cudaMalloc3DArray, &res, &desc, extent, flags);
  return res;
}

static void array_del(buf_t arr) {
  if (arr == nullptr) return;
  CHECK_RET(cudaFreeArray, arr);
}

static auto array_ext(buf_t arr) -> cudaExtent {
  if (arr == nullptr) {
    return {0, 0, 0};
  }

  auto desc = cudaChannelFormatDesc{};
  auto ext = cudaExtent{};
  auto flags = 0U;
  CHECK_RET(cudaArrayGetInfo, &desc, &ext, &flags, arr);
  return ext;
}

template <class T>
static void array_set(buf_t arr, const T* src) {
  if (arr == nullptr || src == nullptr) {
    return;
  }

  const auto ext = array_ext(arr);

  auto copy_params = cudaMemcpy3DParms{};
  copy_params.srcPtr = cuda::pitched_ptr(src, ext);
  copy_params.dstArray = arr;
  copy_params.extent = ext;  // array element count
  copy_params.kind = cudaMemcpyHostToDevice;

  if (auto stream = cuda::stream_get()) {
    CHECK_RET(cudaMemcpy3DAsync, &copy_params, stream);
  } else {
    CHECK_RET(cudaMemcpy3D, &copy_params);
  }
}

template <class T>
Buffer<T>::Buffer() noexcept : _arr{nullptr} {}

template <class T>
Buffer<T>::~Buffer() {
  if (_arr == nullptr) {
    return;
  }

  cuda::array_del(_arr);
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
auto Buffer<T>::xnew(const u32 (&shape)[1]) -> Buffer {
  auto res = Buffer{};
  res._arr = cuda::array_new<T>({shape[0], 0, 0}, false);
  return res;
}

template <class T>
auto Buffer<T>::xnew(const u32 (&shape)[2]) -> Buffer {
  auto res = Buffer{};
  res._arr = cuda::array_new<T>({shape[0], shape[1], 0}, false);
  return res;
}

template <class T>
auto Buffer<T>::xnew(const u32 (&shape)[3]) -> Buffer {
  auto res = Buffer{};
  res._arr = cuda::array_new<T>({shape[0], shape[1], shape[2]}, false);
  return res;
}

template <class T>
auto Buffer<T>::as_ptr() const -> buf_t {
  return _arr;
}

template <class T>
void Buffer<T>::set_data(const T* src) {
  cuda::array_set(_arr, src);
}

template class Buffer<u8>;
template class Buffer<u16>;
template class Buffer<u32>;

template class Buffer<i8>;
template class Buffer<i16>;
template class Buffer<i32>;

template class Buffer<f32>;
}  // namespace sfc::cuda
