#include <cuda.h>

#include "sfc/cuda/buffer.h"
#include "sfc/cuda/stream.h"
#include "sfc/cuda/error.h"

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic ignored "-Wmissing-designated-field-initializers"
#endif

namespace sfc::cuda {

using array_fmt_t = CUarray_format;

template <class T>
static auto array_format() -> array_fmt_t {
  if constexpr (trait::uint_<T>) {
    if constexpr (sizeof(T) == 1) return CU_AD_FORMAT_UNSIGNED_INT8;
    if constexpr (sizeof(T) == 2) return CU_AD_FORMAT_UNSIGNED_INT16;
    if constexpr (sizeof(T) == 4) return CU_AD_FORMAT_UNSIGNED_INT32;
  } else if constexpr (trait::sint_<T>) {
    if constexpr (sizeof(T) == 1) return CU_AD_FORMAT_SIGNED_INT8;
    if constexpr (sizeof(T) == 2) return CU_AD_FORMAT_SIGNED_INT16;
    if constexpr (sizeof(T) == 4) return CU_AD_FORMAT_SIGNED_INT32;
  } else if constexpr (trait::float_<T>) {
    if constexpr (sizeof(T) == 2) return CU_AD_FORMAT_HALF;
    if constexpr (sizeof(T) == 4) return CU_AD_FORMAT_FLOAT;
  } else {
    static_assert(false, "unsupported type");
  }
}

static auto array_format_size(array_fmt_t fmt) -> unsigned {
  switch (fmt) {
    case CU_AD_FORMAT_UNSIGNED_INT8:  return 1;
    case CU_AD_FORMAT_UNSIGNED_INT16: return 2;
    case CU_AD_FORMAT_UNSIGNED_INT32: return 4;
    case CU_AD_FORMAT_SIGNED_INT8:    return 1;
    case CU_AD_FORMAT_SIGNED_INT16:   return 2;
    case CU_AD_FORMAT_SIGNED_INT32:   return 4;
    case CU_AD_FORMAT_HALF:           return 2;
    case CU_AD_FORMAT_FLOAT:          return 4;
    default:  // unsupported/unknown format
      throw cuda::Error{CUDA_ERROR_INVALID_VALUE};
  }
}

auto buffer_new_imp(array_fmt_t format, Extent ext, bool is_layered) -> buf_t {
  const auto flags = is_layered ? CUDA_ARRAY3D_LAYERED : 0U;
  const auto desc = CUDA_ARRAY3D_DESCRIPTOR_st{
      .Width = ext.width,
      .Height = ext.height,
      .Depth = ext.depth,
      .Format = format,
      .NumChannels = 1,
      .Flags = flags,
  };

  auto res = buf_t{nullptr};
  if (auto err = ::cuArray3DCreate_v2(&res, &desc)) {
    panic::panic_fmt("cuArray3DCreate failed, err={}", Error{err});
  }
  return res;
}

template <class T>
auto buffer_new(Extent ext, bool is_layered) -> buf_t {
  const auto format = cuda::array_format<T>();
  return buffer_new_imp(format, ext, is_layered);
}

void buffer_del(buf_t arr) {
  if (arr == nullptr) return;
  if (auto err = ::cuArrayDestroy(arr)) {
    panic::panic_fmt("cuArrayDestroy failed, err={}", Error{err});
  }
}

void buffer_set(buf_t arr, const void* src) {
  if (arr == nullptr || src == nullptr) {
    throw Error{CUDA_ERROR_INVALID_VALUE};
  }

  auto desc = CUDA_ARRAY3D_DESCRIPTOR_st{};
  if (auto err = ::cuArray3DGetDescriptor_v2(&desc, arr)) {
    panic::panic_fmt("cuArray3DGetDescriptor failed, err={}", Error{err});
  }

  const auto fmt_size = cuda::array_format_size(desc.Format);  // in bytes

  auto copy_params = CUDA_MEMCPY3D_st{
      .srcXInBytes = 0,
      .srcY = 0,
      .srcZ = 0,
      .srcLOD = 0,
      .srcMemoryType = CU_MEMORYTYPE_HOST,
      .srcHost = src,
      .srcPitch = desc.Width * fmt_size,
      .srcHeight = desc.Height,
      .dstXInBytes = 0,
      .dstY = 0,
      .dstZ = 0,
      .dstLOD = 0,
      .dstMemoryType = CU_MEMORYTYPE_ARRAY,
      .dstArray = arr,
      .WidthInBytes = desc.Width * fmt_size,
      .Height = desc.Height,
      .Depth = desc.Depth,
  };

  const auto stream = cuda::stream_get();
  if (stream) {
    if (auto err = ::cuMemcpy3DAsync_v2(&copy_params, stream)) {
      panic::panic_fmt("cuMemcpy3DAsync failed, err={}", Error{err});
    }
  } else {
    if (auto err = ::cuMemcpy3D_v2(&copy_params)) {
      panic::panic_fmt("cuMemcpy3D failed, err={}", Error{err});
    }
  }
}

template <class T>
Buffer<T>::Buffer() noexcept : _arr{nullptr} {}

template <class T>
Buffer<T>::~Buffer() {
  cuda::buffer_del(_arr);
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
auto Buffer<T>::xnew(Extent ext) -> Buffer {
  auto res = Buffer{};
  res._arr = cuda::buffer_new<T>(ext, false);
  return res;
}

template <class T>
auto Buffer<T>::as_ptr() const -> buf_t {
  return _arr;
}

template <class T>
void Buffer<T>::set_data(const T* src) {
  cuda::buffer_set(_arr, src);
}

template class Buffer<u8>;
template class Buffer<u16>;
template class Buffer<u32>;

template class Buffer<i8>;
template class Buffer<i16>;
template class Buffer<i32>;

template class Buffer<f32>;
}  // namespace sfc::cuda
