#include <cuda.h>

#include "sfc/core/panic.h"
#include "sfc/math/vec.h"
#include "sfc/cuda/buffer.h"
#include "sfc/cuda/stream.h"
#include "sfc/cuda/error.h"


namespace sfc::cuda {

using array_fmt_t = CUarray_format;

template <>
auto BufExt::from(const math::vec<u32, 1>& dims) -> BufExt {
  return {dims.x, 0, 0};
}

template <>
auto BufExt::from(const math::vec<u32, 2>& dims) -> BufExt {
  return {dims.x, dims.y, 0};
}

template <>
auto BufExt::from(const math::vec<u32, 3>& dims) -> BufExt {
  return {dims.x, dims.y, dims.z};
}

static auto array_fmt_cast(BufFmt fmt) -> array_fmt_t {
  switch (fmt.kind) {
    case BufFmt::UInt:
      if (fmt.size == 1) return CU_AD_FORMAT_UNSIGNED_INT8;
      if (fmt.size == 2) return CU_AD_FORMAT_UNSIGNED_INT16;
      if (fmt.size == 4) return CU_AD_FORMAT_UNSIGNED_INT32;
      break;
    case BufFmt::SInt:
      if (fmt.size == 1) return CU_AD_FORMAT_SIGNED_INT8;
      if (fmt.size == 2) return CU_AD_FORMAT_SIGNED_INT16;
      if (fmt.size == 4) return CU_AD_FORMAT_SIGNED_INT32;
      break;
    case BufFmt::Float:
      if (fmt.size == 2) return CU_AD_FORMAT_HALF;
      if (fmt.size == 4) return CU_AD_FORMAT_FLOAT;
      break;
    default: break;
  }
  return CU_AD_FORMAT_MAX;
}

static auto array_fmt_size(array_fmt_t fmt) -> unsigned {
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

auto buffer_new(BufFmt fmt, BufExt ext, bool is_layered) -> buf_t {
  const auto flags = is_layered ? CUDA_ARRAY3D_LAYERED : 0U;
  const auto format = cuda::array_fmt_cast(fmt);

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

  const auto fmt_size = cuda::array_fmt_size(desc.Format);  // in bytes

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
auto BufFmt::of() -> BufFmt {
  return {Unknown, sizeof(T)};
}

#define IMPL_BUF_FMT(T, Kind)       \
  template <>                       \
  auto BufFmt::of<T>() -> BufFmt {  \
    return BufFmt{Kind, sizeof(T)}; \
  }

IMPL_BUF_FMT(u8, BufFmt::UInt)
IMPL_BUF_FMT(u16, BufFmt::UInt)
IMPL_BUF_FMT(u32, BufFmt::UInt)
IMPL_BUF_FMT(i8, BufFmt::SInt)
IMPL_BUF_FMT(i16, BufFmt::SInt)
IMPL_BUF_FMT(i32, BufFmt::SInt)

IMPL_BUF_FMT(float, BufFmt::Float)

#undef IMPL_BUF_FMT

}  // namespace sfc::cuda
