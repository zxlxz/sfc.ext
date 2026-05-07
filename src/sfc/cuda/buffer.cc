#include <cuda.h>

#include "sfc/cuda/buffer.h"
#include "sfc/cuda/stream.h"
#include "sfc/cuda/error.h"

namespace sfc::cuda {

#define CU_TRY(expr)       \
  if (auto err = (expr)) { \
    throw Error{err};      \
  }

static auto array_fmt_cast(BufFmt fmt) -> CUarray_format {
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

static auto array_fmt_size(CUarray_format fmt) -> unsigned {
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
  CU_TRY(::cuArray3DCreate_v2(&res, &desc));
  return res;
}

void buffer_del(buf_t arr) {
  if (arr == nullptr) return;
  CU_TRY(::cuArrayDestroy(arr));
}

void buffer_set(buf_t arr, const void* src) {
  if (arr == nullptr || src == nullptr) {
    throw Error{CUDA_ERROR_INVALID_VALUE};
  }

  auto desc = CUDA_ARRAY3D_DESCRIPTOR_st{};
  CU_TRY(::cuArray3DGetDescriptor_v2(&desc, arr));

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
    CU_TRY(::cuMemcpy3DAsync_v2(&copy_params, stream));
  } else {
    CU_TRY(::cuMemcpy3D_v2(&copy_params));
  }
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
