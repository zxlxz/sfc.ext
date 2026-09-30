#include <cuda.h>

#include "sfc/cuda/array.h"
#include "sfc/cuda/stream.h"

namespace sfc::cuda {

namespace detail {

using arr_t = CUarray;

template <class T>
auto array_fmt() -> CUarray_format {
  if constexpr (trait::uint_<T>) {
    if constexpr (sizeof(T) == 1) return CU_AD_FORMAT_UNSIGNED_INT8;
    if constexpr (sizeof(T) == 2) return CU_AD_FORMAT_UNSIGNED_INT16;
    return CU_AD_FORMAT_UNSIGNED_INT32;
  } else if constexpr (trait::sint_<T>) {
    if constexpr (sizeof(T) == 1) return CU_AD_FORMAT_SIGNED_INT8;
    if constexpr (sizeof(T) == 2) return CU_AD_FORMAT_SIGNED_INT16;
    return CU_AD_FORMAT_SIGNED_INT32;
  } else if constexpr (trait::flt_<T>) {
    if constexpr (sizeof(T) == 4) return CU_AD_FORMAT_FLOAT;
    static_assert(sizeof(T) == 4, "unsupported floating-point texture type");
  } else {
    static_assert(false, "unsupported type");
  }
}

template <class T>
auto array_new(Extent ext, u32 flags) -> Result<arr_t> {
  const auto desc = CUDA_ARRAY3D_DESCRIPTOR{
      .Width = ext.x,
      .Height = ext.y,
      .Depth = ext.z,
      .Format = detail::array_fmt<T>(),
      .NumChannels = 1,
      .Flags = flags,
  };
  auto res = arr_t{nullptr};
  if (auto err = cuArray3DCreate_v2(&res, &desc)) {
    return Error(err);
  }

  return Ok{res};
}

auto array_del(arr_t arr) -> Result<> {
  if (arr == nullptr) {
    return Ok{};
  }

  if (auto err = cuArrayDestroy(arr)) {
    return Error(err);
  }

  return Ok{};
}

auto array_desc(arr_t arr) -> Result<CUDA_ARRAY3D_DESCRIPTOR> {
  if (arr == nullptr) {
    return Error(CUDA_ERROR_INVALID_VALUE);
  }

  auto desc = CUDA_ARRAY3D_DESCRIPTOR{};
  if (auto err = cuArray3DGetDescriptor_v2(&desc, arr)) {
    return Error(err);
  }
  return desc;
}

auto array_extent(arr_t arr) -> Result<Extent> {
  const auto desc = _TRY(detail::array_desc(arr));
  const auto extent = Extent{desc.Width, desc.Height, desc.Depth};
  return extent;
}

template <class T>
auto array_set(arr_t arr, const T* src, CUstream stream) -> Result<> {
  if (arr == nullptr || src == nullptr) {
    return Error(CUDA_ERROR_INVALID_VALUE);
  }

  const auto desc = _TRY(detail::array_desc(arr));
  auto copy_params = CUDA_MEMCPY3D{};
  copy_params.srcMemoryType = CU_MEMORYTYPE_HOST;
  copy_params.srcHost = src;
  copy_params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
  copy_params.dstArray = arr;
  copy_params.WidthInBytes = desc.Width * sizeof(T);
  copy_params.Height = cmp::max(desc.Height, 1uz);
  copy_params.Depth = cmp::max(desc.Depth, 1uz);

  if (auto err = cuMemcpy3DAsync_v2(&copy_params, stream)) {
    return Error(err);
  }

  return Ok{};
}

template <class T, u32 N>
auto is_f_contiguous(const math::NdView<T, N>& view) -> bool {
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

template <class T>
Array<T>::Array() noexcept : _arr{nullptr} {}

template <class T>
Array<T>::~Array() {
  if (_arr == nullptr) {
    return;
  }

  detail::array_del(_arr).unwrap();
}

template <class T>
Array<T>::Array(Array&& other) noexcept : _arr{mem::take(other._arr)} {}

template <class T>
auto Array<T>::operator=(Array&& other) noexcept -> Array& {
  if (this != &other) {
    mem::swap(_arr, other._arr);
  }
  return *this;
}

template <class T>
auto Array<T>::new_(Extent ext) -> Array {
  auto res = Array{};
  res._arr = detail::array_new<T>(ext, 0).unwrap_or(nullptr);
  return res;
}

template <class T>
auto Array<T>::new_layered(Extent ext) -> Array {
  auto res = Array{};
  res._arr = detail::array_new<T>(ext, CUDA_ARRAY3D_LAYERED).unwrap_or(nullptr);
  return res;
}

template <class T>
auto Array<T>::as_ptr() const -> arr_t {
  return _arr;
}

template <class T>
auto Array<T>::extent() const -> Extent {
  const auto ext = detail::array_extent(_arr).unwrap_or({});
  return ext;
}

template <class T>
template <u32 N>
auto Array<T>::set_data(math::NdView<T, N> view) -> Result<> {
  // 1. view should be 'F' order
  if (!detail::is_f_contiguous(view)) {
    return Error(CUDA_ERROR_INVALID_VALUE);
  }

  // 2. check extent
  const auto dst_ext = this->extent();
  const auto src_ext = Extent::from_shape(view._shape);
  if (dst_ext != src_ext) {
    return Error(CUDA_ERROR_INVALID_VALUE);
  }

  // 3. do copy
  const auto stream = cuda::stream_current();
  return detail::array_set(_arr, view._data, stream);
}

#define IMPL_ARRAY(T)  \
template class Array<T>;\
template auto Array<T>::set_data(math::NdView<T, 1> view) -> Result<>; \
template auto Array<T>::set_data(math::NdView<T, 2> view) -> Result<>; \
template auto Array<T>::set_data(math::NdView<T, 3> view) -> Result<>;

IMPL_ARRAY(u8);
IMPL_ARRAY(u16);
IMPL_ARRAY(u32);

IMPL_ARRAY(i8);
IMPL_ARRAY(i16);
IMPL_ARRAY(i32);

IMPL_ARRAY(f32);
#undef IMPL_ARRAY

}  // namespace sfc::cuda
