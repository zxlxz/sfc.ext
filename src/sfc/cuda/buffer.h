#pragma once

#include <cuda_runtime_api.h>
#include "sfc/core.h"

namespace sfc::cuda {

using buf_t = cudaArray_t;
using tex_t = cudaTextureObject_t;

template <class T>
class Buffer {
  buf_t _arr = nullptr;

 public:
  Buffer() noexcept;
  ~Buffer();
  Buffer(Buffer&& other) noexcept;
  Buffer& operator=(Buffer&& other) noexcept;

  static auto xnew(const u32 (&shape)[1]) -> Buffer;
  static auto xnew(const u32 (&shape)[2]) -> Buffer;
  static auto xnew(const u32 (&shape)[3]) -> Buffer;

 public:
  auto as_ptr() const -> buf_t;
  void set_data(const T* src);
};

}  // namespace sfc::cuda
