
#include <string.h>
#include "sfc/alloc.h"
#include "sfc/cuda/mod.inl"
#include "sfc/cuda/memory.h"
#include "sfc/cuda/stream.h"
#include "sfc/cuda/device.h"

namespace sfc::cuda {

struct alignas(256) SimdBlock {
  u8 _data[256];
};

auto heap_alloc(usize size) -> void* {
  using A = alloc::Global;
  const auto p = A::allocate({size, alignof(SimdBlock)});
  return p;
}

void heap_free(void* ptr) {
  using A = alloc::Global;
  A::deallocate(ptr, {0, alignof(SimdBlock)});
}

auto host_alloc(usize size) -> void* {
  if (size == 0) {
    return nullptr;
  }

  const auto flags = u32{cudaHostAllocDefault};
  void* ptr = nullptr;
  CHECK_RET(cudaHostAlloc, &ptr, size, flags);
  return ptr;
}

void host_free(void* ptr) {
  if (ptr == nullptr) {
    return;
  }

  CHECK_RET(cudaFreeHost, ptr);
}

auto device_alloc(usize size) -> void* {
  if (size == 0) {
    return nullptr;
  }

  void* ptr = nullptr;
  CHECK_RET(cudaMalloc, &ptr, size);
  return ptr;
}

void device_free(void* ptr) {
  if (ptr == nullptr) {
    return;
  }

  CHECK_RET(cudaFree, ptr);
}

auto managed_alloc(usize size) -> void* {
  if (size == 0) {
    return nullptr;
  }

  void* ptr = nullptr;
  CHECK_RET(cudaMallocManaged, &ptr, size, u32{cudaMemAttachGlobal});
  return ptr;
}

void managed_free(void* ptr) {
  if (ptr == nullptr) {
    return;
  }

  CHECK_RET(cudaFree, ptr);
}

void prefetch_cpu(void* ptr, usize size) {
  if (ptr == nullptr || size == 0) {
    return;
  }

  const auto loc = cudaMemLocation{cudaMemLocationTypeHost, 0};
  const auto flags = 0U;  // must be zero now
  const auto stream = cuda::stream_get();
  CHECK_RET(cudaMemPrefetchAsync, ptr, size, loc, flags, stream);
}

void prefetch_gpu(void* ptr, usize size) {
  if (ptr == nullptr || size == 0) {
    return;
  }

  const auto dev = Device::current();
  const auto loc = cudaMemLocation{cudaMemLocationTypeDevice, num::cast_signed(dev.id)};
  const auto flags = 0U;  // must be zero now
  const auto stream = cuda::stream_get();
  CHECK_RET(cudaMemPrefetchAsync, ptr, size, loc, flags, stream);
}

void fill_bytes(void* ptr, u8 val, usize size) {
  if (ptr == nullptr || size == 0) {
    return;
  }

  const auto stream = cuda::stream_get();
  if (stream) {
    CHECK_RET(cudaMemsetAsync, ptr, val, size, stream);
  } else {
    CHECK_RET(cudaMemset, ptr, val, size);
  }
}

void copy_bytes(const void* src, void* dst, usize size) {
  if (size == 0) {
    return;
  }

  if (src == nullptr || dst == nullptr) {
    cuda::check_ret(cudaErrorInvalidValue, "cudaMemcpy");
  }

  const auto stream = cuda::stream_get();
  if (stream) {
    CHECK_RET(cudaMemcpyAsync, dst, src, size, cudaMemcpyDefault, stream);
  } else {
    CHECK_RET(cudaMemcpy, dst, src, size, cudaMemcpyDefault);
  }
}

}  // namespace sfc::cuda
