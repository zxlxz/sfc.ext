
#include <cuda_runtime_api.h>

#include "sfc/alloc.h"
#include "sfc/cuda/mod.h"
#include "sfc/cuda/memory.h"
#include "sfc/cuda/stream.h"
#include "sfc/cuda/device.h"

namespace sfc::cuda {

struct alignas(256) SimdBlock {
  u8 _data[256];
};

static auto heap_allocate(usize size) -> void* {
  if (size == 0) return 0;
  const auto p = alloc::Global::allocate({size, alignof(SimdBlock)});
  return p;
}

static void heap_deallocate(void* ptr) {
  if (ptr == nullptr) return;
  alloc::Global::deallocate(ptr, {0, alignof(SimdBlock)});
}

static auto host_allocate(usize size, u32 flags = cudaHostAllocDefault) -> Result<void*> {
  if (size == 0) return Ok{nullptr};
  void* ptr = nullptr;
  if (auto err = cudaHostAlloc(&ptr, size, flags)) {
    return Error(err);
  }
  return Ok{ptr};
}

static auto host_deallocate(void* ptr) -> Result<> {
  if (ptr == nullptr) return Ok{};
  if (auto err = cudaFreeHost(ptr)) {
    return Error(err);
  }
  return Ok{};
}

static auto device_allocate(usize size) -> Result<void*> {
  if (size == 0) return Ok{nullptr};
  void* ptr = nullptr;
  if (auto err = cudaMalloc(&ptr, size)) {
    return Error(err);
  }
  return Ok{ptr};
}

static auto device_deallocate(void* ptr) -> Result<> {
  if (ptr == nullptr) return Ok{};
  if (auto err = cudaFree(ptr)) {
    return Error(err);
  }
  return Ok{};
}

static auto managed_allocate(usize size, u32 flags = cudaMemAttachGlobal) -> Result<void*> {
  if (size == 0) return Ok{nullptr};
  void* ptr = nullptr;
  if (auto err = cudaMallocManaged(&ptr, size, flags)) {
    return Error(err);
  }
  return Ok{ptr};
}

static auto managed_deallocate(void* ptr) -> Result<> {
  if (ptr == nullptr) return Ok{};
  if (auto err = cudaFree(ptr)) {
    return Error(err);
  }
  return Ok{};
}

auto HeapAllocator::allocate(usize size) -> void* {
  return cuda::heap_allocate(size);
}

void HeapAllocator::deallocate(void* ptr) {
  return cuda::heap_deallocate(ptr);
}

auto HostAllocator::allocate(usize size) -> void* {
  return cuda::host_allocate(size).unwrap();
}

void HostAllocator::deallocate(void* ptr) {
  cuda::host_deallocate(ptr).unwrap();
}

auto DeviceAllocator::allocate(usize size) -> void* {
  return cuda::device_allocate(size).unwrap();
}

void DeviceAllocator::deallocate(void* ptr) {
  cuda::device_deallocate(ptr).unwrap();
}

auto ManagedAllocator::allocate(usize size) -> void* {
  return cuda::managed_allocate(size).unwrap();
}

void ManagedAllocator::deallocate(void* ptr) {
  cuda::managed_deallocate(ptr).unwrap();
}

auto prefetch_cpu(void* ptr, usize size) -> Result<> {
  if (size == 0) {
    return Ok{};
  }

  if (ptr == nullptr) {
    return Error(cudaErrorInvalidValue);
  }

  const auto loc = cudaMemLocation{cudaMemLocationTypeHost, 0};
  const auto flags = 0U;  // must be zero now
  const auto stream = cuda::stream_current();
  if (auto err = cudaMemPrefetchAsync(ptr, size, loc, flags, stream)) {
    return Error(err);
  }

  return Ok{};
}

auto prefetch_gpu(void* ptr, usize size) -> Result<> {
  if (size == 0) {
    return Ok{};
  }

  if (ptr == nullptr) {
    return Error(cudaErrorInvalidValue);
  }

  const auto dev = Device::current();
  const auto loc = cudaMemLocation{cudaMemLocationTypeDevice, static_cast<int>(dev.id)};
  const auto flags = 0U;  // must be zero now
  const auto stream = cuda::stream_current();
  if (auto err = cudaMemPrefetchAsync(ptr, size, loc, flags, stream)) {
    return Error(err);
  }

  return Ok{};
}

auto fill_bytes(void* ptr, u8 val, usize size) -> Result<> {
  if (size == 0) {
    return Ok{};
  }

  if (ptr == nullptr) {
    return Error(cudaErrorInvalidValue);
  }

  const auto stream = cuda::stream_current();
  const auto err_code = stream  //
                            ? cudaMemsetAsync(ptr, val, size, stream)
                            : cudaMemset(ptr, val, size);

  if (err_code != cudaSuccess) {
    return Error(err_code);
  }

  return Ok{};
}

auto copy_bytes(const void* src, void* dst, usize size) -> Result<> {
  if (size == 0) {
    return Ok{};
  }

  if (src == nullptr || dst == nullptr) {
    return Error(cudaErrorInvalidValue);
  }

  const auto kind = cudaMemcpyDefault;
  const auto stream = cuda::stream_current();
  const auto err_code = stream  //
                            ? cudaMemcpyAsync(dst, src, size, kind, stream)
                            : cudaMemcpy(dst, src, size, kind);

  if (err_code != cudaSuccess) {
    return Error(err_code);
  }

  return Ok{};
}

}  // namespace sfc::cuda
