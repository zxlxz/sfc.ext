
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
  const auto a = alloc::Global{};
  const auto p = a.alloc({size, alignof(SimdBlock)});
  return p;
}

void heap_free(void* ptr) {
  const auto a = alloc::Global{};
  return a.dealloc(ptr, {0, alignof(SimdBlock)});
}

auto host_alloc(usize size) -> void* {
  if (size == 0) return nullptr;

  const auto flags = cudaHostAllocDefault;
  void* ptr = nullptr;
  CHECK_RET(cudaHostAlloc, &ptr, size, flags);
  return ptr;
}

void host_free(void* ptr) {
  if (ptr == nullptr) return;

  CHECK_RET(cudaFreeHost, ptr);
}

auto device_alloc(usize size) -> void* {
  if (size == 0) return nullptr;

  void* ptr = nullptr;
  CHECK_RET(cudaMalloc, &ptr, size);
  return ptr;
}

void device_free(void* ptr) {
  if (ptr == nullptr) return;

  CHECK_RET(cudaFree, ptr);
}

auto managed_alloc(usize size) -> void* {
  if (size == 0) return nullptr;

  void* ptr = nullptr;
  CHECK_RET(cudaMallocManaged, &ptr, size, cudaMemAttachGlobal);
  return ptr;
}

void managed_free(void* ptr) {
  if (ptr == nullptr) return;

  CHECK_RET(cudaFree, ptr);
}

void prefetch_cpu(void* ptr, usize size) {
  if (ptr == nullptr || size == 0) return;

  const auto stream = cuda::stream_get();
  const auto loc = cudaMemLocation{cudaMemLocationTypeHost, 0};
  CHECK_RET(cudaMemPrefetchAsync, ptr, size, loc, 0U, stream);
}

void prefetch_gpu(void* ptr, usize size) {
  if (ptr == nullptr || size == 0) return;

  const auto dev = Device::current();
  const auto stream = cuda::stream_get();
  const auto loc = cudaMemLocation{cudaMemLocationTypeDevice, num::cast_signed(dev.id)};
  CHECK_RET(cudaMemPrefetchAsync, ptr, size, loc, 0U, stream);
}

auto get_mem_type(const void* ptr) -> cudaMemoryType {
  if (ptr == nullptr) return cudaMemoryTypeHost;

  auto attrs = cudaPointerAttributes{};
  CHECK_RET(cudaPointerGetAttributes, &attrs, ptr);
  return attrs.type;
}

void cuda_memset(void* ptr, u8 byte, usize size) {
  if (size == 0) return;
  if (ptr == nullptr) return;

  const auto stream = cuda::stream_get();
  if (stream) {
    CHECK_RET(cudaMemsetAsync, ptr, byte, size, stream);
  } else {
    CHECK_RET(cudaMemset, ptr, byte, size);
  }
}

void cuda_memcpy(void* dst, const void* src, usize size) {
  if (size == 0) return;

  const auto stream = cuda::stream_get();
  if (stream) {
    CHECK_RET(cudaMemcpyAsync, dst, src, size, cudaMemcpyDefault, stream);
  } else {
    CHECK_RET(cudaMemcpy, dst, src, size, cudaMemcpyDefault);
  }
}

void fill_bytes(MemBlock blk, u8 val) {
  const auto ptr = blk.ptr;
  const auto size = blk.size;
  if (size == 0 || ptr == nullptr) return;

  switch (blk.mtype) {
    default:
    case MemType::Heap:
    case MemType::Host: {
      ::memset(ptr, val, size);
      break;
    }
    case MemType::Device:
    case MemType::UVA:    {
      cuda::cuda_memset(const_cast<void*>(ptr), val, size);
      break;
    }
  }
}

void copy_bytes(MemBlock src, MemBlock dst) {
  if (src.size == 0) return;
  if (src.ptr == nullptr) return;
  if (dst.ptr == nullptr) return;
  sfc::assert_fmt(src.size == dst.size, fmt::Args{"src.size(=`{}`) not equal to dst.size(=`{}`)", src.size, dst.size});

  const auto src_host = src.mtype == MemType::Heap || src.mtype == MemType::Host;
  const auto dst_host = dst.mtype == MemType::Heap || dst.mtype == MemType::Host;
  if (src_host && dst_host) {
    ::memcpy(dst.ptr, src.ptr, src.size);
  } else {
    cuda::cuda_memcpy(dst.ptr, src.ptr, src.size);
  }
}

auto Alloc::alloc(usize size) -> void* {
  if (size == 0) return nullptr;

  switch (mtype) {
    default:
    case MemType::Heap:   return cuda::heap_alloc(size);
    case MemType::Host:   return cuda::host_alloc(size);
    case MemType::Device: return cuda::device_alloc(size);
    case MemType::UVA:    return cuda::managed_alloc(size);
  }
}

void Alloc::dealloc(void* ptr) {
  if (ptr == nullptr) return;

  switch (mtype) {
    default:
    case MemType::Heap:   return cuda::heap_free(ptr);
    case MemType::Host:   return cuda::host_free(ptr);
    case MemType::Device: return cuda::device_free(ptr);
    case MemType::UVA:    return cuda::managed_free(ptr);
  }
}

}  // namespace sfc::cuda
