
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

  void* ptr = nullptr;
  CHECK_RET(cuMemAllocHost_v2, &ptr, size);
  return ptr;
}

void host_free(void* ptr) {
  if (ptr == nullptr) return;

  CHECK_RET(cuMemFreeHost, ptr);
}

auto device_alloc(usize size) -> void* {
  if (size == 0) return nullptr;

  CUdeviceptr dptr = 0;
  CHECK_RET(cuMemAlloc_v2, &dptr, size);
  return reinterpret_cast<void*>(dptr);
}

void device_free(void* ptr) {
  if (ptr == nullptr) return;

  const auto dptr = reinterpret_cast<CUdeviceptr>(ptr);
  CHECK_RET(cuMemFree_v2, dptr);
}

auto managed_alloc(usize size) -> void* {
  if (size == 0) return nullptr;

  CUdeviceptr dptr = 0;
  CHECK_RET(cuMemAllocManaged, &dptr, size, CU_MEM_ATTACH_GLOBAL);
  return reinterpret_cast<void*>(dptr);
}

void managed_free(void* ptr) {
  if (ptr == nullptr) return;

  const auto dptr = reinterpret_cast<CUdeviceptr>(ptr);
  CHECK_RET(cuMemFree_v2, dptr);
}

void prefetch_cpu(void* ptr, usize size) {
  if (ptr == nullptr || size == 0) return;

  const auto dptr = reinterpret_cast<CUdeviceptr>(ptr);
  const auto mloc = CUmemLocation{CU_MEM_LOCATION_TYPE_HOST, {0}};
  const auto stream = cuda::stream_get();
  CHECK_RET(cuMemPrefetchAsync_v2, dptr, size, mloc, 0, stream);
}

void prefetch_gpu(void* ptr, usize size) {
  if (ptr == nullptr || size == 0) return;

  const auto dev = Device::current();
  const auto dptr = reinterpret_cast<CUdeviceptr>(ptr);
  const auto mloc = CUmemLocation{CU_MEM_LOCATION_TYPE_DEVICE, {dev.id}};
  const auto stream = cuda::stream_get();
  CHECK_RET(cuMemPrefetchAsync_v2, dptr, size, mloc, 0, stream);
}

template <class T>
static auto get_ptr_attr(const void* ptr, CUpointer_attribute attr_kind) -> T {
  const auto dptr = reinterpret_cast<CUdeviceptr>(ptr);

  auto attr_val = T{};
  CHECK_RET(cuPointerGetAttribute, &attr_val, attr_kind, dptr);
  return attr_val;
}

auto get_mem_type(const void* ptr) -> CUmemorytype {
  if (ptr == nullptr) return CU_MEMORYTYPE_HOST;  // Host

  const auto res = cuda::get_ptr_attr<CUmemorytype>(ptr, CU_POINTER_ATTRIBUTE_MEMORY_TYPE);
  return res;
}

void cuda_memset(const void* ptr, u8 byte, usize size) {
  if (size == 0) return;
  if (ptr == nullptr) return;

  const auto dptr = reinterpret_cast<CUdeviceptr>(ptr);
  const auto stream = cuda::stream_get();

  if (dptr % 4 == 0 && size % 4 == 0) {
    const auto cnt = size / 4;
    const auto s32 = byte * 0x01010101U;
    if (stream) {
      CHECK_RET(cuMemsetD32Async, dptr, s32, cnt, stream);
    } else {
      CHECK_RET(cuMemsetD32_v2, dptr, s32, cnt);
    }
  }

  if (dptr % 2 == 0 && size % 2 == 0) {
    const auto cnt = size / 2;
    const auto s16 = u16(byte * 0x0101U);
    if (stream) {
      CHECK_RET(cuMemsetD16Async, dptr, s16, cnt, stream);
    } else {
      CHECK_RET(cuMemsetD16_v2, dptr, s16, cnt);
    }
  }

  const auto cnt = size;
  const auto s8 = byte;
  if (stream) {
    CHECK_RET(cuMemsetD8Async, dptr, s8, cnt, stream);
  } else {
    CHECK_RET(cuMemsetD8_v2, dptr, s8, cnt);
  }
}

void cuda_memcpy(void* dst, const void* src, usize size) {
  if (size == 0) return;

  const auto dptr = reinterpret_cast<CUdeviceptr>(dst);
  const auto sptr = reinterpret_cast<CUdeviceptr>(src);
  const auto stream = cuda::stream_get();
  if (stream) {
    CHECK_RET(cuMemcpyAsync, dptr, sptr, size, stream);
  } else {
    CHECK_RET(cuMemcpy, dptr, sptr, size);
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
      cuda::cuda_memset(ptr, val, size);
      break;
    }
  }
}

void copy_bytes(MemBlock src, MemBlock dst) {
  if (src.size == 0) return;
  if (src.ptr == nullptr) return;
  if (dst.ptr == nullptr) return;
  sfc::assert_fmt(src.size == dst.size, "src.size(=`{}`) not equal to dst.size(=`{}`)", src.size, dst.size);

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
