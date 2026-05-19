#include <cuda.h>
#include <string.h>

#include "sfc/alloc.h"
#include "sfc/cuda/memory.h"
#include "sfc/cuda/stream.h"
#include "sfc/cuda/device.h"
#include "sfc/cuda/error.h"

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
  if (auto e = ::cuMemAllocHost_v2(&ptr, size)) {
    panic::panic_fmt("cuMemAllocHost_v2 failed, err={}", Error{e});
  }
  return ptr;
}

void host_free(void* ptr) {
  if (ptr == nullptr) return;

  if (auto e = ::cuMemFreeHost(ptr)) {
    panic::panic_fmt("cuMemFreeHost failed, err={}", Error{e});
  }
}

auto device_alloc(usize size) -> void* {
  if (size == 0) return nullptr;

  CUdeviceptr dptr = 0;
  if (auto e = ::cuMemAlloc_v2(&dptr, size)) {
    panic::panic_fmt("cuMemAlloc_v2 failed, err={}", Error{e});
  }
  return reinterpret_cast<void*>(dptr);
}

void device_free(void* ptr) {
  if (ptr == nullptr) return;

  const auto dptr = reinterpret_cast<CUdeviceptr>(ptr);
  if (auto e = ::cuMemFree_v2(dptr)) {
    panic::panic_fmt("cuMemFree_v2 failed, err={}", Error{e});
  }
}

auto managed_alloc(usize size) -> void* {
  if (size == 0) return nullptr;

  CUdeviceptr dptr = 0;
  if (auto e = ::cuMemAllocManaged(&dptr, size, CU_MEM_ATTACH_GLOBAL)) {
    panic::panic_fmt("cuMemAllocManaged failed, err={}", Error{e});
  }

  return reinterpret_cast<void*>(dptr);
}

void managed_free(void* ptr) {
  if (ptr == nullptr) return;

  const auto dptr = reinterpret_cast<CUdeviceptr>(ptr);
  if (auto e = ::cuMemFree_v2(dptr)) {
    panic::panic_fmt("cuMemFree_v2 failed, err={}", Error{e});
  }
}

void prefetch_cpu(void* ptr, usize size) {
  if (ptr == nullptr || size == 0) return;

  const auto dptr = reinterpret_cast<CUdeviceptr>(ptr);
  const auto mloc = CUmemLocation{CU_MEM_LOCATION_TYPE_HOST, 0};
  const auto stream = cuda::stream_get();
  if (auto e = ::cuMemPrefetchAsync_v2(dptr, size, mloc, 0, stream)) {
    panic::panic_fmt("cuMemPrefetchAsync_v2 failed, err={}", Error{e});
  }
}

void prefetch_gpu(void* ptr, usize size) {
  if (ptr == nullptr || size == 0) return;

  const auto dev = Device::current();
  const auto dptr = reinterpret_cast<CUdeviceptr>(ptr);
  const auto mloc = CUmemLocation{CU_MEM_LOCATION_TYPE_DEVICE, dev.id};
  const auto stream = cuda::stream_get();
  if (auto e = ::cuMemPrefetchAsync_v2(dptr, size, mloc, 0, stream)) {
    panic::panic_fmt("cuMemPrefetchAsync_v2 failed, err={}", Error{e});
  }
}

template <class T>
static auto get_ptr_attr(const void* ptr, CUpointer_attribute attr_kind) -> T {
  const auto dptr = reinterpret_cast<CUdeviceptr>(ptr);

  auto attr_val = T{};
  if (auto e = ::cuPointerGetAttribute(&attr_val, attr_kind, dptr)) {
    panic::panic_fmt("cuPointerGetAttribute failed, err={}", Error{e});
  }
  return attr_val;
}

auto get_mem_type(const void* ptr) -> CUmemorytype {
  if (ptr == nullptr) return CU_MEMORYTYPE_HOST;  // Host

  const auto res = cuda::get_ptr_attr<CUmemorytype>(ptr, CU_POINTER_ATTRIBUTE_MEMORY_TYPE);
  return res;
}

void cuda_memset(const void* ptr, u8 byte, usize size) {
  if (size == 0) return;
  sfc::expect(ptr != nullptr, "ptr is null");

  const auto dptr = reinterpret_cast<CUdeviceptr>(ptr);
  const auto stream = cuda::stream_get();

  if (dptr % 4 == 0 && size % 4 == 0) {
    const auto cnt = size / 4;
    const auto s32 = static_cast<u32>(byte * 0x01010101);
    if (stream) {
      if (auto e = ::cuMemsetD32Async(dptr, s32, cnt, stream)) {
        panic::panic_fmt("cuMemsetD32Async failed, err={}", Error{e});
      }
    } else {
      if (auto e = ::cuMemsetD32_v2(dptr, s32, cnt)) {
        panic::panic_fmt("cuMemsetD32_v2 failed, err={}", Error{e});
      }
    }
  }

  if (dptr % 2 == 0 && size % 2 == 0) {
    const auto cnt = size / 2;
    const auto s16 = static_cast<u16>(byte * 0x0101);
    if (stream) {
      if (auto e = ::cuMemsetD16Async(dptr, s16, cnt, stream)) {
        panic::panic_fmt("cuMemsetD16Async failed, err={}", Error{e});
      }
    } else {
      if (auto e = ::cuMemsetD16_v2(dptr, s16, cnt)) {
        panic::panic_fmt("cuMemsetD16_v2 failed, err={}", Error{e});
      }
    }
  }

  const auto cnt = size;
  const auto s8 = static_cast<u8>(byte);
  if (stream) {
    if (auto e = ::cuMemsetD8Async(dptr, s8, cnt, stream)) {
      panic::panic_fmt("cuMemsetD8Async failed, err={}", Error{e});
    }
  } else {
    if (auto e = ::cuMemsetD8_v2(dptr, s8, cnt)) {
      panic::panic_fmt("cuMemsetD8_v2 failed, err={}", Error{e});
    }
  }
}

void cuda_memcpy(void* dst, const void* src, usize size) {
  if (size == 0) return;

  const auto dptr = reinterpret_cast<CUdeviceptr>(dst);
  const auto sptr = reinterpret_cast<CUdeviceptr>(src);
  const auto stream = cuda::stream_get();
  if (stream) {
    if (auto e = ::cuMemcpyAsync(dptr, sptr, size, stream)) {
      panic::panic_fmt("cuMemcpyAsync failed, err={}", Error{e});
    }
  } else {
    if (auto e = ::cuMemcpy(dptr, sptr, size)) {
      panic::panic_fmt("cuMemcpy failed, err={}", Error{e});
    }
  }
}

void fill_bytes(MemBlock blk, u8 val) {
  const auto ptr = blk.ptr;
  const auto size = blk.size;
  if (size == 0) return;
  sfc::expect(ptr != nullptr, "ptr is null");

  switch (blk.mtype) {
    default:
    case MemType::Heap:
    case MemType::Host: {
      ::memset(blk.ptr, val, size);
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
  sfc::expect(src.size == dst.size,
              "src.size(=`{}`) not equal to dst.size(=`{}`)",
              src.size,
              dst.size);

  if (src.size == 0) return;
  sfc::expect(src.ptr != nullptr, "src.ptr is null");
  sfc::expect(dst.ptr != nullptr, "dst.ptr is null");

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
