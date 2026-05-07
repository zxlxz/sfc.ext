#include <cuda.h>
#include <string.h>

#include "sfc/cuda/memory.h"
#include "sfc/cuda/stream.h"
#include "sfc/cuda/device.h"
#include "sfc/cuda/error.h"

#define CU_TRY(expr)       \
  if (auto err = (expr)) { \
    throw Error{err};      \
  }

namespace sfc::cuda {

auto heap_alloc(usize size) -> void* {
  if (size == 0) {
    return nullptr;
  }

  return ::malloc(size);
}

void heap_free(void* ptr) {
  if (ptr == nullptr) {
    return;
  }
  ::free(ptr);
}

auto host_alloc(usize size) -> void* {
  if (size == 0) {
    return nullptr;
  }

  void* ptr = nullptr;
  CU_TRY(::cuMemAllocHost_v2(&ptr, size));
  return ptr;
}

void host_free(void* ptr) {
  if (ptr == nullptr) {
    return;
  }

  CU_TRY(::cuMemFreeHost(ptr));
}

auto device_alloc(usize size) -> void* {
  if (size == 0) {
    return nullptr;
  }

  CUdeviceptr dptr = 0;
  CU_TRY(::cuMemAlloc_v2(&dptr, size));
  return reinterpret_cast<void*>(dptr);
}

void device_free(void* ptr) {
  if (ptr == nullptr) {
    return;
  }

  const auto dptr = reinterpret_cast<CUdeviceptr>(ptr);
  CU_TRY(::cuMemFree_v2(dptr));
}

auto managed_alloc(usize size) -> void* {
  if (size == 0) {
    return nullptr;
  }

  CUdeviceptr dptr = 0;
  CU_TRY(::cuMemAllocManaged(&dptr, size, CU_MEM_ATTACH_GLOBAL));

  return reinterpret_cast<void*>(dptr);
}

void managed_free(void* ptr) {
  if (ptr == nullptr) {
    return;
  }

  const auto dptr = reinterpret_cast<CUdeviceptr>(ptr);
  CU_TRY(::cuMemFree_v2(dptr));
}

void prefetch_cpu(void* ptr, usize size) {
  if (ptr == nullptr || size == 0) {
    return;
  }

  const auto dptr = reinterpret_cast<CUdeviceptr>(ptr);
  const auto mloc = CUmemLocation{CU_MEM_LOCATION_TYPE_HOST, 0};
  const auto stream = cuda::stream_get();
  CU_TRY(::cuMemPrefetchAsync_v2(dptr, size, mloc, 0, stream));
}

void prefetch_gpu(void* ptr, usize size) {
  if (ptr == nullptr || size == 0) {
    return;
  }

  const auto dev = Device::current();
  const auto dptr = reinterpret_cast<CUdeviceptr>(ptr);
  const auto mloc = CUmemLocation{CU_MEM_LOCATION_TYPE_DEVICE, dev.id};
  const auto stream = cuda::stream_get();
  CU_TRY(::cuMemPrefetchAsync_v2(dptr, size, mloc, 0, stream));
}

template <class T>
static auto get_ptr_attr(const void* ptr, CUpointer_attribute attr_kind) -> T {
  const auto dptr = reinterpret_cast<CUdeviceptr>(ptr);

  auto attr_val = T{};
  CU_TRY(::cuPointerGetAttribute(&attr_val, attr_kind, dptr));
  return attr_val;
}

auto get_mem_type(const void* ptr) -> CUmemorytype {
  if (ptr == nullptr) {
    return CU_MEMORYTYPE_HOST;  // Host
  }

  const auto res = cuda::get_ptr_attr<CUmemorytype>(ptr, CU_POINTER_ATTRIBUTE_MEMORY_TYPE);
  return res;
}

void copy_bytes(const void* src, void* dst, usize size) {
  if (size == 0) {
    return;
  }
  if (dst == nullptr || src == nullptr) {
    throw Error{CUDA_ERROR_INVALID_VALUE};
  }

  const auto dptr = reinterpret_cast<CUdeviceptr>(dst);
  const auto sptr = reinterpret_cast<CUdeviceptr>(src);
  const auto stream = cuda::stream_get();
  if (stream) {
    CU_TRY(::cuMemcpyAsync(dptr, sptr, size, stream));
  } else {
    CU_TRY(::cuMemcpy(dptr, sptr, size));
  }
}

static void cpu_memset(const void* ptr, u8 val, usize size) {
  if (size == 0) {
    return;
  }
  if (ptr == nullptr) {
    throw Error{CUDA_ERROR_INVALID_VALUE};
  }
  ::memset(const_cast<void*>(ptr), val, size);
}

static void gpu_memset(const void* ptr, u8 byte, usize size) {
  if (size == 0) {
    return;
  }
  if (ptr == nullptr) {
    throw Error{CUDA_ERROR_INVALID_VALUE};
  }

  const auto dptr = reinterpret_cast<CUdeviceptr>(ptr);
  const auto stream = cuda::stream_get();

  if (dptr % 4 == 0 && size % 4 == 0) {
    const auto cnt = size / 4;
    const auto s32 = static_cast<u32>(byte * 0x01010101);
    if (stream) {
      CU_TRY(::cuMemsetD32Async(dptr, s32, cnt, stream));
    } else {
      CU_TRY(::cuMemsetD32_v2(dptr, s32, cnt));
    }
  }

  if (dptr % 2 == 0 && size % 2 == 0) {
    const auto cnt = size / 2;
    const auto s16 = static_cast<u16>(byte * 0x0101);
    if (stream) {
      CU_TRY(::cuMemsetD16Async(dptr, s16, cnt, stream));
    } else {
      CU_TRY(::cuMemsetD16_v2(dptr, s16, cnt));
    }
  }

  const auto cnt = size;
  const auto s8 = static_cast<u8>(byte);
  if (stream) {
    CU_TRY(::cuMemsetD8Async(dptr, s8, cnt, stream));  // async
  } else {
    CU_TRY(::cuMemsetD8_v2(dptr, s8, cnt));  // default
  }
}

void fill_bytes(void* ptr, u8 val, usize cnt) {
  if (cnt == 0) {
    return;
  }
  if (ptr == nullptr) {
    throw Error{CUDA_ERROR_INVALID_VALUE};
  }

  const auto mem_type = cuda::get_mem_type(ptr);
  switch (mem_type) {
    case CU_MEMORYTYPE_HOST: {
      cuda::cpu_memset(ptr, val, cnt);
      break;
    }
    case CU_MEMORYTYPE_DEVICE:
    case CU_MEMORYTYPE_UNIFIED: {
      cuda::gpu_memset(ptr, val, cnt);
      break;
    }
    case CU_MEMORYTYPE_ARRAY: {
      throw Error{CUDA_ERROR_NOT_SUPPORTED};  // Not supported
    }
  }
}

auto Alloc::alloc(usize size) -> void* {
  if (size == 0) {
    return nullptr;
  }

  switch (type) {
    default:
    case MemType::Heap:   return cuda::heap_alloc(size);
    case MemType::Host:   return cuda::host_alloc(size);
    case MemType::Device: return cuda::device_alloc(size);
    case MemType::UVA:    return cuda::managed_alloc(size);
  }
}

void Alloc::dealloc(void* ptr) {
  if (ptr == nullptr) {
    return;
  }

  switch (type) {
    default:
    case MemType::Heap:   return cuda::heap_free(ptr);
    case MemType::Host:   return cuda::host_free(ptr);
    case MemType::Device: return cuda::device_free(ptr);
    case MemType::UVA:    return cuda::managed_free(ptr);
  }
}

}  // namespace sfc::cuda
