#include <cuda.h>

#include "sfc/alloc.h"
#include "sfc/cuda/mod.h"
#include "sfc/cuda/memory.h"
#include "sfc/cuda/stream.h"
#include "sfc/cuda/device.h"

namespace sfc::cuda {

auto to_str(MemKind kind) -> str::Str {
  switch (kind) {
    case MemKind::Heap:    return "Heap";
    case MemKind::Host:    return "Host";
    case MemKind::Device:  return "Device";
    case MemKind::Managed: return "Managed";
  }
}

void MemLocation::fmt(fmt::Formatter& f) const {
  f.write_fmt("{}:{}", to_str(kind), device);
}

auto MemLocation::Heap() -> MemLocation {
  return {.kind = MemKind::Heap, .device = 0};
}

auto MemLocation::Host() -> MemLocation {
  return {.kind = MemKind::Host, .device = 0};
}

auto MemLocation::Device(u32 device) -> MemLocation {
  return {.kind = MemKind::Device, .device = device};
}

auto MemLocation::Managed(u32 device) -> MemLocation {
  return {.kind = MemKind::Managed, .device = device};
}

auto MemLocation::allocate(mem::Layout layout) -> void* {
  const auto size = layout.size;
  if (layout.size == 0) {
    return nullptr;
  }

  switch (kind) {
    case MemKind::Heap: {
      return alloc::Global::allocate(layout);
      break;
    }
    case MemKind::Host: {
      auto scope = cuda::Device{device}.scope();
      void* ptr = nullptr;
      if (auto err = cuMemHostAlloc(&ptr, size, CU_MEMHOSTALLOC_PORTABLE)) {
        return result::Result<void*, Error>{Err{Error(err)}}.unwrap();
      }
      return ptr;
    }
    case MemKind::Device: {
      auto scope = cuda::Device{device}.scope();
      CUdeviceptr ptr = 0;
      if (auto err = cuMemAlloc(&ptr, size)) {
        return result::Result<void*, Error>{Err{Error(err)}}.unwrap();
      }
      return reinterpret_cast<void*>(static_cast<uintptr_t>(ptr));
    }
    case MemKind::Managed: {
      auto scope = cuda::Device{device}.scope();
      CUdeviceptr ptr = 0;
      if (auto err = cuMemAllocManaged(&ptr, size, CU_MEM_ATTACH_GLOBAL)) {
        return result::Result<void*, Error>{Err{Error(err)}}.unwrap();
      }
      return reinterpret_cast<void*>(static_cast<uintptr_t>(ptr));
    }
  }
  return nullptr;
}

void MemLocation::deallocate(void* ptr, mem::Layout layout) {
  if (ptr == nullptr) {
    return;
  }

  switch (kind) {
    case MemKind::Heap: alloc::Global::deallocate(ptr, layout); break;
    case MemKind::Host: {
      auto scope = cuda::Device{device}.scope();
      cuMemFreeHost(ptr);
      break;
    }
    case MemKind::Device:
    case MemKind::Managed: {
      auto scope = cuda::Device{device}.scope();
      cuMemFree_v2(CUdeviceptr(ptr));
      break;
    }
  }
}

auto MemLocation::pool() const -> mem_pool::Pool& {
  using Pool = mem_pool::XPool<MemLocation>;

  static Pool ram_pool = Pool{Host()};
  static Pool gpu_pool[8] = {
      Pool{Device(0)},
      Pool{Device(1)},
      Pool{Device(2)},
      Pool{Device(3)},
      Pool{Device(4)},
      Pool{Device(5)},
      Pool{Device(6)},
      Pool{Device(7)},
  };
  static Pool uva_pool[8] = {
      Pool{Managed(0)},
      Pool{Managed(1)},
      Pool{Managed(2)},
      Pool{Managed(3)},
      Pool{Managed(4)},
      Pool{Managed(5)},
      Pool{Managed(6)},
      Pool{Managed(7)},
  };

  if (device >= kMaxDeviceCnt) {
    return Pool::global();
  }

  switch (kind) {
    case MemKind::Heap:    return Pool::global();
    case MemKind::Host:    return ram_pool;
    case MemKind::Device:  return gpu_pool[device];
    case MemKind::Managed: return uva_pool[device];
  }

  return Pool::global();
}

auto fill_bytes(void* ptr, u8 val, usize size) -> Result<> {
  if (size == 0) {
    return Ok{};
  }

  if (ptr == nullptr) {
    return Error(CUDA_ERROR_INVALID_VALUE);
  }

  const auto device_ptr = CUdeviceptr(ptr);
  const auto stream = cuda::stream_current();
  if (auto err = cuMemsetD8Async(device_ptr, val, size, stream)) {
    return Error(err);
  }

  return Ok{};
}

auto copy_bytes(const void* src, void* dst, usize size) -> Result<> {
  if (size == 0) {
    return Ok{};
  }

  if (src == nullptr || dst == nullptr) {
    return Error(CUDA_ERROR_INVALID_VALUE);
  }

  const auto d_src = CUdeviceptr(src);
  const auto d_dst = CUdeviceptr(dst);

  const auto stream = cuda::stream_current();
  if (auto err = cuMemcpyAsync(d_dst, d_src, size, stream); err != CUDA_SUCCESS) {
    return Error(err);
  }

  return Ok{};
}

}  // namespace sfc::cuda
