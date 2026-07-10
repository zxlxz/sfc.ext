
#include <cuda_runtime_api.h>

#include "sfc/alloc.h"
#include "sfc/cuda/mod.h"
#include "sfc/cuda/memory.h"
#include "sfc/cuda/stream.h"
#include "sfc/cuda/device.h"

namespace sfc::cuda {

struct HeapAllocator {
  struct alignas(256) SimdBlock {
    u8 _data[256];
  };

  static auto allocate(usize size) -> void* {
    if (size == 0) {
      return nullptr;
    }

    const auto layout = mem::Layout{size, alignof(SimdBlock)};
    const auto p = alloc::Global::allocate(layout);
    return p;
  }

  static void deallocate(void* ptr) {
    if (ptr == nullptr) {
      return;
    }
    const auto layout = mem::Layout::of<SimdBlock>();
    alloc::Global::deallocate(ptr, layout);
  }
};

struct HostAllocator {
  static auto allocate(usize size) -> void* {
    constexpr auto flags = u32{cudaHostAllocDefault};
    if (size == 0) {
      return nullptr;
    }

    auto p = ptr::null();
    if (auto err = ::cudaHostAlloc(&p, size, flags); err != cudaSuccess) {
      Result{Error(err)}.unwrap();
    }

    return p;
  }

  static void deallocate(void* ptr) {
    if (ptr == nullptr) {
      return;
    }

    if (auto err = ::cudaFreeHost(ptr); err != cudaSuccess) {
      Result{Error(err)}.unwrap();
    }
  }
};

struct DeviceAllocator {
  static auto allocate(usize size) -> void* {
    if (size == 0) {
      return nullptr;
    }

    auto p = ptr::null();
    if (auto err = ::cudaMalloc(&p, size); err != cudaSuccess) {
      Result{Error(err)}.unwrap();
    }
    return p;
  }

  static void deallocate(void* ptr) {
    if (ptr == nullptr) {
      return;
    }
    if (auto err = ::cudaFree(ptr); err != cudaSuccess) {
      Result{Error(err)}.unwrap();
    }
  }
};

struct ManagedAllocator {
  static auto allocate(usize size) -> void* {
    constexpr auto flags = u32{cudaMemAttachGlobal};

    if (size == 0) {
      return nullptr;
    }

    auto p = ptr::null();
    if (auto err = ::cudaMallocManaged(&p, size, flags); err != cudaSuccess) {
      Result{Error(err)}.unwrap();
    }
    return p;
  }

  static void deallocate(void* ptr) {
    if (ptr == nullptr) {
      return;
    }
    if (auto err = ::cudaFree(ptr); err != cudaSuccess) {
      Result{Error(err)}.unwrap();
    }
  }
};

auto MemLocation::fmt(fmt::Formatter& f) const -> void {
  f.write_fmt("{}:{}", to_str(kind), device);
}

auto to_str(MemKind kind) -> Str {
  switch (kind) {
    case MemKind::CPU: return "CPU";
    case MemKind::RAM: return "RAM";
    case MemKind::GPU: return "GPU";
    case MemKind::UVA: return "UVA";
  }
}

auto mem_allocate(usize size, MemLocation loc) -> void* {
  if (size == 0) {
    return nullptr;
  }

  if (loc.kind == MemKind::GPU || loc.kind == MemKind::UVA) {
    cuda::device_set(loc.device).unwrap();
  }

  switch (loc.kind) {
    case MemKind::CPU: return HeapAllocator::allocate(size);
    case MemKind::RAM: return HostAllocator::allocate(size);
    case MemKind::GPU: return DeviceAllocator::allocate(size);
    case MemKind::UVA: return ManagedAllocator::allocate(size);
  }
  return nullptr;
}

void mem_deallocate(void* ptr, MemLocation loc) {
  if (ptr == nullptr) {
    return;
  }

  if (loc.kind == MemKind::GPU || loc.kind == MemKind::UVA) {
    cuda::device_set(loc.device).unwrap();
  }

  switch (loc.kind) {
    case MemKind::CPU: return HeapAllocator::deallocate(ptr);
    case MemKind::RAM: return HostAllocator::deallocate(ptr);
    case MemKind::GPU: return DeviceAllocator::deallocate(ptr);
    case MemKind::UVA: return ManagedAllocator::deallocate(ptr);
  }
}

auto mem_location(void* ptr) -> MemLocation {
  if (ptr == nullptr) {
    return {};
  }

  auto attr = cudaPointerAttributes{};
  if (auto err = cudaPointerGetAttributes(&attr, ptr); err != cudaSuccess) {
    return {};
  }

  auto res = MemLocation{};
  res.kind = MemKind(attr.type);
  res.device = u32(attr.device);
  return res;
}

auto mem_prefetch(void* ptr, usize size, MemLocation loc) -> Result<> {
  if (size == 0) {
    return Ok{};
  }

  if (ptr == nullptr) {
    return Error(cudaErrorInvalidValue);
  }

  if (loc.kind != MemKind::CPU && loc.kind != MemKind::GPU) {
    return Error(cudaErrorInvalidValue);
  }

  const auto loc_type = loc.kind == MemKind::CPU ? cudaMemLocationTypeHost : cudaMemLocationTypeDevice;
  const auto loc_dev = loc.kind == MemKind::CPU ? 0 : static_cast<int>(loc.device);

  const auto cu_loc = cudaMemLocation{loc_type, loc_dev};
  const auto flags = 0U;  // must be zero now
  const auto stream = cuda::stream_current();
  if (auto err = cudaMemPrefetchAsync(ptr, size, cu_loc, flags, stream); err != cudaSuccess) {
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
  const auto err_code = stream ? cudaMemsetAsync(ptr, val, size, stream) : cudaMemset(ptr, val, size);

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
  const auto err_code = stream ? cudaMemcpyAsync(dst, src, size, kind, stream) : cudaMemcpy(dst, src, size, kind);

  if (err_code != cudaSuccess) {
    return Error(err_code);
  }

  return Ok{};
}

}  // namespace sfc::cuda
