
#include <cuda_runtime_api.h>

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

  auto f = [&](auto alloc_func, auto... args) -> Result<void*> {
    auto scope = cuda::Device{device}.scope();

    auto ptr = ptr::null();
    if (auto err = alloc_func(&ptr, args...)) {
      return Error(err);
    }
    return Ok{ptr};
  };

  switch (kind) {
    case MemKind::Heap:    return alloc::Global::allocate(layout); break;
    case MemKind::Host:    return f(cudaHostAlloc, size, u32{cudaHostAllocDefault}).unwrap(); break;
    case MemKind::Device:  return f(cudaMalloc, size).unwrap(); break;
    case MemKind::Managed: return f(cudaMallocManaged, size, u32{cudaMemAttachGlobal}).unwrap(); break;
  }
  return nullptr;
}

void MemLocation::deallocate(void* ptr, mem::Layout layout) {
  if (ptr == nullptr) {
    return;
  }

  auto f = [&](auto free_func, auto... args) -> Result<> {
    auto scope = cuda::Device{device}.scope();
    if (auto err = free_func(args...)) {
      return Error(err);
    }
    return Ok{};
  };

  switch (kind) {
    case MemKind::Heap:    alloc::Global::deallocate(ptr, layout); break;
    case MemKind::Host:    f(cudaFreeHost, ptr).unwrap(); break;
    case MemKind::Device:  f(cudaFree, ptr).unwrap(); break;
    case MemKind::Managed: f(cudaFree, ptr).unwrap(); break;
  }
}

auto MemLocation::pool() const -> mem_pool::Pool& {
  using Pool = mem_pool::XPool<MemLocation>;
  static Pool cpu_pool = Pool{Heap()};
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
    return cpu_pool;
  }

  switch (kind) {
    case MemKind::Heap:    return cpu_pool;
    case MemKind::Host:    return ram_pool;
    case MemKind::Device:  return gpu_pool[device];
    case MemKind::Managed: return uva_pool[device];
  }

  return cpu_pool;
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
