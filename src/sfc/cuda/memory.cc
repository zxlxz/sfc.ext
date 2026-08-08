
#include <cuda_runtime_api.h>

#include "sfc/alloc.h"
#include "sfc/cuda/mod.h"
#include "sfc/cuda/memory.h"
#include "sfc/cuda/stream.h"
#include "sfc/cuda/device.h"

namespace sfc::cuda {

auto to_str(MemKind kind) -> str::Str {
  switch (kind) {
    case MemKind::CPU: return "CPU";
    case MemKind::RAM: return "RAM";
    case MemKind::GPU: return "GPU";
    case MemKind::UVA: return "UVA";
  }
}

void MemLocation::fmt(fmt::Formatter& f) const {
  f.write_fmt("{}:{}", to_str(kind), device);
}

auto MemLocation::allocate(mem::Layout layout) -> void* {
  const auto size = layout.size;
  if (layout.size == 0) {
    return nullptr;
  }

  auto f = [&](auto alloc_func, auto... args) -> Result<void*> {
    auto scope = Device{device}.scope();

    auto ptr = ptr::null();
    if (auto err = alloc_func(&ptr, args...)) {
      return Error(err);
    }
    return Ok{ptr};
  };

  switch (kind) {
    case MemKind::CPU: return alloc::Global::allocate(layout); break;
    case MemKind::RAM: return f(cudaHostAlloc, size, u32{cudaHostAllocDefault}).unwrap(); break;
    case MemKind::GPU: return f(cudaMalloc, size).unwrap(); break;
    case MemKind::UVA: return f(cudaMallocManaged, size, u32{cudaMemAttachGlobal}).unwrap(); break;
  }
  return nullptr;
}

void MemLocation::deallocate(void* ptr, mem::Layout layout) {
  if (ptr == nullptr) {
    return;
  }

  auto f = [&](auto free_func, auto... args) -> Result<> {
    auto scope = Device{this->device}.scope();
    if (auto err = free_func(args...)) {
      return Error(err);
    }
    return Ok{};
  };

  switch (kind) {
    case MemKind::CPU: alloc::Global::deallocate(ptr, layout); break;
    case MemKind::RAM: f(cudaFreeHost, ptr).unwrap(); break;
    case MemKind::GPU: f(cudaFree, ptr).unwrap(); break;
    case MemKind::UVA: f(cudaFree, ptr).unwrap(); break;
  }
}

auto MemLocation::pool() const -> mem_pool::Pool& {
  using Pool = mem_pool::XPool<MemLocation>;
  static Pool cpu_pool = Pool{{MemKind::CPU, 0}};
  static Pool ram_pool = Pool{{MemKind::RAM, 0}};
  static Pool gpu_pool[8] = {
      Pool{{MemKind::GPU, 0}},
      Pool{{MemKind::GPU, 1}},
      Pool{{MemKind::GPU, 2}},
      Pool{{MemKind::GPU, 3}},
      Pool{{MemKind::GPU, 4}},
      Pool{{MemKind::GPU, 5}},
      Pool{{MemKind::GPU, 6}},
      Pool{{MemKind::GPU, 7}},
  };
  static Pool uva_pool[8] = {
      Pool{{MemKind::UVA, 0}},
      Pool{{MemKind::UVA, 1}},
      Pool{{MemKind::UVA, 2}},
      Pool{{MemKind::UVA, 3}},
      Pool{{MemKind::UVA, 4}},
      Pool{{MemKind::UVA, 5}},
      Pool{{MemKind::UVA, 6}},
      Pool{{MemKind::UVA, 7}},
  };

  if (device >= kMaxDeviceCnt) {
    return cpu_pool;
  }

  switch (kind) {
    case MemKind::CPU: return cpu_pool;
    case MemKind::RAM: return ram_pool;
    case MemKind::GPU: return gpu_pool[device];
    case MemKind::UVA: return uva_pool[device];
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
