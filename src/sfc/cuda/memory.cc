#include <cuda.h>

#include "sfc/alloc.h"
#include "sfc/cuda/mod.h"
#include "sfc/cuda/memory.h"
#include "sfc/cuda/stream.h"
#include "sfc/cuda/device.h"

namespace sfc::cuda {

namespace detail {

using HeapAlloc = sfc::alloc::Global;

auto to_device_ptr(const void* p) -> CUdeviceptr {
  return __builtin_bit_cast(CUdeviceptr, p);
}

auto heap_alloc(Layout layout) -> Result<void*> {
  auto ptr = HeapAlloc::allocate(layout);
  return ptr;
}

auto heap_dealloc(void* ptr, Layout layout) -> Result<> {
  HeapAlloc::deallocate(ptr, layout);
  return Ok{};
}

auto host_alloc(Layout layout) -> Result<void*> {
  void* ptr = nullptr;
  if (auto err = cuMemHostAlloc(&ptr, layout.size, CU_MEMHOSTALLOC_PORTABLE)) {
    return Error(err);
  }
  return Ok{ptr};
}

auto host_dealloc(void* ptr, [[maybe_unused]] Layout layout) -> Result<> {
  if (auto err = cuMemFreeHost(ptr)) {
    return Error(err);
  }
  return Ok{};
}

auto device_alloc(Layout layout) -> Result<void*> {
  auto device_ptr = CUdeviceptr{0};
  if (auto err = cuMemAlloc_v2(&device_ptr, layout.size)) {
    return Error(err);
  }
  const auto ptr = reinterpret_cast<void*>(u64(device_ptr));
  return Ok{ptr};
}

auto device_dealloc(void* ptr, [[maybe_unused]] Layout layout) -> Result<> {
  auto device_ptr = to_device_ptr(ptr);

  if (auto err = cuMemFree_v2(device_ptr)) {
    return Error(err);
  }
  return Ok{};
}

auto memcpy_uva(void* dst, const void* src, usize size, stream_t stream) -> Result<> {
  if (size == 0) {
    return Ok{};
  }

  const auto d_src = to_device_ptr(src);
  const auto d_dst = to_device_ptr(dst);
  if (auto err = cuMemcpyAsync(d_dst, d_src, size, stream)) {
    return Error(err);
  }

  return Ok{};
}

auto memcpy_h2d(void* dst, const void* src, usize size, stream_t stream) -> Result<> {
  if (size == 0) {
    return Ok{};
  }

  const auto d_dst = to_device_ptr(dst);
  if (auto err = cuMemcpyHtoDAsync(d_dst, src, size, stream)) {
    return Error(err);
  }

  return Ok{};
}

auto memcpy_d2h(void* dst, const void* src, usize size, stream_t stream) -> Result<> {
  if (size == 0) {
    return Ok{};
  }

  const auto d_src = to_device_ptr(src);
  if (auto err = cuMemcpyDtoHAsync(dst, d_src, size, stream)) {
    return Error(err);
  }

  return Ok{};
}

auto memcpy_d2d(void* dst, const void* src, usize size, stream_t stream) -> Result<> {
  if (size == 0) {
    return Ok{};
  }

  const auto d_src = to_device_ptr(src);
  const auto d_dst = to_device_ptr(dst);
  if (auto err = cuMemcpyDtoDAsync(d_dst, d_src, size, stream)) {
    return Error(err);
  }

  return Ok{};
}

auto memset_async(void* ptr, u8 val, usize size, stream_t stream) -> Result<> {
  if (size == 0) {
    return Ok{};
  }

  const auto device_ptr = to_device_ptr(ptr);
  if (auto err = cuMemsetD8Async(device_ptr, val, size, stream)) {
    return Error(err);
  }

  return Ok{};
}

}  // namespace detail

auto to_str(MemKind kind) -> str::Str {
  switch (kind) {
    case MemKind::Heap:   return "Heap";
    case MemKind::Host:   return "Host";
    case MemKind::Device: return "Device";
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

auto MemLocation::on_host() const -> bool {
  return kind == MemKind::Heap || kind == MemKind::Host;
}

auto MemLocation::on_device() const -> bool {
  return kind == MemKind::Device;
}

auto MemLocation::allocate(mem::Layout layout) -> void* {
  cuda::init().unwrap();

  if (layout.size == 0) {
    return nullptr;
  }

  void* ptr = nullptr;
  switch (kind) {
    case MemKind::Heap: {
      ptr = detail::heap_alloc(layout).unwrap();
      break;
    }
    case MemKind::Host: {
      auto dev = cuda::Device::try_from(device).unwrap();
      auto enter = dev.enter();
      ptr = detail::host_alloc(layout).unwrap();
      break;
    }
    case MemKind::Device: {
      auto dev = cuda::Device::try_from(device).unwrap();
      auto enter = dev.enter();
      ptr = detail::device_alloc(layout).unwrap();
      break;
    }
  }
  return ptr;
}

void MemLocation::deallocate(void* ptr, mem::Layout layout) {
  cuda::init().unwrap();

  if (ptr == nullptr) {
    return;
  }

  switch (kind) {
    case MemKind::Heap: {
      detail::heap_dealloc(ptr, layout).unwrap();
      break;
    }
    case MemKind::Host: {
      auto dev = cuda::Device::try_from(device).unwrap();
      auto enter = dev.enter();
      detail::host_dealloc(ptr, layout).unwrap();
      break;
    }
    case MemKind::Device: {
      auto dev = cuda::Device::try_from(device).unwrap();
      auto enter = dev.enter();
      detail::device_dealloc(ptr, layout).unwrap();
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

  if (device >= kMaxDeviceCnt) {
    return Pool::global();
  }

  switch (kind) {
    case MemKind::Heap:   return Pool::global();
    case MemKind::Host:   return ram_pool;
    case MemKind::Device: return gpu_pool[device];
  }

  return Pool::global();
}

auto MemBlock::fill_bytes(u8 val) -> Result<> {
  if (size == 0) {
    return Ok{};
  }

  if (ptr == nullptr) {
    return Error(CUDA_ERROR_INVALID_VALUE);
  }

  if (loc.on_host()) {
    __builtin_memset(ptr, val, size);
    return Ok{};
  }

  auto dev = cuda::Device::try_from(loc.device).unwrap();
  auto ctx = dev.enter();
  const auto stream = cuda::stream_current();
  _TRY(detail::memset_async(ptr, val, size, stream));
  return Ok{};
}

auto MemBlock::copy_from(MemBlock src) -> Result<> {
  if (size != src.size) {
    return Error(CUDA_ERROR_INVALID_VALUE);
  }

  if (size == 0) {
    return Ok{};
  }

  if (this->ptr == nullptr || src.ptr == nullptr) {
    return Error(CUDA_ERROR_INVALID_VALUE);
  }

  if (this->loc.on_host() && src.loc.on_host()) {
    __builtin_memcpy(ptr, src.ptr, size);
    return Ok{};
  }

  auto dev_id = loc.on_device() ? loc.device : src.loc.on_device() ? src.loc.device : 0;
  auto dev = cuda::Device::try_from(dev_id).unwrap();
  auto ctx = dev.enter();
  const auto stream = cuda::stream_current();

  if (loc.on_host() && src.loc.on_device()) {
    _TRY(detail::memcpy_h2d(ptr, src.ptr, size, stream));
  } else if (loc.on_device() && src.loc.on_host()) {
    _TRY(detail::memcpy_d2h(ptr, src.ptr, size, stream));
  } else if (loc.on_device() && src.loc.on_device()) {
    _TRY(detail::memcpy_d2d(ptr, src.ptr, size, stream));
  } else {
    // NOTE: this branch will not be reached
    // only to handle unexpected memory locations
    _TRY(detail::memcpy_uva(ptr, src.ptr, size, stream));
  }
  return Ok{};
}

}  // namespace sfc::cuda
