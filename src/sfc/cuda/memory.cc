#include <cuda.h>

#include "sfc/alloc.h"
#include "sfc/cuda/mod.h"
#include "sfc/cuda/memory.h"
#include "sfc/cuda/stream.h"
#include "sfc/cuda/device.h"

namespace sfc::cuda {

namespace detail {

static auto to_device_ptr(const void* p) -> CUdeviceptr {
  return __builtin_bit_cast(CUdeviceptr, p);
}

static auto host_alloc(usize size) -> Result<void*> {
  if (size == 0) {
    return ptr::null();
  }

  void* ptr = nullptr;
  if (auto err = cuMemHostAlloc(&ptr, size, CU_MEMHOSTALLOC_PORTABLE)) {
    return Error(err);
  }
  return Ok{ptr};
}

static auto host_free(void* ptr) -> Result<> {
  if (ptr == nullptr) {
    return Ok{};
  }

  if (auto err = cuMemFreeHost(ptr)) {
    return Error(err);
  }
  return Ok{};
}

static auto device_alloc(usize size) -> Result<void*> {
  if (size == 0) {
    return ptr::null();
  }

  auto device_ptr = CUdeviceptr{0};
  if (auto err = cuMemAlloc_v2(&device_ptr, size)) {
    return Error(err);
  }
  const auto ptr = __builtin_bit_cast(void*, device_ptr);
  return Ok{ptr};
}

static auto device_free(void* ptr) -> Result<> {
  if (ptr == nullptr) {
    return Ok{};
  }

  const auto device_ptr = to_device_ptr(ptr);
  if (auto err = cuMemFree_v2(device_ptr)) {
    return Error(err);
  }
  return Ok{};
}

}  // namespace detail

class HostAllocator : public math::Allocator {
  u32 _device;

 public:
  explicit HostAllocator(u32 device) : _device{device} {}

 public:
  auto device() const -> cuda::Device {
    return cuda::Device::try_from(_device).unwrap();
  }

  auto allocate(mem::Layout layout) -> void* {
    const auto ctx = this->device().enter();
    const auto ptr = detail::host_alloc(layout.size).unwrap();
    return ptr;
  }

  auto allocate_zeroed(mem::Layout layout) -> void* {
    const auto ctx = this->device().enter();
    const auto ptr = detail::host_alloc(layout.size).unwrap();
    if (ptr != nullptr) {
      ptr::write_bytes((u8*)ptr, 0, layout.size);
    }
    return ptr;
  }

  void deallocate(void* ptr, [[maybe_unused]] mem::Layout layout) {
    const auto ctx = this->device().enter();
    detail::host_free(ptr).unwrap();
  }
};

class DeviceAllocator : public math::Allocator {
  u32 _device;

 public:
  explicit DeviceAllocator(u32 device) : _device{device} {}

 public:
  auto device() const -> cuda::Device {
    return cuda::Device::try_from(_device).unwrap();
  }

  auto allocate(mem::Layout layout) -> void* {
    const auto ctx = this->device().enter();
    const auto ptr = detail::device_alloc(layout.size).unwrap();
    return ptr;
  }

  auto allocate_zeroed(mem::Layout layout) -> void* {
    const auto ctx = this->device().enter();
    const auto ptr = detail::device_alloc(layout.size).unwrap();
    if (ptr != nullptr) {
      cuda::bzero(ptr, layout.size).unwrap();
    }
    return ptr;
  }

  void deallocate(void* ptr, [[maybe_unused]] mem::Layout layout) {
    if (ptr == nullptr) {
      return;
    }

    const auto ctx = this->device().enter();
    detail::device_free(ptr).unwrap();
  }
};

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

void MemLocation::fmt(fmt::Formatter& f) const {
  const auto kind_str = [&]() {
    switch (kind) {
      default:              return "mem";
      case MemKind::Heap:   return "cpu";
      case MemKind::Host:   return "uva";
      case MemKind::Device: return "gpu";
    }
  }();

  f.write_fmt("{}:{}", kind_str, device);
}

auto MemLocation::allocator() const -> math::Allocator& {
  static constexpr u32 kDeviceCnt = 8U;

  static HostAllocator host_allocators[] = {
      HostAllocator{0},
      HostAllocator{1},
      HostAllocator{2},
      HostAllocator{3},
      HostAllocator{4},
      HostAllocator{5},
      HostAllocator{6},
      HostAllocator{7},
  };

  static DeviceAllocator device_allocators[] = {
      DeviceAllocator{0},
      DeviceAllocator{1},
      DeviceAllocator{2},
      DeviceAllocator{3},
      DeviceAllocator{4},
      DeviceAllocator{5},
      DeviceAllocator{6},
      DeviceAllocator{7},
  };

  if (device >= kDeviceCnt) {
    return math::Allocator::global();
  }

  switch (kind) {
    default:
    case MemKind::Heap:   return math::Allocator::global();
    case MemKind::Host:   return host_allocators[device];
    case MemKind::Device: return device_allocators[device];
  }
}

auto bzero(void* dst, usize size) -> Result<> {
  if (size == 0) {
    return Ok{};
  }

  const auto stream = cuda::stream_current();
  const auto d_dst = detail::to_device_ptr(dst);
  if (auto err = ::cuMemsetD8Async(d_dst, 0, size, stream)) {
    return Error(err);
  }
  return Ok{};
}

auto copy_h2h(const void* src, void* dst, usize size) -> Result<> {
  if (size == 0) {
    return Ok{};
  }
  ptr::copy_nonoverlapping((const u8*)src, (u8*)dst, size);
  return Ok{};
}

auto copy_uva(const void* src, void* dst, usize size) -> Result<> {
  if (size == 0) {
    return Ok{};
  }

  const auto stream = cuda::stream_current();
  const auto d_src = detail::to_device_ptr(src);
  const auto d_dst = detail::to_device_ptr(dst);
  if (auto err = cuMemcpyAsync(d_dst, d_src, size, stream)) {
    return Error(err);
  }
  return Ok{};
}

auto copy_h2d(const void* src, void* dst, usize size) -> Result<> {
  if (size == 0) {
    return Ok{};
  }

  const auto stream = cuda::stream_current();
  const auto d_dst = detail::to_device_ptr(dst);
  if (auto err = cuMemcpyHtoDAsync_v2(d_dst, src, size, stream)) {
    return Error(err);
  }
  return Ok{};
}

auto copy_d2h(const void* src, void* dst, usize size) -> Result<> {
  if (size == 0) {
    return Ok{};
  }

  const auto stream = cuda::stream_current();
  const auto d_src = detail::to_device_ptr(src);
  if (auto err = cuMemcpyDtoHAsync_v2(dst, d_src, size, stream)) {
    return Error(err);
  }
  return Ok{};
}

auto copy_d2d(const void* src, void* dst, usize size) -> Result<> {
  if (size == 0) {
    return Ok{};
  }

  const auto stream = cuda::stream_current();
  const auto d_src = detail::to_device_ptr(src);
  const auto d_dst = detail::to_device_ptr(dst);
  if (auto err = cuMemcpyDtoDAsync_v2(d_dst, d_src, size, stream)) {
    return Error(err);
  }
  return Ok{};
}

}  // namespace sfc::cuda
