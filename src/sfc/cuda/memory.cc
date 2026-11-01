#include <cuda.h>

#include "sfc/alloc.h"
#include "sfc/cuda/mod.h"
#include "sfc/cuda/memory.h"
#include "sfc/cuda/stream.h"
#include "sfc/cuda/device.h"

namespace sfc::cuda {

static auto to_device_ptr(const void* p) -> CUdeviceptr {
  return CUdeviceptr(p);
}

static auto heap_alloc(usize size) -> Result<void*> {
  const auto ptr = ::malloc(size);
  return ptr;
}

static auto heap_free(void* ptr) -> Result<> {
  ::free(ptr);
  return Ok{};
}

static auto host_alloc(usize size) -> Result<void*> {
  void* ptr = nullptr;
  if (auto err = cuMemHostAlloc(&ptr, size, CU_MEMHOSTALLOC_PORTABLE)) {
    return Error(err);
  }
  return Ok{ptr};
}

static auto host_free(void* ptr) -> Result<> {
  if (auto err = cuMemFreeHost(ptr)) {
    return Error(err);
  }
  return Ok{};
}

static auto device_alloc(usize size) -> Result<void*> {
  auto device_ptr = CUdeviceptr{0};
  if (auto err = cuMemAlloc_v2(&device_ptr, size)) {
    return Error(err);
  }
  const auto ptr = reinterpret_cast<void*>(device_ptr);
  return Ok{ptr};
}

static auto device_free(void* ptr) -> Result<> {
  auto device_ptr = to_device_ptr(ptr);

  if (auto err = cuMemFree_v2(device_ptr)) {
    return Error(err);
  }
  return Ok{};
}

static auto memcpy_uva(const void* src, void* dst, usize size, stream_t stream) -> Result<> {
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

static auto memcpy_h2d(const void* src, void* dst, usize size, stream_t stream) -> Result<> {
  if (size == 0) {
    return Ok{};
  }

  const auto d_dst = to_device_ptr(dst);
  if (auto err = cuMemcpyHtoDAsync(d_dst, src, size, stream)) {
    return Error(err);
  }

  return Ok{};
}

static auto memcpy_d2h(const void* src, void* dst, usize size, stream_t stream) -> Result<> {
  if (size == 0) {
    return Ok{};
  }

  const auto d_src = to_device_ptr(src);
  if (auto err = cuMemcpyDtoHAsync(dst, d_src, size, stream)) {
    return Error(err);
  }

  return Ok{};
}

static auto memcpy_d2d(const void* src, void* dst, usize size, stream_t stream) -> Result<> {
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

static auto memset_async(void* buf, usize len, u8 val, stream_t stream) -> Result<> {
  if (len == 0) {
    return Ok{};
  }

  const auto device_ptr = to_device_ptr(buf);
  if (auto err = cuMemsetD8Async(device_ptr, val, len, stream)) {
    return Error(err);
  }

  return Ok{};
}

auto to_str(MemKind kind) -> str::Str {
  switch (kind) {
    default:              return "Unknown";
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

auto MemBlock::fill_bytes(u8 val) -> Result<> {
  if (buf._len == 0) {
    return Ok{};
  }

  if (loc.on_host()) {
    ptr::write_bytes(buf._ptr, val, buf._len);
    return Ok{};
  }

  {
    auto dev = cuda::Device::try_from(loc.device).unwrap();
    const auto ctx = dev.enter();
    const auto stream = cuda::stream_current();
    _TRY(cuda::memset_async(buf._ptr, buf._len, val, stream));
  }

  return Ok{};
}

auto MemBlock::copy_from(MemBlock src) -> Result<> {
  if (buf._len != src.buf._len) {
    return Error(CUDA_ERROR_INVALID_VALUE);
  }

  if (buf._len == 0) {
    return Ok{};
  }

  const auto src_on_device = src.loc.on_device();
  const auto dst_on_device = loc.on_device();
  const auto dev_id = src_on_device ? src.loc.device : loc.device;

  const auto sptr = ptr::cast<const u8>(src.buf._ptr);
  const auto dptr = buf._ptr;
  const auto len = buf._len;

  if (!src_on_device && !dst_on_device) {
    ptr::copy_nonoverlapping(sptr, dptr, len);
    return Ok{};
  }

  {
    auto dev = Device::try_from(dev_id).unwrap();
    const auto ctx = dev.enter();
    const auto stream = cuda::stream_current();
    if (src_on_device && dst_on_device) {
      _TRY(cuda::memcpy_d2d(sptr, dptr, len, stream));
    } else if (src_on_device) {
      _TRY(cuda::memcpy_d2h(sptr, dptr, len, stream));
    } else if (dst_on_device) {
      _TRY(cuda::memcpy_h2d(sptr, dptr, len, stream));
    } else {
      _TRY(cuda::memcpy_uva(sptr, dptr, len, stream));
    }
  }

  return Ok{};
}

class HeapAllocator : public math::Allocator {
 public:
  static auto instance() -> Allocator& {
    static HeapAllocator a;
    return a;
  }

  auto allocate(mem::Layout layout) -> void* override {
    const auto ptr = cuda::heap_alloc(layout.size).unwrap();
    return ptr;
  }

  auto allocate_zeroed(mem::Layout layout) -> void* override {
    const auto ptr = cuda::heap_alloc(layout.size).unwrap();
    if (ptr != nullptr) {
      ptr::write_bytes((u8*)ptr, 0, layout.size);
    }
    return ptr;
  }

  void deallocate(void* ptr, [[maybe_unused]] mem::Layout layout) override {
    if (ptr == nullptr) {
      return;
    }
    cuda::heap_free(ptr).unwrap();
  }
};

class DeviceAllocator : public math::Allocator {
  u32 _device;

 public:
  explicit DeviceAllocator(u32 device) : _device{device} {}

  static auto instance(u32 device) -> Allocator& {
    static DeviceAllocator a[] = {
        DeviceAllocator{0},
        DeviceAllocator{1},
        DeviceAllocator{2},
        DeviceAllocator{3},
        DeviceAllocator{4},
        DeviceAllocator{5},
        DeviceAllocator{6},
        DeviceAllocator{7},
    };
    if (device < 8) {
      return a[device];
    }
    return math::Allocator::global();
  }

 public:
  auto device() const -> cuda::Device {
    return cuda::Device::try_from(_device).unwrap();
  }

  auto allocate(mem::Layout layout) -> void* {
    const auto ctx = this->device().enter();
    const auto ptr = cuda::device_alloc(layout.size).unwrap();
    return ptr;
  }

  auto allocate_zeroed(mem::Layout layout) -> void* {
    const auto ctx = this->device().enter();
    const auto ptr = cuda::device_alloc(layout.size).unwrap();
    if (ptr != nullptr) {
      const auto stream = cuda::stream_current();
      cuda::memset_async(ptr, layout.size, 0, stream).unwrap();
    }
    return ptr;
  }

  void deallocate(void* ptr, [[maybe_unused]] mem::Layout layout) {
    if (ptr == nullptr) return;
    const auto ctx = this->device().enter();
    cuda::device_free(ptr).unwrap();
  }
};

class HostAllocator : public math::Allocator {
  u32 _device;

 public:
  explicit HostAllocator(u32 device) : _device{device} {}

  static auto instance(u32 device) -> Allocator& {
    static HostAllocator a[] = {
        HostAllocator{0},
        HostAllocator{1},
        HostAllocator{2},
        HostAllocator{3},
        HostAllocator{4},
        HostAllocator{5},
        HostAllocator{6},
        HostAllocator{7},
    };
    if (device < 8) {
      return a[device];
    }
    return math::Allocator::global();
  }

 public:
  auto device() const -> cuda::Device {
    return cuda::Device::try_from(_device).unwrap();
  }

  auto allocate(mem::Layout layout) -> void* {
    const auto ctx = this->device().enter();
    const auto ptr = cuda::host_alloc(layout.size).unwrap();
    return ptr;
  }

  auto allocate_zeroed(mem::Layout layout) -> void* {
    const auto ctx = this->device().enter();
    const auto ptr = cuda::host_alloc(layout.size).unwrap();
    if (ptr != nullptr) {
      ptr::write_bytes((u8*)ptr, 0, layout.size);
    }
    return ptr;
  }

  void deallocate(void* ptr, [[maybe_unused]] mem::Layout layout) {
    const auto ctx = this->device().enter();
    cuda::host_free(ptr).unwrap();
  }
};

auto MemLocation::allocator() const -> math::Allocator& {
  switch (kind) {
    default:
    case MemKind::Heap:   return HeapAllocator::instance();
    case MemKind::Host:   return HostAllocator::instance(device);
    case MemKind::Device: return DeviceAllocator::instance(device);
  }
}

}  // namespace sfc::cuda
