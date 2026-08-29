#include <cuda.h>
#include "sfc/core.h"
#include "sfc/cuda/device.h"
#include "sfc/cuda/stream.h"

namespace sfc::cuda {

namespace detail {

struct DeviceProp {
  int major;
  int minor;
  int sm_count;
  int async_engine_count;
  int l2_cache_size;
  size_t global_memory;
  char name[64];
};

auto init() -> Result<> {
  auto ret = cuInit(0);
  if (ret != CUDA_SUCCESS) {
    return Error(ret);
  }
  return Ok{};
}

auto ctx_sync() -> Result<> {
  if (auto err = cuCtxSynchronize()) {
    return Error(err);
  }
  return Ok{};
}

auto device_count() -> Result<u32> {
  int cnt = 0;
  if (auto err = cuDeviceGetCount(&cnt)) {
    return Error(err);
  }
  return u32(cnt);
}

auto device_of(u32 dev_id) -> Result<CUdevice> {
  CUdevice dev{};
  if (auto err = cuDeviceGet(&dev, int(dev_id))) {
    return Error(err);
  }
  return dev;
}

auto device_get() -> Result<CUdevice> {
  CUcontext ctx{};
  if (auto err = cuCtxGetCurrent(&ctx)) {
    return Error(err);
  }

  if (ctx == nullptr) {
    return Error(CUDA_ERROR_NOT_INITIALIZED);
  }

  auto dev = CUdevice{-1};
  if (auto err = cuCtxGetDevice(&dev)) {
    return Error(err);
  }

  return dev;
}

auto device_set(CUdevice dev) -> Result<CUdevice> {
  static thread_local auto _tls_dev = CUdevice{-1};
  if (dev == _tls_dev || dev == -1) {
    return Ok{dev};
  }

  CUcontext ctx{};
  if (auto err = cuDevicePrimaryCtxRetain(&ctx, dev)) {
    return Error(err);
  }

  if (auto err = cuCtxSetCurrent(ctx)) {
    return Error(err);
  }

  const auto prev_dev = _tls_dev;
  _tls_dev = dev;
  return Ok{prev_dev};
}

auto device_attr(CUdevice dev, CUdevice_attribute attr) -> Result<i32> {
  i32 val = 0;
  if (auto err = cuDeviceGetAttribute(&val, attr, dev)) {
    return Error(err);
  }
  return val;
}

auto device_name(CUdevice dev, Slice<char> name) -> Result<> {
  if (auto err = cuDeviceGetName(name._ptr, int(name._len), dev)) {
    return Error(err);
  }
  return Ok{};
}

auto device_total_mem(CUdevice dev) -> Result<size_t> {
  size_t mem = 0;
  if (auto err = cuDeviceTotalMem_v2(&mem, dev)) {
    return Error(err);
  }
  return mem;
}

auto device_prop(CUdevice dev) -> Result<DeviceProp> {
  auto prop = DeviceProp{};

  prop.major = _TRY(detail::device_attr(dev, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR));
  prop.minor = _TRY(detail::device_attr(dev, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR));
  prop.sm_count = _TRY(detail::device_attr(dev, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT));
  prop.async_engine_count = _TRY(detail::device_attr(dev, CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT));
  prop.l2_cache_size = _TRY(detail::device_attr(dev, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE));

  // others
  prop.global_memory = _TRY(detail::device_total_mem(dev));
  _TRY(detail::device_name(dev, prop.name));

  return prop;
}
}  // namespace detail

auto init() -> Result<> {
  return detail::init();
}

auto Device::count() -> Result<u32> {
  _TRY(detail::init());
  return detail::device_count();
}

auto Device::current() -> Result<Device> {
  _TRY(detail::init());
  const auto dev = _TRY(detail::device_get());
  return Device{dev};
}

auto Device::sync() -> Result<> {
  _TRY(detail::init());
  return detail::ctx_sync();
}

auto Device::try_from(u32 id) -> Result<Device> {
  _TRY(detail::init());
  const auto dev = _TRY(detail::device_of(id));
  return Device{dev};
}

auto Device::info() const -> Result<Device::Info> {
  _TRY(detail::init());
  static constexpr auto kMaxDevId = 64;
  static detail::DeviceProp props[kMaxDevId]{};

  if (_dev < 0 || _dev >= kMaxDevId) {
    return Error{CUDA_ERROR_INVALID_DEVICE};
  }

  auto& prop = props[_dev];
  if (prop.major == 0 && prop.minor == 0) {
    prop = _TRY(detail::device_prop(_dev));
  }

  auto info = Info{
      .device = _dev,
      .compute_capability = u32(prop.major * 10 + prop.minor),
      .sm_count = u32(prop.sm_count),
      .async_engine_count = u32(prop.async_engine_count),
      .l2_cache_size = u32(prop.l2_cache_size),
      .global_memory = u64(prop.global_memory),
      .name = Str::from_cstr(prop.name),
  };
  return info;
}

auto Device::enter() -> Device::Entered {
  return Device::Entered{*this};
}

Device::Entered::Entered(const Device& dev) : _dev_in{dev._dev}, _dev_out{_dev_in} {
  _dev_out = detail::device_set(_dev_in).unwrap();
}

Device::Entered::~Entered() {
  if (_dev_out == -1) {
    return;
  }
  detail::device_set(_dev_out).unwrap();
}

void Device::Info::fmt(fmt::Formatter& f) const {
  f.debug_struct("CudaDevice")
      .field("name", fmt::Args{"{s}", name})
      .field("device", device)
      .field("compute_capability", compute_capability)
      .field("sm_count", sm_count)
      .field("async_engine_count", async_engine_count)
      .field("l2_cache_size", l2_cache_size)
      .field("global_memory", global_memory);
}

}  // namespace sfc::cuda
