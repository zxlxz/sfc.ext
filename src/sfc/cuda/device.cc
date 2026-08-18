#include <cuda.h>
#include "sfc/core.h"
#include "sfc/cuda/device.h"
#include "sfc/cuda/stream.h"

namespace sfc::cuda {

struct DeviceProp {
  int major;
  int minor;
  int sm_count;
  int async_engine_count;
  int l2_cache_size;
  size_t global_memory;
  char name[64];
};

static auto init() -> Result<> {
  static auto ret = cuInit(0);
  if (ret != CUDA_SUCCESS) {
    return Error(ret);
  }
  return Ok{};
}

static auto get_current() -> Result<u32> {
  _TRY(cuda::init());

  CUdevice dev{};
  if (auto err = cuCtxGetDevice(&dev); err != CUDA_SUCCESS) {
    return Error(err);
  }

  const auto dev_id = u32(dev);
  return dev_id;
}

static auto set_current(u32 dev_id) -> Result<> {
  _TRY(cuda::init());

  CUdevice dev{};
  if (auto err = cuDeviceGet(&dev, int(dev_id)); err != CUDA_SUCCESS) {
    return Error(err);
  }

  CUcontext ctx{};
  if (auto err = cuDevicePrimaryCtxRetain(&ctx, dev); err != CUDA_SUCCESS) {
    return Error(err);
  }

  if (auto err = cuCtxSetCurrent(ctx); err != CUDA_SUCCESS) {
    return Error(err);
  }
  return Ok{};
}

static auto device_prop(u32 dev_id) -> Result<DeviceProp> {
  CUdevice dev{};
  if (auto err = cuDeviceGet(&dev, int(dev_id)); err != CUDA_SUCCESS) {
    return Error(err);
  }

  auto prop = DeviceProp{};
  if (auto err = cuDeviceGetAttribute(&prop.major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev)) {
    return Error(err);
  }
  if (auto err = cuDeviceGetAttribute(&prop.minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev)) {
    return Error(err);
  }
  if (auto err = cuDeviceGetAttribute(&prop.sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev)) {
    return Error(err);
  }
  if (auto err = cuDeviceGetAttribute(&prop.async_engine_count, CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT, dev)) {
    return Error(err);
  }
  if (auto err = cuDeviceGetAttribute(&prop.l2_cache_size, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev)) {
    return Error(err);
  }
  if (auto err = cuDeviceTotalMem_v2(&prop.global_memory, dev)) {
    return Error(err);
  }
  if (auto err = cuDeviceGetName(prop.name, sizeof(prop.name), dev)) {
    return Error(err);
  }
  return prop;
}

auto Device::count() -> Result<u32> {
  _TRY(cuda::init());

  int cnt = 0;
  if (auto err = cuDeviceGetCount(&cnt)) {
    return Error(err);
  }

  return u32(cnt);
}

auto Device::current() -> Result<Device> {
  const auto dev_id = _TRY(cuda::get_current());
  return Device{dev_id};
}

auto Device::sync() -> Result<> {
  _TRY(cuda::init());

  if (auto err = cuCtxSynchronize()) {
    return Error(err);
  }
  return Ok{};
}

auto Device::info() const -> Result<DeviceInfo> {
  static constexpr u32 kMaxDevCount = 16U;
  static DeviceProp props[kMaxDevCount]{};

  if (id >= kMaxDevCount) {
    return Error{CUDA_ERROR_INVALID_DEVICE};
  }

  auto& prop = props[id];
  if (prop.major == 0 && prop.minor == 0) {
    prop = _TRY(cuda::device_prop(id));
  }

  auto info = DeviceInfo{};
  info.dev_id = id;
  info.compute_capability = u32(prop.major * 10 + prop.minor);
  info.sm_count = u32(prop.sm_count);
  info.async_engine_count = u32(prop.async_engine_count);
  info.l2_cache_size = u64(prop.l2_cache_size);
  info.global_memory = u64(prop.global_memory);
  info.name = prop.name;
  return info;
}

auto Device::scope() -> Device::Guard {
  return Device::Guard{id};
}

Device::Guard::Guard(u32 id) : _dev_in{0}, _dev_out{0} {
  (void)cuda::init();

  auto dev_out = cuda::get_current().ok();
  if (dev_out != Option{id}) {
    cuda::set_current(id).unwrap();
  }

  _dev_in = id;
  _dev_out = dev_out.unwrap_or(_dev_in);
}

Device::Guard::~Guard() {
  if (_dev_in != _dev_out) {
    cuda::set_current(_dev_out).unwrap();
  }
}

void DeviceInfo::fmt(fmt::Formatter& f) const {
  f.debug_struct("DeviceInfo")
      .field("name", Str::from_cstr(name))
      .field("dev_id", dev_id)
      .field("compute_capability", compute_capability)
      .field("sm_count", sm_count)
      .field("async_engine_count", async_engine_count)
      .field("global_memory", global_memory)
      .field("l2_cache_size", l2_cache_size);
}

}  // namespace sfc::cuda
