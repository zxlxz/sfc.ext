
#include <cuda_runtime_api.h>

#include "sfc/core.h"
#include "sfc/cuda/device.h"
#include "sfc/cuda/stream.h"

namespace sfc::cuda {

void DeviceInfo::fmt(fmt::Formatter& f) const {
  f.debug_struct("DeviceInfo")
      .field("name", name)
      .field("dev_id", dev_id)
      .field("compute_capability", compute_capability)
      .field("sm_count", sm_count)
      .field("async_engine_count", async_engine_count)
      .field("global_memory", global_memory)
      .field("l2_cache_size", l2_cache_size);
}

static auto get_device() -> Result<i32> {
  auto dev = 0;
  if (auto err = ::cudaGetDevice(&dev)) {
    return Error(err);
  }
  return Ok{dev};
}

static auto set_device(int dev) -> Result<> {
  if (auto err = ::cudaSetDevice(dev)) {
    return Error(err);
  }
  return Ok{};
}

static auto device_prop(u32 dev) -> Result<cudaDeviceProp> {
  const auto device = num::cast_signed(dev);

  auto prop = cudaDeviceProp{};
  if (auto err = ::cudaGetDeviceProperties(&prop, device)) {
    return Error(err);
  }
  return Ok{prop};
}

DeviceGuard::DeviceGuard(int enter, int exit) : _dev_enter{enter}, _dev_exit{exit} {
  cuda::set_device(_dev_enter).unwrap();
}

DeviceGuard::~DeviceGuard() {
  cuda::set_device(_dev_exit).unwrap();
}

auto Device::count() -> u32 {
  auto cnt = 0;
  if (auto err = ::cudaGetDeviceCount(&cnt); err != cudaSuccess) {
    return 0;
  }
  return num::cast_unsigned(cnt);
}

auto Device::current() -> Device {
  auto dev = 0;
  ::cudaGetDevice(&dev);
  return Device{num::cast_unsigned(dev)};
}

auto Device::sync() -> Result<> {
  if (auto err = ::cudaDeviceSynchronize()) {
    return Error(err);
  }
  return Ok{};
}

auto Device::info() const -> DeviceInfo {
  static const auto MAX_DEV_COUNT = 16U;
  static cudaDeviceProp props[MAX_DEV_COUNT];
  if (id >= MAX_DEV_COUNT) {
    return {};
  }

  if (props[id].totalGlobalMem == 0) {
    const auto p = cuda::device_prop(id).unwrap();
    props[id] = p;
  }

  const auto& p = props[id];
  const auto info = DeviceInfo{
      .dev_id = id,
      .compute_capability = u32(p.major * 10 + p.minor),
      .sm_count = u32(p.multiProcessorCount),
      .async_engine_count = u32(p.asyncEngineCount),
      .global_memory = p.totalGlobalMem,
      .l2_cache_size = u32(p.l2CacheSize),
      .name = p.name,
  };
  return info;
}

auto Device::scope() -> DeviceGuard {
  const auto curr_id = cuda::get_device().unwrap();
  const auto next_id = int(this->id);
  return DeviceGuard{curr_id, next_id};
}

}  // namespace sfc::cuda
