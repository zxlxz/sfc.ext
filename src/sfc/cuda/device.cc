
#include <cuda_runtime_api.h>

#include "sfc/core.h"
#include "sfc/cuda/device.h"
#include "sfc/cuda/stream.h"

namespace sfc::cuda {

auto to_str(Error err) -> cstr_t {
  const auto code = cudaError_t(err);
  const auto name = ::cudaGetErrorName(code);
  return name;
}

auto device_get() -> Result<u32> {
  auto dev = 0;
  if (auto err = ::cudaGetDevice(&dev)) {
    return Err{Error(err)};
  }
  return Ok{num::cast_unsigned(dev)};
}

auto device_set(u32 dev_id) -> Result<> {
  const auto device = num::cast_signed(dev_id);
  if (auto err = ::cudaSetDevice(device)) {
    return Err{Error(err)};
  }
  return Ok{};
}

auto device_prop(u32 dev) -> Result<cudaDeviceProp> {
  const auto device = num::cast_signed(dev);

  auto prop = cudaDeviceProp{};
  if (auto err = ::cudaGetDeviceProperties(&prop, device)) {
    return Err{Error(err)};
  }
  return Ok{prop};
}

auto device_count() -> u32 {
  auto cnt = 0;
  if (auto err = ::cudaGetDeviceCount(&cnt); err != cudaSuccess) {
    return 0;
  }
  return num::cast_unsigned(cnt);
}

auto device_sync() -> Result<> {
  if (auto err = ::cudaDeviceSynchronize()) {
    return Err{Error(err)};
  }
  return Ok{};
}

auto Device::current() -> Device {
  const auto id = cuda::device_get().unwrap();
  return Device{id};
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

void DeviceInfo::fmt(fmt::Formatter& f) const {
  auto imp = f.debug_struct("DeviceInfo");
  imp.field("name", name);
  imp.field("dev_id", dev_id);
  imp.field("compute_capability", compute_capability);
  imp.field("sm_count", sm_count);
  imp.field("async_engine_count", async_engine_count);
  imp.field("global_memory", global_memory);
  imp.field("l2_cache_size", l2_cache_size);
}

auto Device::scope() -> ScopeGuard {
  const auto curr_id = cuda::device_get().unwrap();
  const auto next_id = this->id;
  return ScopeGuard{curr_id, next_id};
}

Device::ScopeGuard::ScopeGuard(u32 enter, u32 exit) : _dev_enter{enter}, _dev_exit{exit} {
  cuda::device_set(_dev_enter).unwrap();
}

Device::ScopeGuard::~ScopeGuard() {
  cuda::device_set(_dev_exit).unwrap();
}

}  // namespace sfc::cuda
