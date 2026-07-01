
#include "sfc/cuda/mod.inl"
#include "sfc/cuda/device.h"
#include "sfc/cuda/stream.h"

namespace sfc::cuda {

auto error_name(cudaError_t code) -> cstr_t {
  const auto name = ::cudaGetErrorName(code);
  return name;
}

auto dev_count() -> u32 {
  auto cnt = 0;
  CHECK_RET(cudaGetDeviceCount, &cnt);
  return num::cast_unsigned(cnt);
}

auto device_get() -> u32 {
  auto dev = 0;
  CHECK_RET(cudaGetDevice, &dev);
  return num::cast_unsigned(dev);
}

void device_set(u32 dev_id) {
  const auto device = num::cast_signed(dev_id);
  CHECK_RET(cudaSetDevice, device);
}

void device_sync() {
  CHECK_RET(cudaDeviceSynchronize);
}

auto device_prop(u32 dev) -> const cudaDeviceProp& {
  static constexpr auto kMaxDevCount = 16U;
  static cudaDeviceProp props[kMaxDevCount] = {};
  if (dev >= kMaxDevCount) {
    return props[kMaxDevCount - 1];
  }

  const auto is_init = props[dev].totalGlobalMem != 0;
  if (!is_init) {
    const auto device = num::cast_signed(dev);
    CHECK_RET(cudaGetDeviceProperties, &props[dev], device);
  }

  return props[dev];
}

auto Device::current() -> Device {
  const auto id = cuda::device_get();
  return Device{id};
}

auto Device::name() const -> Str {
  const auto& p = cuda::device_prop(id);
  return Str::from_cstr(p.name);
}

auto Device::compute_capability() const -> u32 {
  const auto& p = cuda::device_prop(id);
  const auto ret = p.major * 10 + p.minor;
  return num::cast_unsigned(ret);
}
auto Device::sm_count() const -> u32 {
  const auto& p = cuda::device_prop(id);
  return num::cast_unsigned(p.multiProcessorCount);
}
auto Device::global_memory() const -> u64 {
  const auto& p = cuda::device_prop(id);
  return p.totalGlobalMem;
}
auto Device::l2_cache_size() const -> u64 {
  const auto& p = cuda::device_prop(id);
  return num::cast_unsigned(p.l2CacheSize);
}
auto Device::async_engine_count() const -> u32 {
  const auto& p = cuda::device_prop(id);
  return num::cast_unsigned(p.asyncEngineCount);
}

}  // namespace sfc::cuda
