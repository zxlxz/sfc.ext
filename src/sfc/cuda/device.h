#pragma once

#include "sfc/cuda/mod.h"

namespace sfc::cuda {

struct DeviceInfo {
  u32 dev_id;
  u32 compute_capability;
  u32 sm_count;
  u32 async_engine_count;
  u64 global_memory;
  u64 l2_cache_size;
  const char* name;

 public:
  void fmt(fmt::Formatter& f) const;
};

struct Device {
  u32 id = 0;

 public:
  static auto current() -> Device;

 public:
  auto info() const -> DeviceInfo;

  class ScopeGuard;
  auto scope() -> ScopeGuard;
};

class Device::ScopeGuard {
  u32 _dev_enter;
  u32 _dev_exit;

 public:
  ScopeGuard(u32 enter, u32 exit);
  ~ScopeGuard();

  ScopeGuard(const ScopeGuard&) = delete;
  ScopeGuard& operator=(const ScopeGuard&) = delete;
};

auto device_count() -> u32;
auto device_sync() -> Result<>;

auto device_get() -> Result<u32>;
auto device_set(u32 idx) -> Result<>;

}  // namespace sfc::cuda
