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

class DeviceGuard {
  int _dev_enter;
  int _dev_exit;

 public:
  DeviceGuard(int enter_id, int exit_id);
  ~DeviceGuard();

  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;
};

struct Device {
  u32 id = 0;

 public:
  static auto count() -> u32;
  static auto current() -> Device;
  static auto sync() -> Result<>;

 public:
  auto info() const -> DeviceInfo;
  auto scope() -> DeviceGuard;
};

}  // namespace sfc::cuda
