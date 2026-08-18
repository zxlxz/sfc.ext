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

class DeviceGuard;

struct Device {
  u32 id = 0;

 public:
  static auto count() -> Result<u32>;
  static auto current() -> Result<Device>;
  static auto sync() -> Result<>;

 public:
  auto info() const -> Result<DeviceInfo>;

 public:
  class Guard;
  auto scope() -> Guard;
};

class Device::Guard {
  u32 _dev_in;   // in scope
  u32 _dev_out;  // out scope

 public:
  Guard(u32 id);
  ~Guard();

  Guard(const Guard&) = delete;
  Guard& operator=(const Guard&) = delete;
};

}  // namespace sfc::cuda
