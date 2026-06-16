#pragma once

#include "sfc/core.h"

namespace sfc::cuda {

auto dev_count() -> u32;
auto device_get() -> u32;
void device_set(u32 dev_id);
void device_sync();

struct Device {
  u32 id = 0;

 public:
  static auto current() -> Device;

  auto name() const -> Str;
  auto compute_capability() const -> u32;
  auto sm_count() const -> u32;
  auto global_memory() const -> u64;
  auto l2_cache_size() const -> u64;
  auto async_engine_count() const -> u32;
};

}  // namespace sfc::cuda
