#pragma once

#include "sfc/core/mod.h"

namespace sfc::cuda {

void init();

auto dev_count() -> int;
auto device_get() -> int;
void device_set(int);
void device_sync();

struct Device {
  int id = 0;

 public:
  static auto current() -> Device;
  auto name() const -> const char*;
  auto total_memory() const -> usize;
};

}  // namespace sfc::cuda
