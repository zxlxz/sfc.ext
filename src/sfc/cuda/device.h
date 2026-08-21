#pragma once

#include "sfc/cuda/mod.h"

namespace sfc::cuda {

auto init() -> Result<>;

struct Device {
  int _dev = -1;

 public:
  static auto count() -> Result<u32>;
  static auto current() -> Result<Device>;
  static auto sync() -> Result<>;
  static auto of(u32 id) -> Result<Device>;

 public:
  struct Info;
  auto info() const -> Result<Info>;

 public:
  class ScopeGuard;
  auto scope() -> ScopeGuard;
};

struct Device::Info {
  i32 device;
  u32 compute_capability;
  u32 sm_count;
  u32 async_engine_count;
  u32 l2_cache_size;
  u64 global_memory;
  Str name;

 public:
  void fmt(fmt::Formatter& f) const;
};

class Device::ScopeGuard {
  int _dev_in;   // in scope
  int _dev_out;  // out scope

 public:
  explicit ScopeGuard(const Device& dev);
  ~ScopeGuard();

  ScopeGuard(const ScopeGuard&) = delete;
  ScopeGuard& operator=(const ScopeGuard&) = delete;
};

}  // namespace sfc::cuda
