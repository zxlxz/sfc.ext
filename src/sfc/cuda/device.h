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
  static auto try_from(u32 id) -> Result<Device>;

 public:
  struct Info;
  auto info() const -> Result<Info>;

 public:
  class Entered;
  auto enter() -> Entered;
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

class Device::Entered {
  int _dev_in;   // in enter
  int _dev_out;  // out enter

 public:
  explicit Entered(const Device& dev);
  ~Entered();

  Entered(const Entered&) = delete;
  Entered& operator=(const Entered&) = delete;
};

}  // namespace sfc::cuda
