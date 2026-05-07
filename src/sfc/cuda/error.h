#pragma once

#include "sfc/cuda/mod.h"

namespace sfc::cuda {

struct Error {
  int _code = 0;

 public:
  auto name() const noexcept -> const char*;
};

}  // namespace sfc::cuda
