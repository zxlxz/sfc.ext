#pragma once

#include "sfc/core/mod.h"

namespace sfc::cuda {

struct Error {
  int _code = 0;

 public:
  auto name() const noexcept -> const char*;

  void fmt(auto& f) const {
    const auto s = this->name();
    f.write_str(s);
  }
};

}  // namespace sfc::cuda
