#pragma once

#include "sfc/core.h"

namespace sfc::cuda {

struct Error {
  int _code = 0;

 public:
  auto to_str() const -> cstr_t;

  void fmt(auto& f) const {
    const auto s = this->to_str();
    f.write_str(s);
  }
};

}  // namespace sfc::cuda
