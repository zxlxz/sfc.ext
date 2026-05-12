#pragma once

#include "sfc/math/mod.h"

namespace sfc::math {

struct c32 {
  f32 real;
  f32 imag;

 public:
  _hd auto operator~() const -> c32 {
    return c32{real, -imag};
  }

  void fmt(auto& f) const {
    f.write_val(real);
    auto spec = f._spec;
    spec._sign = '+';
    f.write_arg(spec, imag);
    f.write_str("j");
  }
};

_hd inline auto expj(f32 theta) -> c32 {
  const auto c = __builtin_cosf(theta);
  const auto s = __builtin_sinf(theta);
  return c32{c, s};
}

_hd inline auto conj(c32 c) -> c32 {
  return c32{c.real, -c.imag};
}

_hd inline auto operator+(c32 a, c32 b) -> c32 {
  return c32{a.real + b.real, a.imag + b.imag};
}

_hd inline auto operator-(c32 a, c32 b) -> c32 {
  return c32{a.real - b.real, a.imag - b.imag};
}

_hd inline auto operator*(c32 a, c32 b) -> c32 {
  return c32{a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real};
}

_hd inline auto operator*(f32 k, c32 c) -> c32 {
  return c32{k * c.real, k * c.imag};
}

}  // namespace sfc::math

namespace sfc {
using math::c32;
}  // namespace sfc
