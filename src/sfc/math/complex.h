#pragma once

#include "sfc/math/mod.h"

namespace sfc::math {

struct c32 {
  f32 real;
  f32 imag = 0.0;

 public:
  void fmt(auto& f) const {
    auto spec = f._spec;
    spec._sign = '+';
    f.write_arg(spec, real);
    f.write_arg(spec, imag);
    f.write_str("i");
  }
};

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
