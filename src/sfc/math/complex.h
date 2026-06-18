#pragma once

#include "sfc/math/mod.h"

namespace sfc::math {

struct c32 {
  f32 real;
  f32 imag;

 public:
  __hd auto operator~() const -> c32 {
    return c32{real, -imag};
  }

  __hd auto operator+=(c32 b) -> c32& {
    real += b.real;
    imag += b.imag;
    return *this;
  }

  __hd auto operator-=(c32 b) -> c32& {
    real -= b.real;
    imag -= b.imag;
    return *this;
  }

  __hd auto operator*=(f32 k) -> c32& {
    real *= k;
    imag *= k;
    return *this;
  }

 public:
  void fmt(auto& f) const {
    auto spec = f._spec;
    spec._sign = '+';

    f.write_val(real);
    f._spec = spec;
    f.write_val(imag);
    f.write_str("j");
  }
};

__hd inline auto expj(f32 theta) -> c32 {
  return c32{math::cosf(theta), math::sinf(theta)};
}

__hd inline auto conj(c32 c) -> c32 {
  return c32{c.real, -c.imag};
}

__hd inline auto operator+(c32 a, c32 b) -> c32 {
  return c32{a.real + b.real, a.imag + b.imag};
}

__hd inline auto operator-(c32 a, c32 b) -> c32 {
  return c32{a.real - b.real, a.imag - b.imag};
}

__hd inline auto operator*(c32 a, c32 b) -> c32 {
  return c32{a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real};
}

__hd inline auto operator*(f32 k, c32 c) -> c32 {
  return c32{k * c.real, k * c.imag};
}

}  // namespace sfc::math

namespace sfc {
using math::c32;
}  // namespace sfc
