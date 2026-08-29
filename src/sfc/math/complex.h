#pragma once

#include "sfc/math/mod.h"

namespace sfc::math {

struct c32 {
  f32 real;
  f32 imag;

 public:
  __hd auto operator-() const -> c32 {
    return c32{-real, -imag};
  }

  __hd auto operator+(c32 b) const -> c32 {
    return c32{real + b.real, imag + b.imag};
  }

  __hd auto operator-(c32 b) const -> c32 {
    return c32{real - b.real, imag - b.imag};
  }

  __hd auto operator*(c32 b) const -> c32 {
    return c32{real * b.real - imag * b.imag, real * b.imag + imag * b.real};
  }

  __hd auto operator/(c32 b) const -> c32 {
    f32 denom = b.real * b.real + b.imag * b.imag;
    return c32{(real * b.real + imag * b.imag) / denom, (imag * b.real - real * b.imag) / denom};
  }

 public:
  __hd auto operator==(const c32& b) const -> bool {
    return real == b.real && imag == b.imag;
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

 public:
  void fmt(auto& f) const {
    auto spec = f.spec();
    spec._sign = '+';
    f.write_val(real);
    f.write_val(imag);
    f.write_char('i');
  }
};

__hd inline auto expj(f32 theta) -> c32 {
  return c32{math::cosf(theta), math::sinf(theta)};
}

__hd inline auto conj(c32 z) -> c32 {
  return c32{z.real, -z.imag};
}

__hd inline auto operator*(f32 k, c32 z) -> c32 {
  return c32{k * z.real, k * z.imag};
}

__hd inline auto operator/(c32 c, f32 k) -> c32 {
  return c32{c.real / k, c.imag / k};
}

}  // namespace sfc::math

namespace sfc {
using math::c32;
}  // namespace sfc
