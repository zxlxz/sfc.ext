#include "sfc/test.h"
#include "sfc/io.h"
#include "sfc/math/complex.h"

namespace sfc::math::complex::test {

SFC_TEST(complex_add) {
  const auto r = c32{1.0f, 2.0f} + c32{3.0f, 4.0f};
  sfc::assert_flt_eq(r.real, 4.0f);
  sfc::assert_flt_eq(r.imag, 6.0f);
}

SFC_TEST(complex_sub) {
  const auto r = c32{5.0f, 7.0f} - c32{2.0f, 3.0f};
  sfc::assert_flt_eq(r.real, 3.0f);
  sfc::assert_flt_eq(r.imag, 4.0f);
}

SFC_TEST(complex_mul) {
  // (1+2j) * (3+4j) = -5 + 10j
  const auto r = c32{1.0f, 2.0f} * c32{3.0f, 4.0f};
  sfc::assert_flt_eq(r.real, -5.0f);
  sfc::assert_flt_eq(r.imag, 10.0f);
}

SFC_TEST(complex_scalar_mul) {
  const auto r = 2.0f * c32{3.0f, 4.0f};
  sfc::assert_flt_eq(r.real, 6.0f);
  sfc::assert_flt_eq(r.imag, 8.0f);
}

SFC_TEST(complex_conj) {
  const auto r = conj(c32{3.0f, 4.0f});
  sfc::assert_flt_eq(r.real, 3.0f);
  sfc::assert_flt_eq(r.imag, -4.0f);

  const auto neg = ~c32{3.0f, 4.0f};
  sfc::assert_flt_eq(neg.imag, -4.0f);
}

SFC_TEST(complex_compound_ops) {
  auto a = c32{1.0f, 1.0f};
  a += c32{2.0f, 3.0f};
  sfc::assert_flt_eq(a.real, 3.0f);
  sfc::assert_flt_eq(a.imag, 4.0f);

  a -= c32{1.0f, 1.0f};
  sfc::assert_flt_eq(a.real, 2.0f);
  sfc::assert_flt_eq(a.imag, 3.0f);

  a *= 2.0f;
  sfc::assert_flt_eq(a.real, 4.0f);
  sfc::assert_flt_eq(a.imag, 6.0f);
}

SFC_TEST(complex_expj) {
  // expj(0) = 1 + 0j
  const auto e0 = expj(0.0f);
  sfc::assert_flt_eq(e0.real, 1.0f);
  sfc::assert_flt_eq(e0.imag, 0.0f);

  // expj(pi/2) = 0 + 1j
  const auto e90 = expj(math::PI / 2.0f);
  sfc::assert_flt_eq(e90.real, 0.0f, 1e-6);
  sfc::assert_flt_eq(e90.imag, 1.0f, 1e-6);
}

SFC_TEST(complex_fmt) {
  io::println("c32 = {}\n", c32{1.0f, -2.0f});
}

}  // namespace sfc::math::test
