#include "sfc/test.h"
#include "sfc/io.h"
#include "sfc/math/complex.h"

namespace sfc::math::complex::test {

SFC_TEST(complex_binary_ops) {
  const auto a = c32{1.0f, 2.0f};
  const auto b = c32{3.0f, 4.0f};

  // a+b
  {
    const auto c = a + b;
    sfc::assert_eq(c, c32{4.0f, 6.0f});
  }

  // a-b
  {
    const auto c = a - b;
    sfc::assert_eq(c, c32{-2.0f, -2.0f});
  }

  // a*b
  {
    const auto c = a * b;
    sfc::assert_eq(c, c32{-5.0f, 10.0f});
  }

  // a/b
  {
    const auto c = a / c32{1, 0};
    sfc::assert_eq(c, a);

    const auto d = a / c32{0, 1};
    sfc::assert_eq(d, c32{2.0f, -1.0f});
  }
}

SFC_TEST(complex_scalar_ops) {
  const auto a = c32{1.0f, 2.0f};
  const auto k = 2.0f;

  // a*k
  {
    const auto c = k * a;
    sfc::assert_eq(c, c32{2.0f, 4.0f});
  }

  // a/k
  {
    const auto c = a / k;
    sfc::assert_eq(c, c32{0.5f, 1.0f});
  }
}

SFC_TEST(complex_unary_ops) {
  const auto a = c32{1.0f, 2.0f};

  // -a
  {
    const auto c = -a;
    sfc::assert_eq(c, c32{-1.0f, -2.0f});
  }
}

SFC_TEST(complex_assign_ops) {
  const auto a = c32{1.0f, 2.0f};
  const auto b = c32{3.0f, 4.0f};

  // a += b
  {
    auto c = a;
    c += b;
    sfc::assert_eq(c, c32{4.0f, 6.0f});
  }

  // a -= b
  {
    auto c = a;
    c -= b;
    sfc::assert_eq(c, c32{-2.0f, -2.0f});
  }
}

SFC_TEST(expj) {
  const auto theta = math::PI / 2;
  const auto z = expj(f32(theta));

  sfc::assert_flt_eq(z.real, 0.0f, 1e-4f);
  sfc::assert_flt_eq(z.imag, 1.0f, 1e-4f);
}

SFC_TEST(complex_fmt) {
  io::println("c32 = {}\n", c32{1.0f, -2.0f});
}

}  // namespace sfc::math::complex::test
