#include "sfc/test.h"
#include "sfc/io.h"
#include "sfc/math/geo.h"

namespace sfc::math::geo::test {

SFC_TEST(rot) {
  const auto rot_0 = Rot::identity();
  sfc::assert_flt_eq(rot_0.cos, 1.0f);
  sfc::assert_flt_eq(rot_0.sin, 0.0f);

  const auto rot_90 = Rot::from_deg(90.0f);
  sfc::assert_flt_eq(rot_90.cos, 0.0f, 1e-6);
  sfc::assert_flt_eq(rot_90.sin, 1.0f, 1e-6);

  const auto rot_neg_90 = -rot_90;
  sfc::assert_flt_eq(rot_neg_90.cos, 0.0f, 1e-6);
  sfc::assert_flt_eq(rot_neg_90.sin, -1.0f, 1e-6);

  const auto rot_pi = Rot::from_rad(math::PI);
  sfc::assert_flt_eq(rot_pi.cos, -1.0f, 1e-6);
  sfc::assert_flt_eq(rot_pi.sin, 0.0f, 1e-6);
}

SFC_TEST(rot_trans) {
  const auto a = vec2f{1.0f, 1.0f};

  const auto rot_90 = Rot::from_deg(90.0f);
  const auto b = rot_90(a);
  sfc::assert_flt_eq(b.x, -1.0f, 1e-6);
  sfc::assert_flt_eq(b.y, +1.0f, 1e-6);
}

SFC_TEST(rot_fmt) {
  io::println("rot(90deg) = (cos={}, sin={})\n", Rot::from_deg(90.0f).cos, Rot::from_deg(90.0f).sin);
}

}  // namespace sfc::math::test
