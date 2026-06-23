#include "sfc/test.h"
#include "sfc/math/ndview.h"
#include "sfc/math/sampler.h"

namespace sfc::math::test {

SFC_TEST(nearest_sampler_1d) {
  const int v[] = {1, 2, 3, 4};
  const auto s = Sampler{NdView{v}};
  for (auto i = 0; i < 4; ++i) {
    sfc::assert_eq(s.load_nearest(f32(i)), v[i]);
    sfc::assert_eq(s.load_nearest(f32(i) + 0.5f), v[i]);
  }
}

SFC_TEST(nearest_sampler_2d) {
  const int v[2][2] = {{1, 2}, {3, 4}};
  const auto s = Sampler{NdView(v)};

  sfc::assert_eq(s.load_nearest({0, 0}), v[0][0]);
  sfc::assert_eq(s.load_nearest({0, 1}), v[0][1]);
  sfc::assert_eq(s.load_nearest({1, 0}), v[1][0]);
  sfc::assert_eq(s.load_nearest({1, 1}), v[1][1]);
}

SFC_TEST(linear_sampler_1d) {
  const f32 v[] = {1, 2, 3};
  const auto s = Sampler{NdView{v}};

  sfc::assert_eq(s.load_linear(-1), 0.f);

  sfc::assert_eq(s.load_linear(0.0f), 1.0f);
  sfc::assert_eq(s.load_linear(0.1f), 1.0f);
  sfc::assert_eq(s.load_linear(0.5f), 1.0f);

  sfc::assert_eq(s.load_linear(1), 1.5f);
  sfc::assert_eq(s.load_linear(1.5f), 2.0f);

  sfc::assert_eq(s.load_linear(2), 2.5f);
  sfc::assert_eq(s.load_linear(2.5f), 3.0f);
  sfc::assert_eq(s.load_linear(2.9f), 3.0f);

  sfc::assert_eq(s.load_linear(3.0f), 3.0f);
  sfc::assert_eq(s.load_linear(3.1f), 0.0f);
}

SFC_TEST(linear_sampler_2d) {
  f32 v[2][2] = {{1, 2}, {3, 4}};
  const auto s = Sampler{NdView{v}};

  sfc::assert_eq(s.load_linear({-1, -1}), 0.f);

  // edge
  sfc::assert_eq(s.load_linear({0, 0}), v[0][0]);
  sfc::assert_eq(s.load_linear({0, 1}), 0.5f * v[0][0] + 0.5f * v[0][1]);
  sfc::assert_eq(s.load_linear({0, 2}), v[0][1]);

  // inner
  sfc::assert_eq(s.load_linear({0.5f, 0.5f}), v[0][0]);
  sfc::assert_eq(s.load_linear({0.5f, 1.5f}), v[0][1]);
  sfc::assert_eq(s.load_linear({1.5f, 0.5f}), v[1][0]);
  sfc::assert_eq(s.load_linear({1.5f, 1.5f}), v[1][1]);

  sfc::assert_eq(s.load_linear({0.5f, 0.0f}), v[0][0]);
  sfc::assert_eq(s.load_linear({0.5f, 1.0f}), 0.5f * v[0][0] + 0.5f * v[0][1]);
  sfc::assert_eq(s.load_linear({0.5f, 2.0f}), v[0][1]);

  sfc::assert_eq(s.load_linear({0.0f, 0.5f}), v[0][0]);
  sfc::assert_eq(s.load_linear({1.0f, 0.5f}), 0.5f * v[0][0] + 0.5f * v[1][0]);
  sfc::assert_eq(s.load_linear({2.0f, 0.5f}), v[1][0]);

  sfc::assert_eq(s.load_linear({1.0f, 1.0f}), 0.25f * v[0][0] + 0.25f * v[0][1] + 0.25f * v[1][0] + 0.25f * v[1][1]);

  // edge
  sfc::assert_eq(s.load_linear({2.0f, 2.0f}), 4.0f);
}

}  // namespace sfc::math::test
