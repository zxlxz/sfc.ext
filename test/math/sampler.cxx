#include "sfc/test.h"
#include "sfc/math/ndslice.h"
#include "sfc/math/sampler.h"

namespace sfc::math::test {

template <class T, int N>
auto make_slice1d(T (&v)[N]) -> NdSlice<T, 1> {
  return NdSlice<T, 1>{v, {N}, {1}};
}

template <class T, int NX, int NY>
auto make_slice2d(T (&v)[NY][NX]) -> NdSlice<T, 2> {
  return NdSlice<T, 2>{v[0], {NX, NY}, {1, NX}};
}

SFC_TEST(nearest_sampler_1d) {
  const int v[] = {1, 2, 3, 4};
  const auto s = Sampler{make_slice1d(v)};
  for (auto i = 0; i < 4; ++i) {
    sfc::expect_eq(s.load_nearest(f32(i)), v[i]);
    sfc::expect_eq(s.load_nearest(f32(i) + 0.5f), v[i]);
  }
}

SFC_TEST(nearest_sampler_2d) {
  const int v[2][3] = {{1, 2, 3}, {4, 5, 6}};
  const auto s = Sampler{make_slice2d(v)};

  for (auto j = 0; j < 2; ++j) {
    for (auto i = 0; i < 3; ++i) {
      const auto t = v[j][i];
      sfc::expect_eq(s.load_nearest({f32(i), f32(j)}), t);
      sfc::expect_eq(s.load_nearest({f32(i) + 0.5f, f32(j) + 0.5f}), t);
    }
  }
}

SFC_TEST(linear_sampler_1d) {
  const f32 v[] = {1, 2, 3};
  const auto s = Sampler{make_slice1d(v)};

  sfc::expect_eq(s.load_linear(-1), 0.f);

  sfc::expect_eq(s.load_linear(0.0f), 1.0f);
  sfc::expect_eq(s.load_linear(0.1f), 1.0f);
  sfc::expect_eq(s.load_linear(0.5f), 1.0f);

  sfc::expect_eq(s.load_linear(1), 1.5f);
  sfc::expect_eq(s.load_linear(1.5f), 2.0f);

  sfc::expect_eq(s.load_linear(2), 2.5f);
  sfc::expect_eq(s.load_linear(2.5f), 3.0f);
  sfc::expect_eq(s.load_linear(2.9f), 3.0f);

  sfc::expect_eq(s.load_linear(3.0f), 3.0f);
  sfc::expect_eq(s.load_linear(3.1f), 0.0f);
}

SFC_TEST(linear_sampler_2d) {
  f32 v[2][2] = {{1, 2}, {3, 4}};
  const auto s = Sampler{make_slice2d(v)};

  sfc::expect_eq(s.load_linear({-1, -1}), 0.f);

  // edge
  sfc::expect_eq(s.load_linear({0, 0}), 1.f);
  sfc::expect_eq(s.load_linear({1, 0}), 1.5f);
  sfc::expect_eq(s.load_linear({2, 0}), 2.0f);

  // inner
  sfc::expect_eq(s.load_linear({0.5f, 0.5f}), 1.0f);
  sfc::expect_eq(s.load_linear({0.5f, 1.0f}), 2.0f);
  sfc::expect_eq(s.load_linear({1.0f, 0.5f}), 1.5f);
  sfc::expect_eq(s.load_linear({1.0f, 1.0f}), 2.5f);

  // edge
  sfc::expect_eq(s.load_linear({2.0f, 2.0f}), 4.0f);
}

}  // namespace sfc::math::test
