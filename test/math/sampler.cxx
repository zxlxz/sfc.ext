#include "sfc/test.h"
#include "sfc/math/ndview.h"
#include "sfc/math/sampler.h"

namespace sfc::math::test {

template <class T, int N0>
auto as_slice_1d(T (&v)[N0]) -> NdSlice<T, 1> {
  return {v, {N0}, {1}};
}

template <class T, int N0, int N1>
auto as_slice_2d(T (&v)[N0][N1]) -> NdSlice<T, 2> {
  return {v[0], {N0, N1}, {N1, 1}};
}

SFC_TEST(nearest_sampler_1d) {
  int buf[] = {1, 2, 3, 4};

  const auto col = as_slice_1d(buf);
  const auto s = Sampler{col};

  sfc::assert_eq(s.load_nearest(-1.0f), col[0]);
  sfc::assert_eq(s.load_nearest(0.0f), col[0]);
  sfc::assert_eq(s.load_nearest(1.0f), col[1]);
  sfc::assert_eq(s.load_nearest(1.1f), col[1]);

  sfc::assert_eq(s.load_nearest(2.9f), col[2]);
  sfc::assert_eq(s.load_nearest(2.0f), col[2]);

  sfc::assert_eq(s.load_nearest(3.0f), col[3]);
  sfc::assert_eq(s.load_nearest(4.0f), col[3]);
}

SFC_TEST(nearest_sampler_2d) {
  int buf[2][2] = {{1, 2}, {3, 4}};

  const auto mat = as_slice_2d(buf);
  const auto s = Sampler{mat};

  sfc::assert_eq(s.load_nearest({0, 0}), mat[0][0]);
  sfc::assert_eq(s.load_nearest({0, 1}), mat[0][1]);
  sfc::assert_eq(s.load_nearest({1, 0}), mat[1][0]);
  sfc::assert_eq(s.load_nearest({1, 1}), mat[1][1]);
}

SFC_TEST(linear_sampler_1d) {
  f32 buf[] = {1, 2, 3};

  const auto col = as_slice_1d(buf);
  const auto s = Sampler{col};

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
  f32 buff[2][2] = {{1, 2}, {3, 4}};

  const auto mat = as_slice_2d(buff);
  const auto s = Sampler{mat};

  sfc::assert_eq(s.load_linear({-1, -1}), 0.f);

  // edge
  sfc::assert_eq(s.load_linear({0, 0}), mat[0][0]);
  sfc::assert_eq(s.load_linear({0, 1}), 0.5f * mat[0][0] + 0.5f * mat[0][1]);
  sfc::assert_eq(s.load_linear({0, 2}), mat[0][1]);

  // inner
  sfc::assert_eq(s.load_linear({0.5f, 0.5f}), mat[0][0]);
  sfc::assert_eq(s.load_linear({0.5f, 1.5f}), mat[0][1]);
  sfc::assert_eq(s.load_linear({1.5f, 0.5f}), mat[1][0]);
  sfc::assert_eq(s.load_linear({1.5f, 1.5f}), mat[1][1]);

  sfc::assert_eq(s.load_linear({0.5f, 0.0f}), mat[0][0]);
  sfc::assert_eq(s.load_linear({0.5f, 1.0f}), 0.5f * mat[0][0] + 0.5f * mat[0][1]);
  sfc::assert_eq(s.load_linear({0.5f, 2.0f}), mat[0][1]);

  sfc::assert_eq(s.load_linear({0.0f, 0.5f}), mat[0][0]);
  sfc::assert_eq(s.load_linear({1.0f, 0.5f}), 0.5f * mat[0][0] + 0.5f * mat[1][0]);
  sfc::assert_eq(s.load_linear({2.0f, 0.5f}), mat[1][0]);

  sfc::assert_eq(s.load_linear({1.0f, 1.0f}),
                 0.25f * mat[0][0] + 0.25f * mat[0][1] + 0.25f * mat[1][0] + 0.25f * mat[1][1]);

  // edge
  sfc::assert_eq(s.load_linear({2.0f, 2.0f}), 4.0f);
}

}  // namespace sfc::math::test
