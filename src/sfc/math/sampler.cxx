#include "sfc/test.h"
#include "sfc/math/ndslice.h"
#include "sfc/math/sampler.h"

namespace sfc::math::sampler::test {

template <class T, int N0>
auto as_slice_1d(T (&v)[N0]) -> NdSlice<T, 1> {
  return {v, {N0}, {1}};
}

template <class T, int N0, int N1>
auto as_slice_2d(T (&v)[N0][N1]) -> NdSlice<T, 2> {
  return {v[0], {N0, N1}, {N1, 1}};
}

SFC_TEST(nearest_sampler_1d) {
  int buf[] = {1, 2};

  const auto v = as_slice_1d(buf);
  const auto s = NearestSampler{v};

  // (..., 0) -> 0
  sfc::assert_eq(s.load_1d(-1.0f), 0);

  // [0, 1) -> 0
  sfc::assert_eq(s.load_1d(0.0f), v[0]);

  // [1, 2) -> [1]
  sfc::assert_eq(s.load_1d(1.0f), v[1]);
  sfc::assert_eq(s.load_1d(1.5f), v[1]);

  // [2, ...) -> 0
  sfc::assert_eq(s.load_1d(2.0f), 0);
  sfc::assert_eq(s.load_1d(2.1f), 0);
}

SFC_TEST(nearest_sampler_2d) {
  int buf[2][2] = {{1, 2}, {3, 4}};

  const auto m = as_slice_2d(buf);
  const auto s = NearestSampler{m};

  sfc::assert_eq(s.load_2d(0, 0), m[0][0]);
  sfc::assert_eq(s.load_2d(0, 1), m[0][1]);
  sfc::assert_eq(s.load_2d(1, 0), m[1][0]);
  sfc::assert_eq(s.load_2d(1, 1), m[1][1]);
}

SFC_TEST(linear_sampler_1d) {
  f32 buf[] = {1, 2, 3};

  const auto v = as_slice_1d(buf);
  const auto s = LinearSampler{v};

  // (..., 0) -> 0
  sfc::assert_eq(s.load_1d(-1.0f), 0.f);

  // (0, 0.5) -> v[0]
  sfc::assert_eq(s.load_1d(0.0f), v[0]);
  sfc::assert_eq(s.load_1d(0.5f), v[0]);

  // (0.5, 1.5) -> [0, 1]
  sfc::assert_eq(s.load_1d(1.0f), 0.5f * v[0] + 0.5f * v[1]);

  // (1.5, 2.5) -> [1, 2]
  sfc::assert_eq(s.load_1d(2.0f), 0.5f * v[1] + 0.5f * v[2]);

  // (2.5, 3) -> v[2]
  sfc::assert_eq(s.load_1d(2.5f), v[2]);
  sfc::assert_eq(s.load_1d(3.0f), v[2]);

  // (3, ...) -> 0
  sfc::assert_eq(s.load_1d(3.1f), 0.f);
  sfc::assert_eq(s.load_1d(4.0f), 0.f);
}

SFC_TEST(linear_sampler_2d) {
  f32 buff[2][2] = {{1, 2}, {3, 4}};

  const auto m = as_slice_2d(buff);
  const auto s = LinearSampler{m};

  // (-1, -1)-> 0
  sfc::assert_eq(s.load_2d(-1.0f, -1.0f), 0.f);

  // (0, 0) -> m[0][0]
  sfc::assert_eq(s.load_2d(0.0f, 0.0f), m[0][0]);

  // (0, 0.5) -> m[0][0]
  sfc::assert_eq(s.load_2d(0.0f, 0.5f), m[0][0]);

  // (0.5, 0) -> m[0][0]
  sfc::assert_eq(s.load_2d(0.5f, 0.0f), m[0][0]);

  // (0.5, 0.5) -> m[0][0]
  sfc::assert_eq(s.load_2d(0.5f, 0.5f), m[0][0]);

  // (0, 1) -> m[0][0] * 0.5 + m[0][1] * 0.5
  sfc::assert_eq(s.load_2d(0.5f, 1.0f), 0.5f * m[0][0] + 0.5f * m[0][1]);

  // (1, 0) -> m[0][0] * 0.5 + m[1][0] * 0.5
  sfc::assert_eq(s.load_2d(1.0f, 0.5f), 0.5f * m[0][0] + 0.5f * m[1][0]);

  // (1, 1) -> m[0][0] * 0.25 + m[0][1] * 0.25 + m[1][0] * 0.25 + m[1][1] * 0.25
  sfc::assert_eq(s.load_2d(1.0f, 1.0f), 0.25f * m[0][0] + 0.25f * m[0][1] + 0.25f * m[1][0] + 0.25f * m[1][1]);

  // (1, 2) -> m[0][1] * 0.5 + m[1][1] * 0.5
  sfc::assert_eq(s.load_2d(1.0f, 2.0f), 0.5f * m[0][1] + 0.5f * m[1][1]);

  // (2, 2) -> m[1][1]
  sfc::assert_eq(s.load_2d(2.0f, 2.0f), m[1][1]);
}

}  // namespace sfc::math::test
