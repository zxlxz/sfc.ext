#include "sfc/test.h"
#include "sfc/math/ndslice.h"

namespace sfc::math::test {

template <class T, int N>
auto make_slice1d(T (&v)[N]) -> NdSlice<T, 1> {
  return NdSlice<T, 1>{v, {N}, {1}};
}

template <class T, int NX, int NY>
auto make_slice2d(T (&v)[NY][NX]) -> NdSlice<T, 2> {
  return NdSlice<T, 2>{v[0], {NX, NY}, {1, NX}};
}

SFC_TEST(slice_properties) {
  // slice 1d
  {
    int v[] = {1, 2, 3, 4};
    const auto s = make_slice1d(v);
    sfc::expect_eq(s.len(), 4U);
    sfc::expect_eq(s.shape(), vec1u{4});
    sfc::expect_eq(s.strides(), vec1u{1});
    sfc::expect_eq(s.numel(), 4U);
  }

  // slice 2d
  {
    int v[2][3] = {{1, 2, 3}, {4, 5, 6}};
    const auto s = make_slice2d(v);
    sfc::expect_eq(s.len(), 2U);
    sfc::expect_eq(s.shape(), vec2u{3, 2});
    sfc::expect_eq(s.strides(), vec2u{1, 3});
    sfc::expect_eq(s.numel(), 6U);
  }
}

SFC_TEST(slice_visit) {
  // slice 1d
  {
    int v[] = {1, 2, 3, 4};
    const auto s = make_slice1d(v);
    sfc::expect_eq(s[0], 1);
    sfc::expect_eq(s[1], 2);
    sfc::expect_eq(s[2], 3);
    sfc::expect_eq(s[3], 4);
  }

  // slice 2d
  {
    int v[2][3] = {{1, 2, 3}, {4, 5, 6}};
    const auto s = make_slice2d(v);
    sfc::expect_eq(s[0][0], 1);
    sfc::expect_eq(s[0][1], 2);
    sfc::expect_eq(s[0][2], 3);
    sfc::expect_eq(s[1][0], 4);
    sfc::expect_eq(s[1][1], 5);
    sfc::expect_eq(s[1][2], 6);
  }
}

SFC_TEST(slice_load) {
  // slice 1d
  {
    int v[] = {1, 2, 3, 4};
    const auto s = make_slice1d(v);

    for (auto i = 0; i < 4; ++i) {
      sfc::expect_eq(s.load(i), v[i]);
      sfc::expect_eq(s.load(i + 0.5f), v[i]);
    }
  }

  // slice 2d
  {
    int v[2][3] = {{1, 2, 3}, {4, 5, 6}};
    const auto s = make_slice2d(v);

    for (auto j = 0; j < 2; ++j) {
      for (auto i = 0; i < 3; ++i) {
        const auto t = v[j][i];
        sfc::expect_eq(s.load({float(i), float(j)}), t);
        sfc::expect_eq(s.load({float(i) + 0.5f, float(j) + 0.5f}), t);
      }
    }
  }
}

SFC_TEST(slice_load_interp_1d) {
  f32 v[] = {1, 2, 3};
  const auto s = make_slice1d(v);

  sfc::expect_eq(s.load_interp(-1), 0);

  sfc::expect_eq(s.load_interp(0.0f), 1.0f);
  sfc::expect_eq(s.load_interp(0.1f), 1.0f);
  sfc::expect_eq(s.load_interp(0.5f), 1.0f);

  sfc::expect_eq(s.load_interp(1), 1.5f);
  sfc::expect_eq(s.load_interp(1.5f), 2.0f);

  sfc::expect_eq(s.load_interp(2), 2.5f);
  sfc::expect_eq(s.load_interp(2.5f), 3.0f);
  sfc::expect_eq(s.load_interp(2.9f), 3.0f);

  sfc::expect_eq(s.load_interp(3.0f), 3.0f);
  sfc::expect_eq(s.load_interp(3.1f), 0.0f);
}

SFC_TEST(slice_load_interp_2d) {
  f32 v[2][2] = {{1, 2}, {3, 4}};
  const auto s = make_slice2d(v);

  sfc::expect_eq(s.load_interp_2d({-1, -1}), 0);

  // edge
  sfc::expect_eq(s.load_interp_2d({0, 0}), 1);
  sfc::expect_eq(s.load_interp_2d({1, 0}), 1.5f);
  sfc::expect_eq(s.load_interp_2d({2, 0}), 2.0f);

  // inner
  sfc::expect_eq(s.load_interp_2d({0.5f, 0.5f}), 1.0f);
  sfc::expect_eq(s.load_interp_2d({0.5f, 1.0f}), 2.0f);
  sfc::expect_eq(s.load_interp_2d({1.0f, 0.5f}), 1.5f);
  sfc::expect_eq(s.load_interp_2d({1.0f, 1.0f}), 2.5f);

  // edge
  sfc::expect_eq(s.load_interp_2d({2.0f, 2.0f}), 4.0f);
}

}  // namespace sfc::math::test
