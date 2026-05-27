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
    sfc::assert_eq(s.len(), 4U);
    sfc::assert_eq(s.shape(), vec1u{4});
    sfc::assert_eq(s.strides(), vec1u{1});
    sfc::assert_eq(s.numel(), 4U);
  }

  // slice 2d
  {
    int v[2][3] = {{1, 2, 3}, {4, 5, 6}};
    const auto s = make_slice2d(v);
    sfc::assert_eq(s.len(), 2U);
    sfc::assert_eq(s.shape(), vec2u{3, 2});
    sfc::assert_eq(s.strides(), vec2u{1, 3});
    sfc::assert_eq(s.numel(), 6U);
  }
}

SFC_TEST(slice_visit) {
  // slice 1d
  {
    int v[] = {1, 2, 3, 4};
    const auto s = make_slice1d(v);
    sfc::assert_eq(s[0], 1);
    sfc::assert_eq(s[1], 2);
    sfc::assert_eq(s[2], 3);
    sfc::assert_eq(s[3], 4);
  }

  // slice 2d
  {
    int v[2][3] = {{1, 2, 3}, {4, 5, 6}};
    const auto s = make_slice2d(v);
    sfc::assert_eq(s[0][0], 1);
    sfc::assert_eq(s[0][1], 2);
    sfc::assert_eq(s[0][2], 3);
    sfc::assert_eq(s[1][0], 4);
    sfc::assert_eq(s[1][1], 5);
    sfc::assert_eq(s[1][2], 6);
  }
}


}  // namespace sfc::math::test
