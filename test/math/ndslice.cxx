#include "sfc/test.h"
#include "sfc/io.h"
#include "sfc/math/ndslice.h"

namespace sfc::math::test {

template <class T, int N0>
auto as_slice_1d(T (&v)[N0]) -> NdSlice<T, 1> {
  return {v, {N0}, {1}};
}

template <class T, int N0, int N1>
auto as_slice_2d(T (&v)[N0][N1]) -> NdSlice<T, 2> {
  return {v[0], {N0, N1}, {N1, 1}};
}

SFC_TEST(ndview_1) {
  int buf[] = {1, 2, 3, 4};

  const auto s = as_slice_1d(buf);
  sfc::assert_eq(s.len(), 4U);
  sfc::assert_eq(s._shape[0], 4U);
  sfc::assert_eq(s._strides[0], 1U);
  sfc::assert_eq(s.numel(), 4U);

  sfc::assert_eq(s[0], 1);
  sfc::assert_eq(s[3], 4);

  io::println("s = \n {}\n", s);
}

SFC_TEST(ndview_2) {
  int buf[4][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};

  const auto s = as_slice_2d(buf);
  sfc::assert_eq(s.len(), 4U);
  sfc::assert_eq(s._shape[0], 4U);
  sfc::assert_eq(s._shape[1], 3U);
  sfc::assert_eq(s._strides[0], 3U);
  sfc::assert_eq(s._strides[1], 1U);
  sfc::assert_eq(s.numel(), 12U);

  sfc::assert_eq(s[0][0], 1);
  sfc::assert_eq(s[0][2], 3);
  sfc::assert_eq(s[1][0], 4);
  sfc::assert_eq(s[1][2], 6);

  io::println("s = {:2d}\n", s);
}


SFC_TEST(ndview_3) {
  int buf[2][3][4] = {
      {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}},
      {{13, 14, 15, 16}, {17, 18, 19, 20}, {21, 22, 23, 24}},
  };

  const auto s = NdSlice<int, 3>{buf[0][0], {2, 3, 4}, {12, 4, 1}};
  sfc::assert_eq(s.len(), 2U);
  sfc::assert_eq(s._shape[0], 2U);
  sfc::assert_eq(s._shape[1], 3U);
  sfc::assert_eq(s._shape[2], 4U);

  io::println("s = \n{:2d}\n", s);
}

}  // namespace sfc::math::test
