#include "sfc/test.h"
#include "sfc/io.h"
#include "sfc/math/ndview.h"

namespace sfc::math::test {

SFC_TEST(ndview_1) {
  int buf[] = {1, 2, 3, 4};

  const auto s = NdView<int, 1>{buf};
  sfc::assert_eq(s.len(), 4U);
  sfc::assert_eq(s.shape()[0], 4U);
  sfc::assert_eq(s.strides()[0], 1U);
  sfc::assert_eq(s.numel(), 4U);

  sfc::assert_eq(s[0], 1);
  sfc::assert_eq(s[3], 4);

  io::println("s = {}\n", s);
}

SFC_TEST(ndview_2) {
  int buf[2][3] = {{1, 2, 3}, {4, 5, 6}};

  const auto s = NdView<int, 2>{buf};
  sfc::assert_eq(s.len(), 2U);
  sfc::assert_eq(s.shape()[0], 2U);
  sfc::assert_eq(s.shape()[1], 3U);
  sfc::assert_eq(s.strides()[0], 3U);
  sfc::assert_eq(s.strides()[1], 1U);
  sfc::assert_eq(s.numel(), 6U);

  sfc::assert_eq(s[0][0], 1);
  sfc::assert_eq(s[0][2], 3);
  sfc::assert_eq(s[1][0], 4);
  sfc::assert_eq(s[1][2], 6);

  io::println("s = {}\n", s);
}

}  // namespace sfc::math::test
