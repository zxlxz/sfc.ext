#include "sfc/test.h"
#include "sfc/io.h"
#include "sfc/math/ndview.h"

namespace sfc::math::test {

SFC_TEST(ndview_len) {
  int buf[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

  auto v1 = NdView<int, 1>::with_shape(buf, {12});
  sfc::assert_eq(v1.len(), 12U);

  auto v2 = NdView<int, 2>::with_shape(buf, {3, 4});
  sfc::assert_eq(v2.len(), 3U);

  auto v3 = NdView<int, 3>::with_shape(buf, {3, 2, 2});
  sfc::assert_eq(v3.len(), 3U);

  auto v4 = NdView<int, 4>::with_shape(buf, {3, 2, 2, 1});
  sfc::assert_eq(v4.len(), 3U);
}

SFC_TEST(ndview_numel) {
  int buf[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

  auto v1 = NdView<int, 1>::with_shape(buf, {12});
  sfc::assert_eq(v1.numel(), 12U);

  auto v2 = NdView<int, 2>::with_shape(buf, {3, 4});
  sfc::assert_eq(v2.numel(), 12U);

  auto v3 = NdView<int, 3>::with_shape(buf, {3, 2, 2});
  sfc::assert_eq(v3.numel(), 12U);

  auto v4 = NdView<int, 4>::with_shape(buf, {3, 2, 2, 1});
  sfc::assert_eq(v4.numel(), 12U);
}

SFC_TEST(ndview_contains) {
  int buf[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

  auto v1 = NdView<int, 1>::with_shape(buf, {12});
  sfc::assert_eq(v1.contains(5), true);
  sfc::assert_eq(v1.contains(12), false);

  auto v2 = NdView<int, 2>::with_shape(buf, {3, 4});
  sfc::assert_eq(v2.contains(1, 2), true);
  sfc::assert_eq(v2.contains(1, 4), false);

  auto v3 = NdView<int, 3>::with_shape(buf, {3, 2, 2});
  sfc::assert_eq(v3.contains(2, 1, 1), true);
  sfc::assert_eq(v3.contains(3, 0, 0), false);

  auto v4 = NdView<int, 4>::with_shape(buf, {3, 2, 2, 1});
  sfc::assert_eq(v4.contains(2, 1, 1, 0), true);
  sfc::assert_eq(v4.contains(3, 0, 0, 0), false);
}

SFC_TEST(ndview_visit) {
  int buf[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

  auto v1 = NdView<int, 1>::with_shape(buf, {12});
  sfc::assert_eq(v1[5], 5);

  auto v2 = NdView<int, 2>::with_shape(buf, {3, 4});
  sfc::assert_eq(v2[1, 2], 6);

  auto v3 = NdView<int, 3>::with_shape(buf, {3, 2, 2});
  sfc::assert_eq(v3[2, 1, 1], 11);

  auto v4 = NdView<int, 4>::with_shape(buf, {3, 2, 2, 1});
  sfc::assert_eq(v4[2, 1, 1, 0], 11);
}

SFC_TEST(ndview_reshape) {
  int buf[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

  auto v1 = NdView<int, 1>::with_shape(buf, {12});

  // reshape v2
  {
    auto v2_ok = reshape(v1, {3, 4});
    sfc::assert_eq(v2_ok.len(), 3U);
    sfc::assert_eq(v2_ok.numel(), 12U);

    auto v2_err = reshape(v1, {3, 5});
    sfc::assert_eq(v2_err.len(), 0U);
    sfc::assert_eq(v2_err.numel(), 0U);
  }

  // reshape v3
  {
    auto v3_ok = reshape(v1, {3, 2, 2});
    sfc::assert_eq(v3_ok.len(), 3U);
    sfc::assert_eq(v3_ok.numel(), 12U);

    auto v3_err = reshape(v1, {3, 2, 3});
    sfc::assert_eq(v3_err.len(), 0U);
    sfc::assert_eq(v3_err.numel(), 0U);
  }

  // reshape v4
  {
    auto v4_ok = reshape(v1, {3, 2, 2, 1});
    sfc::assert_eq(v4_ok.len(), 3U);
    sfc::assert_eq(v4_ok.numel(), 12U);

    auto v4_err = reshape(v1, {3, 2, 2, 2});
    sfc::assert_eq(v4_err.len(), 0U);
    sfc::assert_eq(v4_err.numel(), 0U);
  }
}

SFC_TEST(ndview_fmt) {
  int buf[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

  {
    auto v1 = NdView<int, 1>::with_shape(buf, {12});
    io::println("v1 = {}", v1);
  }

  {
    auto v2 = NdView<int, 2>::with_shape(buf, {3, 4});
    io::println("v2 = \n{#}", v2);
  }

  {
    auto v3 = NdView<int, 3>::with_shape(buf, {2, 2, 3});
    io::println("v3 = \n{#}", v3);
  }

  {
    auto v4 = NdView<int, 4>::with_shape(buf, {3, 2, 2, 1});
    io::println("v4 = \n{#}", v4);
  }
}

SFC_TEST(ndview_is_contiguous) {
  int buf[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

  {
    auto v1_dense = NdView<int, 1>{buf, {12}, {1}};
    auto v1_sparse = NdView<int, 1>{buf, {6}, {2}};
    sfc::assert_eq(v1_dense[1], 1);
    sfc::assert_eq(v1_sparse[1], 2);

    sfc::assert_eq(v1_dense.is_contiguous(), true);
    sfc::assert_eq(v1_sparse.is_contiguous(), false);
  }

  {
    auto v2_dense = NdView<int, 2>{buf, {3, 4}, {4, 1}};
    auto v2_sparse = NdView<int, 2>{buf, {3, 2}, {4, 2}};
    sfc::assert_eq(v2_dense[1, 2], 6);
    sfc::assert_eq(v2_sparse[1, 1], 6);
    sfc::assert_eq(v2_dense.is_contiguous(), true);
    sfc::assert_eq(v2_sparse.is_contiguous(), false);
  }

  {
    auto v3_dense = NdView<int, 3>{buf, {2, 3, 2}, {6, 2, 1}};
    auto v3_sparse = NdView<int, 3>{buf, {2, 2, 2}, {6, 2, 2}};
    sfc::assert_eq(v3_dense[1, 2, 1], 11);
    sfc::assert_eq(v3_sparse[1, 1, 1], 11);
    sfc::assert_eq(v3_dense.is_contiguous(), true);
    sfc::assert_eq(v3_sparse.is_contiguous(), false);
  }

  {
    auto v4_dense = NdView<int, 4>{buf, {3, 2, 2, 1}, {4, 2, 1, 1}};
    auto v4_sparse = NdView<int, 4>{buf, {3, 2, 2, 1}, {4, 2, 2, 1}};
    sfc::assert_eq(v4_dense[2, 1, 1, 0], 11);
    sfc::assert_eq(v4_sparse[2, 1, 1, 0], 11);
    sfc::assert_eq(v4_dense.is_contiguous(), true);
    sfc::assert_eq(v4_sparse.is_contiguous(), false);
  }
}

}  // namespace sfc::math::test
