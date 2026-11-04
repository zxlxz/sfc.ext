#include "sfc/test.h"
#include "sfc/io.h"
#include "sfc/math/ndslice.h"

namespace sfc::math::test {

SFC_TEST(ndview_len) {
  int buf[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

  auto v1 = NdSlice<int, 1>::with_shape(buf, {12});
  sfc::assert_eq(v1.len(), 12U);

  auto v2 = NdSlice<int, 2>::with_shape(buf, {3, 4});
  sfc::assert_eq(v2.len(), 3U);

  auto v3 = NdSlice<int, 3>::with_shape(buf, {3, 2, 2});
  sfc::assert_eq(v3.len(), 3U);

  auto v4 = NdSlice<int, 4>::with_shape(buf, {3, 2, 2, 1});
  sfc::assert_eq(v4.len(), 3U);
}

SFC_TEST(ndview_numel) {
  int buf[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

  auto v1 = NdSlice<int, 1>::with_shape(buf, {12});
  sfc::assert_eq(v1.numel(), 12U);

  auto v2 = NdSlice<int, 2>::with_shape(buf, {3, 4});
  sfc::assert_eq(v2.numel(), 12U);

  auto v3 = NdSlice<int, 3>::with_shape(buf, {3, 2, 2});
  sfc::assert_eq(v3.numel(), 12U);

  auto v4 = NdSlice<int, 4>::with_shape(buf, {3, 2, 2, 1});
  sfc::assert_eq(v4.numel(), 12U);
}

SFC_TEST(ndview_contains) {
  int buf[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

  auto v1 = NdSlice<int, 1>::with_shape(buf, {12});
  sfc::assert_eq(v1.contains({5}), true);
  sfc::assert_eq(v1.contains({12}), false);

  auto v2 = NdSlice<int, 2>::with_shape(buf, {3, 4});
  sfc::assert_eq(v2.contains({1, 2}), true);
  sfc::assert_eq(v2.contains({1, 4}), false);

  auto v3 = NdSlice<int, 3>::with_shape(buf, {3, 2, 2});
  sfc::assert_eq(v3.contains({2, 1, 1}), true);
  sfc::assert_eq(v3.contains({3, 0, 0}), false);

  auto v4 = NdSlice<int, 4>::with_shape(buf, {3, 2, 2, 1});
  sfc::assert_eq(v4.contains({2, 1, 1, 0}), true);
  sfc::assert_eq(v4.contains({3, 0, 0, 0}), false);
}

SFC_TEST(ndview_visit) {
  int buf[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

  auto v1 = NdSlice<int, 1>::with_shape(buf, {12});
  sfc::assert_eq(v1[5], 5);

  auto v2 = NdSlice<int, 2>::with_shape(buf, {3, 4});
  sfc::assert_eq(v2.get({1, 2}), 6);

  auto v3 = NdSlice<int, 3>::with_shape(buf, {3, 2, 2});
  sfc::assert_eq(v3.get({2, 1, 1}), 11);

  auto v4 = NdSlice<int, 4>::with_shape(buf, {3, 2, 2, 1});
  sfc::assert_eq(v4.get({2, 1, 1, 0}), 11);
}

SFC_TEST(ndview_get_set) {
  int buf[12] = {};

  auto v1 = NdSlice<int, 1>::with_shape(buf, {12});
  v1.set({5}, 5);
  sfc::assert_eq(v1.get({5}), 5);

  auto v2 = NdSlice<int, 2>::with_shape(buf, {3, 4});
  v2.set({1, 2}, 6);
  sfc::assert_eq(v2.get({1, 2}), 6);

  auto v3 = NdSlice<int, 3>::with_shape(buf, {3, 2, 2});
  v3.set({2, 1, 1}, 11);
  sfc::assert_eq(v3.get({2, 1, 1}), 11);

  auto v4 = NdSlice<int, 4>::with_shape(buf, {3, 2, 2, 1});
  v4.set({2, 1, 1, 0}, 11);
  sfc::assert_eq(v4.get({2, 1, 1, 0}), 11);
}

SFC_TEST(ndview_for_each) {
  int buf[12] = {};

  auto v2 = NdSlice<int, 2>::with_shape(buf, {3, 4});
  v2.fill_with([](u32 i, u32 j) { return int(i * 4 + j); });
  v2.for_each([](auto& e, u32 i, u32 j) { sfc::assert_eq(e, int(i * 4 + j)); });

  auto v3 = NdSlice<int, 3>::with_shape(buf, {3, 2, 2});
  v3.fill_with([](u32 i, u32 j, u32 k) { return int(i * 4 + j * 2 + k); });
  v3.for_each([](auto& e, u32 i, u32 j, u32 k) { sfc::assert_eq(e, int(i * 4 + j * 2 + k)); });

  auto v4 = NdSlice<int, 4>::with_shape(buf, {3, 2, 2, 1});
  v4.fill_with([](u32 i, u32 j, u32 k, u32 l) { return int(i * 4 + j * 2 + k + l); });
  v4.for_each([](auto& e, u32 i, u32 j, u32 k, u32 w) { sfc::assert_eq(e, int(i * 4 + j * 2 + k + w)); });
}

SFC_TEST(ndview_permute) {
  int buf[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

  auto v2 = NdSlice<int, 2>::with_shape(buf, {3, 4});
  auto t2 = v2.transpose();
  sfc::assert_eq(t2._shape[0], 4U);
  sfc::assert_eq(t2.get({2, 1}), v2.get({1, 2}));

  auto v3 = NdSlice<int, 3>::with_shape(buf, {3, 2, 2});
  auto p3 = v3.permute<2, 0, 1>();
  sfc::assert_eq(p3.get({1, 2, 0}), v3.get({2, 0, 1}));

  auto v4 = NdSlice<int, 4>::with_shape(buf, {3, 2, 2, 1});
  auto p4 = v4.permute<1, 0, 3, 2>();
  sfc::assert_eq(p4.get({1, 2, 0, 1}), v4.get({2, 1, 1, 0}));
}

SFC_TEST(ndview_reshape) {
  int buf[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

  auto v1 = NdSlice<int, 1>::with_shape(buf, {12});

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
    auto v1 = NdSlice<int, 1>::with_shape(buf, {12});
    io::println("v1 = {}", v1);
  }

  {
    auto v2 = NdSlice<int, 2>::with_shape(buf, {3, 4});
    io::println("v2 = \n{#}", v2);
  }

  {
    auto v3 = NdSlice<int, 3>::with_shape(buf, {2, 2, 3});
    io::println("v3 = \n{#}", v3);
  }

  {
    auto v4 = NdSlice<int, 4>::with_shape(buf, {3, 2, 2, 1});
    io::println("v4 = \n{#}", v4);
  }
}

SFC_TEST(ndview_is_contiguous) {
  int buf[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

  {
    auto v1_dense = NdSlice<int, 1>{buf, {12}, {1}};
    auto v1_sparse = NdSlice<int, 1>{buf, {6}, {2}};
    sfc::assert_eq(v1_dense[1], 1);
    sfc::assert_eq(v1_sparse[1], 2);

    sfc::assert_eq(v1_dense.is_contiguous(), true);
    sfc::assert_eq(v1_sparse.is_contiguous(), false);
  }

  {
    auto v2_dense = NdSlice<int, 2>{buf, {3, 4}, {4, 1}};
    auto v2_sparse = NdSlice<int, 2>{buf, {3, 2}, {4, 2}};
    sfc::assert_eq(v2_dense.get({1, 2}), 6);
    sfc::assert_eq(v2_sparse.get({1, 1}), 6);
    sfc::assert_eq(v2_dense.is_contiguous(), true);
    sfc::assert_eq(v2_sparse.is_contiguous(), false);
  }

  {
    auto v3_dense = NdSlice<int, 3>{buf, {2, 3, 2}, {6, 2, 1}};
    auto v3_sparse = NdSlice<int, 3>{buf, {2, 2, 2}, {6, 2, 2}};
    sfc::assert_eq(v3_dense.get({1, 2, 1}), 11);
    sfc::assert_eq(v3_sparse.get({1, 1, 1}), 10);
    sfc::assert_eq(v3_dense.is_contiguous(), true);
    sfc::assert_eq(v3_sparse.is_contiguous(), false);
  }

  {
    auto v4_dense = NdSlice<int, 4>{buf, {3, 2, 2, 1}, {4, 2, 1, 1}};
    auto v4_sparse = NdSlice<int, 4>{buf, {3, 2, 2, 1}, {4, 2, 2, 1}};
    sfc::assert_eq(v4_dense.get({1, 1, 1, 0}), 7);
    sfc::assert_eq(v4_sparse.get({1, 1, 1, 0}), 8);
    sfc::assert_eq(v4_dense.is_contiguous(), true);
    sfc::assert_eq(v4_sparse.is_contiguous(), false);
  }
}

}  // namespace sfc::math::test
