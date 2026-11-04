#include "sfc/test.h"
#include "sfc/io.h"
#include "sfc/math/ndarray.h"

namespace sfc::math::ndarray::test {

SFC_TEST(tensor_1d_shape) {
  auto a = math::empty<int>({4U});
  sfc::assert_eq(a->_shape[0], 4U);
}

SFC_TEST(tensor_2d_shape) {
  auto a = math::empty<int>({3U, 4U});
  sfc::assert_eq(a->_shape[0], 3U);
  sfc::assert_eq(a->_shape[1], 4U);
}

SFC_TEST(tensor_get_set) {
  auto a = math::empty<int>({3U});

  a.set({0U}, 10);
  a.set({1U}, 20);
  a.set({2U}, 30);

  sfc::assert_eq(a.get({0U}), 10);
  sfc::assert_eq(a.get({1U}), 20);
  sfc::assert_eq(a.get({2U}), 30);

  a.set({1U}, 99);
  sfc::assert_eq(a.get({1U}), 99);
}

SFC_TEST(tensor_2d_get_set) {
  auto a = math::empty<int>({2U, 3U});

  a.set({0U, 0U}, 1);
  a.set({1U, 2U}, 7);

  sfc::assert_eq(a.get({0U, 0U}), 1);
  sfc::assert_eq(a.get({1U, 2U}), 7);
}

SFC_TEST(tensor_index_op) {
  auto a = math::empty<int>({2U, 2U});
  a.set({0U, 0U}, 1);
  a.set({0U, 1U}, 2);
  a.set({1U, 0U}, 3);
  a.set({1U, 1U}, 4);

  // operator[](u32) returns a row NdSlice
  sfc::assert_eq(a.get({0U, 0U}), 1);
  sfc::assert_eq(a.get({1U, 1U}), 4);
}

SFC_TEST(tensor_for_each) {
  auto a = math::empty<i32>({4U});
  a->fill_with([](u32 i) { return i32(i * i); });
  sfc::assert_eq(a.get({0U}), 0);
  sfc::assert_eq(a.get({1U}), 1);
  sfc::assert_eq(a.get({2U}), 4);
  sfc::assert_eq(a.get({3U}), 9);
}

SFC_TEST(tensor_move) {
  auto a = math::empty<i32>({3U});
  a.set({0U}, 1);
  a.set({1U}, 2);
  a.set({2U}, 3);

  auto b = sfc::mem::move(a);
  sfc::assert_eq(b->numel(), 3U);
  sfc::assert_eq(b.get({0U}), 1);
  sfc::assert_eq(b.get({1U}), 2);
  sfc::assert_eq(b.get({2U}), 3);
}

SFC_TEST(tensor_fmt) {
  auto a = math::empty<i32>({2U, 3U});

  auto p = a->_data;
  for (auto i : Range{a->numel()}) {
    p[i] = i32(i);
  }

  io::println("a.shape = {}", a->_shape);
  io::println("a.strides = {}", a->_strides);
  io::println("a = \n {:2d}\n", *a);
}

}  // namespace sfc::math::ndarray::test
