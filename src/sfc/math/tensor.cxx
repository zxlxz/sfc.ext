#include "sfc/test.h"
#include "sfc/io.h"
#include "sfc/math/tensor.h"

namespace sfc::math::tensor::test {

SFC_TEST(tensor_1d_shape) {
  auto a = math::empty<int>({4U});

  sfc::assert_eq(a.numel(), 4U);
  sfc::assert_eq(a.shape()[0], 4U);
  sfc::assert_ne(a.as_ptr(), nullptr);
}

SFC_TEST(tensor_2d_shape) {
  auto a = math::empty<int>({3U, 4U});

  sfc::assert_eq(a.numel(), 12U);
  sfc::assert_eq(a.shape()[0], 3U);
  sfc::assert_eq(a.shape()[1], 4U);
}

SFC_TEST(tensor_get_set) {
  auto a = math::empty<int>({3U});

  a.set({0}, 10);
  a.set({1}, 20);
  a.set({2}, 30);

  sfc::assert_eq(a.get({0}), 10);
  sfc::assert_eq(a.get({1}), 20);
  sfc::assert_eq(a.get({2}), 30);

  a.set({1}, 99);
  sfc::assert_eq(a.get({1}), 99);
}

SFC_TEST(tensor_2d_get_set) {
  auto a = math::empty<int>({2U, 3U});

  a[{0, 0}] = 1;
  a[{1, 2}] = 7;

  sfc::assert_eq(a[{0, 0}], 1);
  sfc::assert_eq(a[{1, 2}], 7);
}

SFC_TEST(tensor_index_op) {
  auto a = math::empty<int>({2U, 2U});
  a[{0, 0}] = 1;
  a[{0, 1}] = 2;
  a[{1, 0}] = 3;
  a[{1, 1}] = 4;

  // operator[](u32) returns a row NdSlice
  sfc::assert_eq(a[0][0], 1);
  sfc::assert_eq(a[1][1], 4);
}

SFC_TEST(tensor_for_each) {
  auto a = math::empty<i32>({4U});
  a.for_each_mut([](i32& val, u32 i) { val = i32(i) * i32(i); });
  sfc::assert_eq(a.get({0}), 0);
  sfc::assert_eq(a.get({1}), 1);
  sfc::assert_eq(a.get({2}), 4);
  sfc::assert_eq(a.get({3}), 9);
}

SFC_TEST(tensor_move) {
  auto a = math::empty<i32>({3U});
  a.set({0}, 1);
  a.set({1}, 2);
  a.set({2}, 3);

  auto b = sfc::mem::move(a);
  sfc::assert_eq(b.numel(), 3U);
  sfc::assert_eq(b.get({0}), 1);
  sfc::assert_eq(b.get({1}), 2);
  sfc::assert_eq(b.get({2}), 3);
}

SFC_TEST(tensor_fmt) {
  auto a = math::empty<i32>({2U, 3U});

  auto p = a.as_mut_ptr();
  for (auto i : ops::Range{a.numel()}) {
    p[i] = i32(i);
  }

  io::println("a.shape = {}", a.shape());
  io::println("a.strides = {}", a.strides());
  io::println("a = \n {:2d}\n", a);
}

}  // namespace sfc::math::tensor::test
