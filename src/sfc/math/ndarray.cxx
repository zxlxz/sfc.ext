#include "sfc/test.h"
#include "sfc/io.h"
#include "sfc/math/ndarray.h"

namespace sfc::math::ndarray::test {

SFC_TEST(ndarray_1d_shape) {
  auto a = math::array<f32>({4U});

  sfc::assert_eq(a.numel(), 4U);
  sfc::assert_eq(a.shape()[0], 4U);
  sfc::assert_ne(a.data(), nullptr);
}

SFC_TEST(ndarray_2d_shape) {
  auto a = math::array<f32>({3U, 4U});

  sfc::assert_eq(a.numel(), 12U);
  sfc::assert_eq(a.shape()[0], 3U);
  sfc::assert_eq(a.shape()[1], 4U);
}

SFC_TEST(ndarray_get_set) {
  auto a = math::array<i32>({3U});

  a.set({0}, 10);
  a.set({1}, 20);
  a.set({2}, 30);

  sfc::assert_eq(a.get({0}), 10);
  sfc::assert_eq(a.get({1}), 20);
  sfc::assert_eq(a.get({2}), 30);

  a.set({1}, 99);
  sfc::assert_eq(a.get({1}), 99);
}

SFC_TEST(ndarray_2d_get_set) {
  auto a = math::array<i32>({2U, 3U});

  a[{0, 0}] = 1;
  a[{1, 2}] = 7;

  sfc::assert_eq(a[{0, 0}], 1);
  sfc::assert_eq(a[{1, 2}], 7);
}

SFC_TEST(ndarray_index_op) {
  auto a = math::array<i32>({2U, 2U});
  a[{0, 0}] = 1;
  a[{0, 1}] = 2;
  a[{1, 0}] = 3;
  a[{1, 1}] = 4;

  // operator[](u32) returns a row NdSlice
  sfc::assert_eq(a[0][0], 1);
  sfc::assert_eq(a[1][1], 4);
}

SFC_TEST(ndarray_bzero) {
  auto a = math::array<i32>({3});
  a.set({0}, 5);
  a.set({1}, 5);
  a.set({2}, 5);
  a.bzero();
  sfc::assert_eq(a.get({0}), 0);
  sfc::assert_eq(a.get({1}), 0);
  sfc::assert_eq(a.get({2}), 0);
}

SFC_TEST(ndarray_for_each) {
  auto a = math::array<i32>({4U});
  a.for_each([](u32 i, i32& val) { val = i32(i) * i32(i); });
  sfc::assert_eq(a.get({0}), 0);
  sfc::assert_eq(a.get({1}), 1);
  sfc::assert_eq(a.get({2}), 4);
  sfc::assert_eq(a.get({3}), 9);
}

SFC_TEST(ndarray_move) {
  auto a = math::array<i32>({3U});
  a.set({0}, 1);
  a.set({1}, 2);
  a.set({2}, 3);

  auto b = sfc::mem::move(a);
  sfc::assert_eq(b.numel(), 3U);
  sfc::assert_eq(b.get({0}), 1);
  sfc::assert_eq(b.get({1}), 2);
  sfc::assert_eq(b.get({2}), 3);
}

SFC_TEST(ndarray_mem_location) {
  const auto a = math::array<f32>({2U});
  const auto loc = a.mem_location();
  sfc::assert_eq(int(loc.kind), int(MemKind::CPU));
}

SFC_TEST(ndarray_fmt) {
  auto a = math::array<i32>({2U, 3U});
  a.for_each([](u32 i, u32 j, i32& val) { val = i32(i * 3 + j); });
  io::println("ndarray = {:2d}\n", a);
}

}  // namespace sfc::math::test
