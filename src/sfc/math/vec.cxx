#include "sfc/test.h"
#include "sfc/io.h"
#include "sfc/math/vec.h"

namespace sfc::math::vecs::test {

SFC_TEST(vec_simple) {
  const auto v1 = vec1f{1.0f};
  sfc::assert_eq(v1.x, 1.0f);
  io::println("v1 = {:.2}", v1);

  const auto v2 = vec2f{1.0f, 2.0f};
  sfc::assert_eq(v2.x, 1.0f);
  sfc::assert_eq(v2.y, 2.0f);
  io::println("v2 = {:.2}", v2);

  const auto v3 = vec3f{1.0f, 2.0f, 3.0f};
  sfc::assert_eq(v3.x, 1.0f);
  sfc::assert_eq(v3.y, 2.0f);
  sfc::assert_eq(v3.z, 3.0f);
  io::println("v3 = {:.2}", v3);

  const auto v4 = vec4f{1.0f, 2.0f, 3.0f, 4.0f};
  sfc::assert_eq(v4.x, 1.0f);
  sfc::assert_eq(v4.y, 2.0f);
  sfc::assert_eq(v4.z, 3.0f);
  sfc::assert_eq(v4.w, 4.0f);
  io::println("v4 = {:.2}", v4);
}

SFC_TEST(vec_from_array) {
  const i32 v1[] = {1};
  sfc::assert_eq(vec{v1}, vec{1});

  const i32 v2[] = {1, 2};
  sfc::assert_eq(vec{v2}, vec{1, 2});

  const i32 v3[] = {1, 2, 3};
  sfc::assert_eq(vec{v3}, vec{1, 2, 3});

  const i32 v4[] = {1, 2, 3, 4};
  sfc::assert_eq(vec{v4}, vec{1, 2, 3, 4});
}

SFC_TEST(vec_cmp) {
  const auto a = vec{1, 2, 3};
  const auto b = vec{2, 1, 3};
  const auto c = vec{2, 3, 4};

  sfc::assert_eq(a == a, true);
  sfc::assert_eq(a != a, false);

  sfc::assert_eq(a == b, false);
  sfc::assert_eq(a != b, true);
}

SFC_TEST(vec_add_sub_mul_div) {
  const auto a = vec{10, 20};
  const auto b = vec{1, 2};

  const auto c = a + b;
  sfc::assert_eq(c, vec{11, 22});

  const auto d = a - b;
  sfc::assert_eq(d, vec{9, 18});

  const auto e = a * b;
  sfc::assert_eq(e, vec{10, 40});

  const auto f = a / b;
  sfc::assert_eq(f, vec{10, 10});
}

SFC_TEST(vec_scalar_mul_div) {
  const auto v = vec{10, 20};

  sfc::assert_eq(2 * v, vec{20, 40});
  sfc::assert_eq(20 / v, vec{2, 1});
}

SFC_TEST(vec_cast) {
  const auto v = vec{1.1, 2.2};

  sfc::assert_eq(cast<i32>(v), vec{1, 2});
}

SFC_TEST(vec_len) {
  const auto l1 = math::len(vec2f{3.0f, 4.0f});
  sfc::assert_eq(l1, 5.0f);

  const auto l2 = math::len(vec3f{0.0f, 0.0f, 5.0f});
  sfc::assert_eq(l2, 5.0f);
}

SFC_TEST(vec_reduce) {
  sfc::assert_eq(reduce_add(vec{1, 2, 3}), 6);
  sfc::assert_eq(reduce_mul(vec{1, 2, 3, 4}), 24);
}

SFC_TEST(vec_dot) {
  sfc::assert_eq(dot(vec{1, 2}, vec{3, 4}), 11);
  sfc::assert_eq(dot(vec{1, 2, 3}, vec{4, 5, 6}), 32);
}

}  // namespace sfc::math::vecs::test
