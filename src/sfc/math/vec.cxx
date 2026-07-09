#include "sfc/test.h"
#include "sfc/io.h"
#include "sfc/math/vec.h"

namespace sfc::math::vecs::test {

SFC_TEST(vec_construct_1d) {
  const vec1f a{1.0f};
  sfc::assert_eq(a.x, 1.0f);

  const f32 arr[] = {3.0f};
  const vec1f b{arr};
  sfc::assert_eq(b.x, 3.0f);
}

SFC_TEST(vec_construct_2d) {
  const vec2f a{1.0f, 2.0f};
  sfc::assert_eq(a.x, 1.0f);
  sfc::assert_eq(a.y, 2.0f);

  const f32 arr[] = {5.0f, 6.0f};
  const vec2f b{arr};
  sfc::assert_eq(b.x, 5.0f);
  sfc::assert_eq(b.y, 6.0f);
}

SFC_TEST(vec_construct_3d) {
  const vec3f a{1.0f, 2.0f, 3.0f};
  sfc::assert_eq(a.x, 1.0f);
  sfc::assert_eq(a.y, 2.0f);
  sfc::assert_eq(a.z, 3.0f);

  const f32 arr[] = {7.0f, 8.0f, 9.0f};
  const vec3f b{arr};
  sfc::assert_eq(b.z, 9.0f);
}

SFC_TEST(vec_construct_4d) {
  const vec4f a{1.0f, 2.0f, 3.0f, 4.0f};
  sfc::assert_eq(a.w, 4.0f);

  const f32 arr[] = {1.0f, 2.0f, 3.0f, 4.0f};
  const vec4f b{arr};
  sfc::assert_eq(b.x, 1.0f);
  sfc::assert_eq(b.w, 4.0f);
}

SFC_TEST(vec_eq) {
  sfc::assert_eq(vec2f{1.0f, 2.0f} == vec2f{1.0f, 2.0f}, true);
  sfc::assert_eq(vec2f{1.0f, 2.0f} == vec2f{1.0f, 3.0f}, false);

  sfc::assert_eq(vec3i{1, 2, 3} == vec3i{1, 2, 3}, true);
  sfc::assert_eq(vec3i{1, 2, 3} == vec3i{1, 2, 4}, false);
}

SFC_TEST(vec_lt) {
  sfc::assert_eq(vec2f{1.0f, 2.0f} < vec2f{2.0f, 3.0f}, true);
  sfc::assert_eq(vec2f{1.0f, 2.0f} < vec2f{1.0f, 3.0f}, false);
  sfc::assert_eq(vec3i{1, 2, 3} < vec3i{2, 3, 4}, true);
}

SFC_TEST(vec_add) {
  const auto r = vec2i{1, 2} + vec2i{3, 4};
  sfc::assert_eq(r.x, 4);
  sfc::assert_eq(r.y, 6);

  const auto r3 = vec3i{1, 1, 1} + vec3i{2, 3, 4};
  sfc::assert_eq(r3.z, 5);
}

SFC_TEST(vec_sub) {
  const auto r = vec2i{5, 7} - vec2i{2, 3};
  sfc::assert_eq(r.x, 3);
  sfc::assert_eq(r.y, 4);

  const auto r4 = vec4i{10, 10, 10, 10} - vec4i{1, 2, 3, 4};
  sfc::assert_eq(r4.w, 6);
}

SFC_TEST(vec_mul_div) {
  const auto r = vec2i{2, 3} * vec2i{4, 5};
  sfc::assert_eq(r.x, 8);
  sfc::assert_eq(r.y, 15);

  const auto q = vec2i{8, 6} / vec2i{2, 3};
  sfc::assert_eq(q.x, 4);
  sfc::assert_eq(q.y, 2);
}

SFC_TEST(vec_scalar_mul_div) {
  const auto r = 2 * vec2i{1, 2};
  sfc::assert_eq(r.x, 2);
  sfc::assert_eq(r.y, 4);

  const auto q = 6 / vec2i{2, 3};
  sfc::assert_eq(q.x, 3);
  sfc::assert_eq(q.y, 2);
}

SFC_TEST(vec_cast) {
  const auto v = cast<i32>(vec2f{1.5f, 2.7f});
  sfc::assert_eq(v.x, 1);
  sfc::assert_eq(v.y, 2);
}

SFC_TEST(vec_len) {
  const auto l1 = math::len(vec2f{3.0f, 4.0f});
  sfc::assert_eq(l1, 5.0f);

  const auto l2 = math::len(vec3f{0.0f, 0.0f, 5.0f});
  sfc::assert_eq(l2, 5.0f);
}

SFC_TEST(vec_reduce_add) {
  sfc::assert_eq(reduce_add(vec3i{1, 2, 3}), 6);
  sfc::assert_eq(reduce_add(vec4i{1, 2, 3, 4}), 10);
}

SFC_TEST(vec_reduce_mul) {
  sfc::assert_eq(reduce_mul(vec3i{1, 2, 3}), 6);
  sfc::assert_eq(reduce_mul(vec4i{1, 2, 3, 4}), 24);
}

SFC_TEST(vec_dot) {
  sfc::assert_eq(dot(vec2i{1, 2}, vec2i{3, 4}), 11);
  sfc::assert_eq(dot(vec3i{1, 2, 3}, vec3i{4, 5, 6}), 32);
}

SFC_TEST(vec_fmt) {
  io::println("vec3f = {}\n", vec3f{1.0f, 2.0f, 3.0f});
}

}  // namespace sfc::math::test
