#pragma once

#include "sfc/math/mod.h"

namespace sfc::math {

template <class T, int N>
struct vec;

template <class T>
struct vec<T, 1> {
  static constexpr auto NDIM = 1U;
  T x;

 public:
  void fmt(auto& f) const {
    f.write_fmt("({-})", x);
  }
};

template <class T>
struct vec<T, 2> {
  static constexpr auto NDIM = 2U;
  T x, y;

 public:
  void fmt(auto& f) const {
    f.write_fmt("({-},{-})", x, y);
  }
};

template <class T>
struct vec<T, 3> {
  static constexpr auto NDIM = 3U;
  T x, y, z;

 public:
  void fmt(auto& f) const {
    f.write_fmt("({-},{-},{-})", x, y, z);
  }
};

template <class T>
struct vec<T, 4> {
  static constexpr auto NDIM = 4U;
  T x, y, z, w;

 public:
  void fmt(auto& f) const {
    f.write_fmt("({-},{-},{-},{-})", x, y, z, w);
  }
};

using vec1i = math::vec<i32, 1>;
using vec2i = math::vec<i32, 2>;
using vec3i = math::vec<i32, 3>;
using vec4i = math::vec<i32, 4>;

using vec1u = math::vec<u32, 1>;
using vec2u = math::vec<u32, 2>;
using vec3u = math::vec<u32, 3>;
using vec4u = math::vec<u32, 4>;

using vec1f = math::vec<f32, 1>;
using vec2f = math::vec<f32, 2>;
using vec3f = math::vec<f32, 3>;
using vec4f = math::vec<f32, 4>;

template <class T, class F, int N>
__hd inline auto cast(vec<F, N> v) -> vec<T, N> {
  if constexpr (N == 1) return {(T)(v.x)};
  if constexpr (N == 2) return {(T)(v.x), (T)(v.y)};
  if constexpr (N == 3) return {(T)(v.x), (T)(v.y), (T)(v.z)};
  if constexpr (N == 4) return {(T)(v.x), (T)(v.y), (T)(v.z), (T)(v.w)};
}

template <class T, int N>
__hd inline auto make_vec(const T (&v)[N]) -> vec<T, N> {
  if constexpr (N == 1) return {v[0]};
  if constexpr (N == 2) return {v[0], v[1]};
  if constexpr (N == 3) return {v[0], v[1], v[2]};
  if constexpr (N == 4) return {v[0], v[1], v[2], v[3]};
}

template <class T, int N>
__hd inline auto operator+(vec<T, N> a, vec<T, N> b) -> vec<T, N> {
  if constexpr (N == 1) return {a.x + b.x};
  if constexpr (N == 2) return {a.x + b.x, a.y + b.y};
  if constexpr (N == 3) return {a.x + b.x, a.y + b.y, a.z + b.z};
  if constexpr (N == 4) return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}

template <class T, int N>
__hd inline auto operator-(vec<T, N> a, vec<T, N> b) -> vec<T, N> {
  if constexpr (N == 1) return {a.x - b.x};
  if constexpr (N == 2) return {a.x - b.x, a.y - b.y};
  if constexpr (N == 3) return {a.x - b.x, a.y - b.y, a.z - b.z};
  if constexpr (N == 4) return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
}

template <class T, int N>
__hd inline auto operator*(vec<T, N> a, vec<T, N> b) -> vec<T, N> {
  if constexpr (N == 1) return {a.x * b.x};
  if constexpr (N == 2) return {a.x * b.x, a.y * b.y};
  if constexpr (N == 3) return {a.x * b.x, a.y * b.y, a.z * b.z};
  if constexpr (N == 4) return {a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w};
}

template <class T, int N>
__hd inline auto operator/(vec<T, N> a, vec<T, N> b) -> vec<T, N> {
  if constexpr (N == 1) return {a.x / b.x};
  if constexpr (N == 2) return {a.x / b.x, a.y / b.y};
  if constexpr (N == 3) return {a.x / b.x, a.y / b.y, a.z / b.z};
  if constexpr (N == 4) return {a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w};
}

template <class T, int N>
__hd inline auto operator*(T s, const vec<T, N>& v) -> vec<T, N> {
  if constexpr (N == 1) return {s * v.x};
  if constexpr (N == 2) return {s * v.x, s * v.y};
  if constexpr (N == 3) return {s * v.x, s * v.y, s * v.z};
  if constexpr (N == 4) return {s * v.x, s * v.y, s * v.z, s * v.w};
}

template <class T, int N>
__hd inline auto operator/(T s, const vec<T, N>& v) -> vec<T, N> {
  if constexpr (N == 1) return {s / v.x};
  if constexpr (N == 2) return {s / v.x, s / v.y};
  if constexpr (N == 3) return {s / v.x, s / v.y, s / v.z};
  if constexpr (N == 4) return {s / v.x, s / v.y, s / v.z, s / v.w};
}

template <int N>
__hd inline auto len(vec<f32, N> a) -> float {
  if constexpr (N == 1) return math::fabsf(a.x);
  if constexpr (N == 2) return math::sqrtf(a.x * a.x + a.y * a.y);
  if constexpr (N == 3) return math::sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
  if constexpr (N == 4) return math::sqrtf(a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w);
}

struct Rot {
  f32 cos = 1.0f;
  f32 sin = 0.0f;

 public:
  static auto identity() -> Rot {
    return {1.0f, 0.0f};
  }

  static auto from_rad(f32 angle) -> Rot {
    return {math::cosf(angle), math::sinf(angle)};
  }

  static auto from_deg(f32 deg) -> Rot {
    const auto a = deg * (math::PI / 180.0f);
    return Rot::from_rad(a);
  }

  __hd auto operator-() const -> Rot {
    return Rot{cos, -sin};
  }

  __hd auto operator()(vec2f v) const -> vec2f {
    const auto x = cos * v.x - sin * v.y;
    const auto y = sin * v.x + cos * v.y;
    return vec2f{x, y};
  }
};

}  // namespace sfc::math
