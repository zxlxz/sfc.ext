#pragma once

#include "sfc/math/mod.h"

namespace sfc::math {

template <class T, u32 N>
struct vec;

template <class T>
struct vec<T, 1> {
  static constexpr auto NDIM = 1U;
  T x;

 public:
  vec() = default;
  __hd vec(T x) : x{x} {}
  __hd vec(const T (&v)[1]) : x{v[0]} {}

  template <class U>
  explicit __hd operator vec<U, 1>() const {
    return {static_cast<U>(x)};
  }

 public:
  void fmt(auto& f) const {
    f.debug_tuple("").field(x);
  }
};

template <class T>
struct vec<T, 2> {
  static constexpr auto NDIM = 2U;
  T x, y;

 public:
  vec() = default;
  __hd vec(T x, T y) : x{x}, y{y} {}
  __hd vec(const T (&v)[2]) : x{v[0]}, y{v[1]} {}

  template <class U>
  explicit __hd operator vec<U, 2>() const {
    return {static_cast<U>(x), static_cast<U>(y)};
  }

 public:
  void fmt(auto& f) const {
    f.debug_tuple("").field(x).field(y);
  }
};

template <class T>
struct vec<T, 3> {
  static constexpr auto NDIM = 3U;
  T x, y, z;

 public:
  vec() = default;
  __hd vec(T x, T y, T z) : x{x}, y{y}, z{z} {}
  __hd vec(const T (&v)[3]) : x{v[0]}, y{v[1]}, z{v[2]} {}

  template <class U>
  explicit __hd operator vec<U, 3>() const {
    return {static_cast<U>(x), static_cast<U>(y), static_cast<U>(z)};
  }

 public:
  void fmt(auto& f) const {
    f.debug_tuple("").field(x).field(y).field(z);
  }
};

template <class T>
struct vec<T, 4> {
  static constexpr auto NDIM = 4U;
  T x, y, z, w;

 public:
  vec() = default;
  __hd vec(T x, T y, T z, T w) : x{x}, y{y}, z{z}, w{w} {}
  __hd vec(const T (&v)[4]) : x{v[0]}, y{v[1]}, z{v[2]}, w{v[3]} {}

  template <class U>
  explicit __hd operator vec<U, 4>() const {
    return {static_cast<U>(x), static_cast<U>(y), static_cast<U>(z), static_cast<U>(w)};
  }

 public:
  void fmt(auto& f) const {
    f.debug_tuple("").field(x).field(y).field(z).field(w);
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

template <class T>
vec(T) -> vec<T, 1>;

template <class T>
vec(T, T) -> vec<T, 2>;

template <class T>
vec(T, T, T) -> vec<T, 3>;

template <class T>
vec(T, T, T, T) -> vec<T, 4>;

template <class T, u32 N>
vec(const T (&arr)[N]) -> vec<T, N>;

template <class T, u32 N>
__hd inline auto operator==(vec<T, N> a, vec<T, N> b) -> bool {
  if constexpr (N == 1) return a.x == b.x;
  if constexpr (N == 2) return a.x == b.x && a.y == b.y;
  if constexpr (N == 3) return a.x == b.x && a.y == b.y && a.z == b.z;
  if constexpr (N == 4) return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

template <class T, u32 N>
__hd inline auto operator+(vec<T, N> a) -> vec<T, N> {
  if constexpr (N == 1) return {a.x};
  if constexpr (N == 2) return {a.x, a.y};
  if constexpr (N == 3) return {a.x, a.y, a.z};
  if constexpr (N == 4) return {a.x, a.y, a.z, a.w};
}

template <class T, u32 N>
__hd inline auto operator-(vec<T, N> a) -> vec<T, N> {
  if constexpr (N == 1) return {-a.x};
  if constexpr (N == 2) return {-a.x, -a.y};
  if constexpr (N == 3) return {-a.x, -a.y, -a.z};
  if constexpr (N == 4) return {-a.x, -a.y, -a.z, -a.w};
}

template <class T, u32 N>
__hd inline auto operator+(vec<T, N> a, vec<T, N> b) -> vec<T, N> {
  if constexpr (N == 1) return {a.x + b.x};
  if constexpr (N == 2) return {a.x + b.x, a.y + b.y};
  if constexpr (N == 3) return {a.x + b.x, a.y + b.y, a.z + b.z};
  if constexpr (N == 4) return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}

template <class T, u32 N>
__hd inline auto operator-(vec<T, N> a, vec<T, N> b) -> vec<T, N> {
  if constexpr (N == 1) return {a.x - b.x};
  if constexpr (N == 2) return {a.x - b.x, a.y - b.y};
  if constexpr (N == 3) return {a.x - b.x, a.y - b.y, a.z - b.z};
  if constexpr (N == 4) return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
}

template <class T, u32 N>
__hd inline auto operator*(vec<T, N> a, vec<T, N> b) -> vec<T, N> {
  if constexpr (N == 1) return {a.x * b.x};
  if constexpr (N == 2) return {a.x * b.x, a.y * b.y};
  if constexpr (N == 3) return {a.x * b.x, a.y * b.y, a.z * b.z};
  if constexpr (N == 4) return {a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w};
}

template <class T, u32 N>
__hd inline auto operator/(vec<T, N> a, vec<T, N> b) -> vec<T, N> {
  if constexpr (N == 1) return {a.x / b.x};
  if constexpr (N == 2) return {a.x / b.x, a.y / b.y};
  if constexpr (N == 3) return {a.x / b.x, a.y / b.y, a.z / b.z};
  if constexpr (N == 4) return {a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w};
}

template <class T, u32 N>
__hd inline auto operator*(T s, vec<T, N> v) -> vec<T, N> {
  if constexpr (N == 1) return {s * v.x};
  if constexpr (N == 2) return {s * v.x, s * v.y};
  if constexpr (N == 3) return {s * v.x, s * v.y, s * v.z};
  if constexpr (N == 4) return {s * v.x, s * v.y, s * v.z, s * v.w};
}

template <class T, u32 N>
__hd inline auto operator/(vec<T, N> v, T k) -> vec<T, N> {
  if constexpr (N == 1) return {v.x / k};
  if constexpr (N == 2) return {v.x / k, v.y / k};
  if constexpr (N == 3) return {v.x / k, v.y / k, v.z / k};
  if constexpr (N == 4) return {v.x / k, v.y / k, v.z / k, v.w / k};
}

template <class T, u32 N>
__hd inline auto operator+=(vec<T, N>& self, vec<T, N> other) -> vec<T, N>& {
  self = self + other;
  return self;
}

template <class T, u32 N>
__hd inline auto operator-=(vec<T, N>& self, vec<T, N> other) -> vec<T, N>& {
  self = self - other;
  return self;
}

template <u32 N>
__hd inline auto length(vec<f32, N> a) -> float {
  if constexpr (N == 1) return math::fabsf(a.x);
  if constexpr (N == 2) return math::sqrtf(a.x * a.x + a.y * a.y);
  if constexpr (N == 3) return math::sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
  if constexpr (N == 4) return math::sqrtf(a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w);
}

template <class T, u32 N>
__hd inline auto reduce_add(vec<T, N> v) -> T {
  if constexpr (N == 1) return v.x;
  if constexpr (N == 2) return v.x + v.y;
  if constexpr (N == 3) return v.x + v.y + v.z;
  if constexpr (N == 4) return v.x + v.y + v.z + v.w;
}

template <class T, u32 N>
__hd inline auto reduce_mul(vec<T, N> v) -> T {
  if constexpr (N == 1) return v.x;
  if constexpr (N == 2) return v.x * v.y;
  if constexpr (N == 3) return v.x * v.y * v.z;
  if constexpr (N == 4) return v.x * v.y * v.z * v.w;
}

template <class T, u32 N>
__hd inline auto dot(vec<T, N> a, vec<T, N> b) -> T {
  if constexpr (N == 1) return a.x * b.x;
  if constexpr (N == 2) return a.x * b.x + a.y * b.y;
  if constexpr (N == 3) return a.x * b.x + a.y * b.y + a.z * b.z;
  if constexpr (N == 4) return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

template <class T>
__hd inline auto cross(vec<T, 3> a, vec<T, 3> b) -> vec<T, 3> {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

}  // namespace sfc::math
