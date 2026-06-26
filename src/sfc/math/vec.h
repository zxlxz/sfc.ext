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
  __hd vec(T x) : x{x} {}
  __hd vec(const T(&v)[1]) : x{v[0]} {}

 public:
  void fmt(auto& f) const {
    f.debug_tuple().entry(x);
  }
};

template <class T>
struct vec<T, 2> {
  static constexpr auto NDIM = 2U;
  T x, y;

 public:
  __hd vec(T x, T y) : x{x}, y{y} {}
  __hd vec(const T(&v)[2]) : x{v[0]}, y{v[1]} {}

  void fmt(auto& f) const {
    f.debug_tuple().entry(x).entry(y);
  }
};

template <class T>
struct vec<T, 3> {
  static constexpr auto NDIM = 3U;
  T x, y, z;

 public:
  __hd vec(T x, T y, T z) : x{x}, y{y}, z{z} {}
  __hd vec(const T(&v)[3]) : x{v[0]}, y{v[1]}, z{v[2]} {}

  void fmt(auto& f) const {
    f.debug_tuple().entry(x).entry(y).entry(z);
  }
};

template <class T>
struct vec<T, 4> {
  static constexpr auto NDIM = 4U;
  T x, y, z, w;

 public:
  __hd vec(T x, T y, T z, T w) : x{x}, y{y}, z{z}, w{w} {}
  __hd vec(const T(&v)[4]) : x{v[0]}, y{v[1]}, z{v[2]}, w{v[3]} {}

  void fmt(auto& f) const {
    f.debug_tuple().entry(x).entry(y).entry(z).entry(w);
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

template <class T, int N>
__hd inline auto operator==(const vec<T, N>& a, const vec<T, N>& b) -> bool {
  if constexpr (N == 1) return a.x == b.x;
  if constexpr (N == 2) return a.x == b.x && a.y == b.y;
  if constexpr (N == 3) return a.x == b.x && a.y == b.y && a.z == b.z;
  if constexpr (N == 4) return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

template <class T, int N>
__hd inline auto operator<(const vec<T, N>& a, const vec<T, N>& b) -> bool {
  if constexpr (N == 1) return a.x < b.x;
  if constexpr (N == 2) return a.x < b.x && a.y < b.y;
  if constexpr (N == 3) return a.x < b.x && a.y < b.y && a.z < b.z;
  if constexpr (N == 4) return a.x < b.x && a.y < b.y && a.z < b.z && a.w < b.w;
}

template <class T, int N>
__hd inline auto operator+(const vec<T, N>& a, const vec<T, N>& b) -> vec<T, N> {
  if constexpr (N == 1) return {a.x + b.x};
  if constexpr (N == 2) return {a.x + b.x, a.y + b.y};
  if constexpr (N == 3) return {a.x + b.x, a.y + b.y, a.z + b.z};
  if constexpr (N == 4) return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}

template <class T, int N>
__hd inline auto operator-(const vec<T, N>& a, const vec<T, N>& b) -> vec<T, N> {
  if constexpr (N == 1) return {a.x - b.x};
  if constexpr (N == 2) return {a.x - b.x, a.y - b.y};
  if constexpr (N == 3) return {a.x - b.x, a.y - b.y, a.z - b.z};
  if constexpr (N == 4) return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
}

template <class T, int N>
__hd inline auto operator*(const vec<T, N>& a, const vec<T, N>& b) -> vec<T, N> {
  if constexpr (N == 1) return {a.x * b.x};
  if constexpr (N == 2) return {a.x * b.x, a.y * b.y};
  if constexpr (N == 3) return {a.x * b.x, a.y * b.y, a.z * b.z};
  if constexpr (N == 4) return {a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w};
}

template <class T, int N>
__hd inline auto operator/(const vec<T, N>& a, const vec<T, N>& b) -> vec<T, N> {
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

template <class T, class F, int N>
__hd inline auto cast(const vec<F, N>& v) -> vec<T, N> {
  if constexpr (N == 1) return {(T)(v.x)};
  if constexpr (N == 2) return {(T)(v.x), (T)(v.y)};
  if constexpr (N == 3) return {(T)(v.x), (T)(v.y), (T)(v.z)};
  if constexpr (N == 4) return {(T)(v.x), (T)(v.y), (T)(v.z), (T)(v.w)};
}

template <int N>
__hd inline auto len(const vec<f32, N>& a) -> float {
  if constexpr (N == 1) return math::fabsf(a.x);
  if constexpr (N == 2) return math::sqrtf(a.x * a.x + a.y * a.y);
  if constexpr (N == 3) return math::sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
  if constexpr (N == 4) return math::sqrtf(a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w);
}

template <class T, int N>
__hd inline auto reduce_add(const vec<T, N>& v) -> T {
  if constexpr (N == 1) return v.x;
  if constexpr (N == 2) return v.x + v.y;
  if constexpr (N == 3) return v.x + v.y + v.z;
  if constexpr (N == 4) return v.x + v.y + v.z + v.w;
}

template <class T, int N>
__hd inline auto reduce_mul(const vec<T, N>& v) -> T {
  if constexpr (N == 1) return v.x;
  if constexpr (N == 2) return v.x * v.y;
  if constexpr (N == 3) return v.x * v.y * v.z;
  if constexpr (N == 4) return v.x * v.y * v.z * v.w;
}

template <class T, int N>
__hd inline auto dot(const vec<T, N>& a, const vec<T, N>& b) -> T {
  if constexpr (N == 1) return a.x * b.x;
  if constexpr (N == 2) return a.x * b.x + a.y * b.y;
  if constexpr (N == 3) return a.x * b.x + a.y * b.y + a.z * b.z;
  if constexpr (N == 4) return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

}  // namespace sfc::math
