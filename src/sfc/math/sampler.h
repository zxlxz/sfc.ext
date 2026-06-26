#pragma once

#include "sfc/math/ndslice.h"

namespace sfc::math {

struct NearestSampler {
  using Self = NearestSampler;

  static __hd auto load_1d(const auto& row, vec1f p) -> f32 {
    static_assert(row.NDIM == 1);

    const auto nx = row._shape[0];
    if (p.x < 0 || p.x >= f32(nx)) return 0.0f;

    const auto ix = static_cast<u32>(p.x);
    const auto val = row.get(ix);
    return val;
  }

  static __hd auto load_2d(const auto& mat, vec2f p) -> f32 {
    static_assert(mat.NDIM == 2);

    const auto nx = mat._shape[0];
    const auto ny = mat._shape[1];
    if (p.x < 0 || p.x >= f32(nx)) return 0.0f;
    if (p.y < 0 || p.y >= f32(ny)) return 0.0f;

    const auto ix = static_cast<u32>(p.x);
    const auto iy = static_cast<u32>(p.y);
    const auto val = mat.get(ix, iy);
    return val;
  }
};

struct LinearSampler {
  using Self = LinearSampler;

  static __hd auto interp(f32 t, f32 v0, f32 v1) -> f32 {
    return (1.0f - t) * v0 + t * v1;
  }

  static __hd auto load_1d(const auto& row, vec1f p) -> f32 {
    static_assert(row.NDIM == 1);

    const auto nx = row._shape[0];
    if (p.x < 0 || p.x > f32(nx)) return 0.0f;

    if (p.x <= 0.5f) {
      return row.get(0);
    }

    if (p.x >= f32(nx) - 0.5f) {
      return row.get(nx - 1);
    }

    const auto fx = p.x - 0.5f;
    const auto ix = u32(fx);
    const auto t0 = row.get(ix);
    const auto t1 = row.get(ix + 1);

    const auto px = fx - f32(ix);
    return (1.0f - px) * t0 + px * t1;
  }

  static auto load_2d(const auto& mat, vec2f p) -> f32 {
    static_assert(mat.NDIM == 2);

    const auto nx = mat._shape[0];
    const auto ny = mat._shape[1];
    if (p.x < 0 || p.x > f32(nx)) return 0.0f;
    if (p.y < 0 || p.y > f32(ny)) return 0.0f;

    if (p.x <= 0.5f) {
      const auto row = mat[0];
      return Self::load_1d(row, {p.y});
    }

    if (p.x >= f32(nx) - 0.5f) {
      const auto row = mat[nx - 1];
      return Self::load_1d(row, {p.y});
    }

    const auto fx = p.x - 0.5f;
    const auto fy = p.y - 0.5f;
    const auto ix = u32(fx);
    const auto iy = u32(fy);
    const auto px = fx - f32(ix);
    const auto py = fy - f32(iy);

    const auto t00 = mat.get(ix + 0, iy + 0);
    const auto t01 = mat.get(ix + 0, iy + 1);
    const auto t10 = mat.get(ix + 1, iy + 0);
    const auto t11 = mat.get(ix + 1, iy + 1);

    const auto t0y = Self::interp(py, t00, t01);
    const auto t1y = Self::interp(py, t10, t11);
    const auto val = Self::interp(px, t0y, t1y);
    return val;
  }
};

}  // namespace sfc::math
