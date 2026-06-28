#pragma once

#include "sfc/math/ndslice.h"

namespace sfc::math {

template <class V>
struct NearestSampler {
  V _view;
  using Self = NearestSampler;

  __hd auto load_1d(f32 x) const -> f32 {
    const auto nx = _view._shape[0];
    if (x < 0 || x >= f32(nx)) return 0.0f;

    const auto ix = static_cast<u32>(x);
    const auto val = _view.get(ix);
    return val;
  }

  __hd auto load_2d(f32 x, f32 y) const -> f32 {
    const auto nx = _view._shape[0];
    const auto ny = _view._shape[1];
    if (x < 0 || x >= f32(nx)) return 0.0f;
    if (y < 0 || y >= f32(ny)) return 0.0f;

    const auto ix = static_cast<u32>(x);
    const auto iy = static_cast<u32>(y);
    const auto val = _view.get(ix, iy);
    return val;
  }
};

template <class V>
struct LinearSampler {
  using Self = LinearSampler;
  V _view;

 public:
  static __hd auto interp(f32 t, f32 v0, f32 v1) -> f32 {
    return (1.0f - t) * v0 + t * v1;
  }

  __hd auto load_1d(f32 x) const -> f32 {
    const auto nx = _view._shape[0];
    if (x < 0 || x >= f32(nx)) return 0.0f;

    if (x <= 0.5f) {
      return _view.get(0);
    }

    if (x >= f32(nx) - 0.5f) {
      return _view.get(nx - 1);
    }

    const auto fx = x - 0.5f;
    const auto ix = u32(fx);
    const auto t0 = _view.get(ix);
    const auto t1 = _view.get(ix + 1);

    const auto px = fx - f32(ix);
    return (1.0f - px) * t0 + px * t1;
  }

  __hd auto load_2d(f32 x, f32 y) const -> f32 {
    const auto nx = _view._shape[0];
    const auto ny = _view._shape[1];
    if (x < 0 || x >= f32(nx)) return 0.0f;
    if (y < 0 || y >= f32(ny)) return 0.0f;

    const auto fx = x - 0.5f;
    const auto fy = y - 0.5f;
    const auto ix = u32(fx);
    const auto iy = u32(fy);
    const auto px = fx - f32(ix);
    const auto py = fy - f32(iy);

    if (x <= 0.5f || x >= f32(nx) - 0.5f) {
      const auto ix = x <= 0.5f ? 0 : nx - 1;
      const auto tx0 = _view.get(ix, iy + 0);
      const auto tx1 = _view.get(ix, iy + 1);
      return Self::interp(py, tx0, tx1);
    }

    if (y <= 0.5f || y >= f32(ny) - 0.5f) {
      const auto iy = y <= 0.5f ? 0 : ny - 1;
      const auto t0y = _view.get(ix + 0, iy);
      const auto t1y = _view.get(ix + 1, iy);
      return Self::interp(px, t0y, t1y);
    }

    const auto t00 = _view.get(ix + 0, iy + 0);
    const auto t01 = _view.get(ix + 0, iy + 1);
    const auto t10 = _view.get(ix + 1, iy + 0);
    const auto t11 = _view.get(ix + 1, iy + 1);
    const auto t0y = Self::interp(py, t00, t01);
    const auto t1y = Self::interp(py, t10, t11);
    const auto val = Self::interp(px, t0y, t1y);
    return val;
  }
};

}  // namespace sfc::math
