#pragma once

#include "sfc/math/ndslice.h"

namespace sfc::math {

template <class V>
struct NearestSampler {
  using Self = NearestSampler;
  using Item = V::Item;

  V _view;

  __hd auto load_1d(f32 x) const -> Item {
    const auto nx = _view._shape[0];
    const auto ix = static_cast<i32>(x);
    if (ix < 0 || ix >= i32(nx)) return 0.0f;

    const auto val = _view.get(u32(ix));
    return val;
  }

  __hd auto load_2d(f32 y, f32 x) const -> Item {
    const auto nx = _view._shape[0];
    const auto ny = _view._shape[1];
    const auto ix = static_cast<i32>(y);
    const auto iy = static_cast<i32>(x);
    if (ix < 0 || ix >= i32(nx)) return 0.0f;
    if (iy < 0 || iy >= i32(ny)) return 0.0f;

    const auto val = _view.get({u32(ix), u32(iy)});
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

    if (x <= 0.5f) {
      return x >= 0 ? _view.get({0}) : 0.0f;
    }

    if (x >= f32(nx) - 0.5f) {
      return x <= f32(nx) ? _view.get({nx - 1}) : 0.0f;
    }

    const auto fx = x - 0.5f;
    const auto ix = u32(fx);
    const auto t0 = _view.get({ix});
    const auto t1 = _view.get({ix + 1});

    const auto px = fx - f32(ix);
    return (1.0f - px) * t0 + px * t1;
  }

  __hd auto load_2d(f32 x, f32 y) const -> f32 {
    const auto nx = _view._shape[0];
    const auto ny = _view._shape[1];
    if (x < 0 || x > f32(nx)) return 0.0f;
    if (y < 0 || y > f32(ny)) return 0.0f;

    const auto fx = x - 0.5f;
    const auto fy = y - 0.5f;
    const auto ix = i32(fx);
    const auto iy = i32(fy);
    const auto px = fx < f32(ix) ? 0 : fx - f32(ix);
    const auto py = fy < f32(iy) ? 0 : fy - f32(iy);

    const auto x0 = ix < 0 ? 0 : u32(ix);
    const auto x1 = ix + 1 >= i32(nx) ? nx - 1 : u32(ix + 1);
    const auto y0 = iy < 0 ? 0 : u32(iy);
    const auto y1 = iy + 1 >= i32(ny) ? ny - 1 : u32(iy + 1);

    const auto t00 = _view.get({x0, y0});
    const auto t01 = _view.get({x0, y1});
    const auto t10 = _view.get({x1, y0});
    const auto t11 = _view.get({x1, y1});
    const auto t0y = Self::interp(py, t00, t01);
    const auto t1y = Self::interp(py, t10, t11);
    const auto val = Self::interp(px, t0y, t1y);
    return val;
  }
};

}  // namespace sfc::math
