#pragma once

#include "sfc/math/ndslice.h"

namespace sfc::math {

template <class T, int N>
struct Sampler;

template <class T>
struct Sampler<T, 1> {
  NdView<T, 1> _inn;
  const f32 _max = f32(_inn._shape[0]);

 public:
  auto load_nearest(f32 x) const -> T {
    if (!(x >= 0 && x < _max)) return T{0};

    const auto ix = static_cast<u32>(x);
    return _inn._data[ix];
  }

  auto load_linear(f32 loc) const -> T {
    if (!(loc >= 0 && loc <= _max)) {
      return T{0};
    }

    if (loc <= 0.5f) {
      return _inn._data[0];
    }

    if (loc >= _max - 0.5f) {
      return _inn._data[_inn._shape[0] - 1];
    }

    const auto fx = loc - 0.5f;
    const auto ix = u32(fx);
    const auto t0 = _inn[ix];
    const auto t1 = _inn[ix + 1];

    const auto px = fx - f32(ix);
    return (1.0f - px) * t0 + px * t1;
  }
};

template <class T>
struct Sampler<T, 2> {
  NdView<T, 2> _inn;
  const f32 _max_x = f32(_inn._shape[0]);
  const f32 _max_y = f32(_inn._shape[1]);

 public:
  auto load_nearest(vec2f p) const -> T {
    if (p.x < 0 || p.x >= _max_x) {
      return T{0};
    }
    if (p.y < 0 || p.y >= _max_y) {
      return T{0};
    }

    const auto ix = static_cast<u32>(p.x);
    const auto iy = static_cast<u32>(p.y);
    const auto val = _inn[{ix, iy}];
    return val;
  }

  auto load_linear(vec2f loc) const -> T {
    if (!(loc.x >= 0 && loc.x <= _max_x)) {
      return T{0};
    }
    if (!(loc.y >= 0 && loc.y <= _max_y)) {
      return T{0};
    }

    if (loc.y <= 0.5f) {
      const auto row = _inn[0];
      return Sampler<T, 1>{row}.load_linear(loc.x);
    }

    if (loc.y >= _max_y - 0.5f) {
      const auto row = _inn[_inn._shape[1] - 1];
      return Sampler<T, 1>{row}.load_linear(loc.x);
    }

    const auto fy = loc.y - 0.5f;
    const auto iy = u32(fy);
    const auto s0 = _inn[iy + 0];
    const auto s1 = _inn[iy + 1];
    const auto t0 = Sampler<T, 1>{s0}.load_linear(loc.x);
    const auto t1 = Sampler<T, 1>{s1}.load_linear(loc.x);

    const auto py = fy - f32(iy);
    return (1.0f - py) * t0 + py * t1;
  }
};

template <class T, int N>
Sampler(NdView<T, N>) -> Sampler<T, N>;

}  // namespace sfc::math
