#pragma once

#include "sfc/math/vec.h"

namespace sfc::math {

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

  __hd auto operator()(const vec2f& v) const -> vec2f {
    const auto x = cos * v.x - sin * v.y;
    const auto y = sin * v.x + cos * v.y;
    return vec2f{x, y};
  }
};

}  // namespace sfc::math
