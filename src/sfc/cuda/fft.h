#pragma once

#include "sfc/cuda/mod.h"
#include "sfc/math/complex.h"
#include "sfc/math/tensor.h"

namespace sfc::cuda {

class CFFT {
  u32 _len{0};
  u32 _batch{0};
  int _plan{0};

 public:
  CFFT() noexcept;
  ~CFFT();
  CFFT(CFFT&& other) noexcept;
  CFFT& operator=(CFFT&& other) noexcept;

  static auto new_(u32 len, u32 batch = 1) -> CFFT;

 public:
  auto len() const -> usize;
  auto batch() const -> usize;

  auto fft(math::NdView<c32, 1> in, math::NdView<c32, 1> out) -> Result<>;
  auto ifft(math::NdView<c32, 1> in, math::NdView<c32, 1> out) -> Result<>;

  auto fft(math::NdView<c32, 2> in, math::NdView<c32, 2> out) -> Result<>;
  auto ifft(math::NdView<c32, 2> in, math::NdView<c32, 2> out) -> Result<>;
};

class RFFT {
  u32 _len{0};
  u32 _batch{0};
  int _plan_r2c{0};
  int _plan_c2r{0};

 public:
  RFFT() noexcept;
  ~RFFT();
  RFFT(RFFT&& other) noexcept;
  RFFT& operator=(RFFT&& other) noexcept;

  static auto create(u32 len, u32 batch = 1) -> RFFT;

 public:
  auto fft(math::NdView<f32, 1> in, math::NdView<c32, 1> out) -> Result<>;
  auto ifft(math::NdView<c32, 1> in, math::NdView<f32, 1> out) -> Result<>;

  auto fft(math::NdView<f32, 2> in, math::NdView<c32, 2> out) -> Result<>;
  auto ifft(math::NdView<c32, 2> in, math::NdView<f32, 2> out) -> Result<>;
};

}  // namespace sfc::cuda
