#pragma once

#include "sfc/math/complex.h"
#include "sfc/math/ndarray.h"

struct fftwf_plan_s;

namespace sfc::math::fft {

class FFT {
  using plan_t = struct fftwf_plan_s*;

  u32 _len{0};
  plan_t _fwd_inplace{nullptr};
  plan_t _inv_inplace{nullptr};
  plan_t _fwd_outplace{nullptr};
  plan_t _inv_outplace{nullptr};

 public:
  FFT() noexcept;
  ~FFT();
  FFT(FFT&& other) noexcept;
  FFT& operator=(FFT&& other) noexcept;

  static auto new_(u32 len) -> FFT;

 public:
  auto len() const -> usize;

  void fft(math::NdSlice<c32, 1> in, math::NdSlice<c32, 1> out);
  void ifft(math::NdSlice<c32, 1> in, math::NdSlice<c32, 1> out);

  void fft(math::NdSlice<c32, 2> in, math::NdSlice<c32, 2> out);
  void ifft(math::NdSlice<c32, 2> in, math::NdSlice<c32, 2> out);
};

class RFFT {
  using plan_t = struct fftwf_plan_s*;
  u32 _len{0};
  plan_t _r2c{nullptr};
  plan_t _c2r{nullptr};

 public:
  RFFT() noexcept;
  ~RFFT();
  RFFT(RFFT&& other) noexcept;
  RFFT& operator=(RFFT&& other) noexcept;

  static auto new_(u32 len) -> RFFT;

 public:
  void fft(math::NdSlice<f32, 1> in, math::NdSlice<c32, 1> out);
  void ifft(math::NdSlice<c32, 1> in, math::NdSlice<f32, 1> out);

  void fft(math::NdSlice<f32, 2> in, math::NdSlice<c32, 2> out);
  void ifft(math::NdSlice<c32, 2> in, math::NdSlice<f32, 2> out);
};

}  // namespace sfc::math::fft
