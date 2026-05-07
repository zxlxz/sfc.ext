#pragma once

#include "sfc/cuda/mod.h"
#include "sfc/math/complex.h"
#include "sfc/math/ndslice.h"

namespace sfc::cuda {

using fft_plan_t = int;

using math::f32;
using math::c32;

auto fft_plan_c2c(u32 nx, u32 batch) -> fft_plan_t;
auto fft_plan_r2c(u32 nx, u32 batch) -> fft_plan_t;
auto fft_plan_c2r(u32 nx, u32 batch) -> fft_plan_t;

void fft_drop(fft_plan_t plan);

void fft_exec_c2c(fft_plan_t plan, const c32* in, c32* out, int direction);
void fft_exec_r2c(fft_plan_t plan, const f32* in, c32* out);
void fft_exec_c2r(fft_plan_t plan, const c32* in, f32* out);

template <class I = c32, class O = c32>
struct FFT {
  fft_plan_t _plan = -1;

 public:
  FFT();
  ~FFT();
  FFT(FFT&& other) noexcept;
  FFT& operator=(FFT&& other) noexcept;

 public:
  static auto create(u32 nx, u32 batch) -> FFT;
  void operator()(NdSlice<I, 2> in, NdSlice<O, 2> out, int dir = -1);
};

}  // namespace sfc::cuda
