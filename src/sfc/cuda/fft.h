#pragma once

#include "sfc/cuda/mod.h"
#include "sfc/math/complex.h"

namespace sfc::math {
template <class T, int N>
struct NdSlice;
}

namespace sfc::cuda {

using fft_plan_t = int;

void fft_drop(fft_plan_t plan);

auto fft_plan_c2c(u32 nx, u32 batch) -> fft_plan_t;
auto fft_plan_r2c(u32 nx, u32 batch) -> fft_plan_t;
auto fft_plan_c2r(u32 nx, u32 batch) -> fft_plan_t;

void fft_exec_c2c(fft_plan_t plan, const c32* in, c32* out, int direction);
void fft_exec_r2c(fft_plan_t plan, const f32* in, c32* out);
void fft_exec_c2r(fft_plan_t plan, const c32* in, f32* out);

template <class I, class O>
class FFT {
  using plan_t = fft_plan_t;

  u32 _len{0};
  u32 _batch{0};
  plan_t _plan{-1};

 public:
  FFT();
  ~FFT();
  FFT(FFT&&) noexcept;
  FFT& operator=(FFT&&) noexcept;

  static auto create(u32 len, u32 batch) -> FFT;

 public:
  void exec(const I in[], O out[], int DIR = 0);

  void operator()(math::NdSlice<I, 2> in, math::NdSlice<O, 2> out) {
    this->exec(in._data, out._data);
  }
};

}  // namespace sfc::cuda
