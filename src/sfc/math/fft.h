#pragma once

#include "sfc/math/complex.h"
#include "sfc/cuda/memory.h"

namespace sfc::math {

template <class I, class O>
class FFTPlan;

template <class I, class O>
class FFT {
  FFTPlan<I, O>* _inn;

 public:
  FFT();
  ~FFT();
  FFT(FFT&& other) noexcept;
  FFT& operator=(FFT&& other) noexcept;

  static auto create(u32 len, u32 batch, cuda::MemType mtype) -> FFT;

  void forward(const I in[], O out[]);
  void inverse(const I in[], O out[]);
  void operator()(const I in[], O out[]);
};

void fft_c2c(u32 N, const c32 X[], c32 Y[], int direction);
void fft_r2c(u32 N, const f32 X[], c32 Y[]);
void fft_c2r(u32 N, const c32 X[], f32 Y[]);

}  // namespace sfc::math
