#pragma once

#include "sfc/cuda/memory.h"
#include "sfc/math/complex.h"
#include "sfc/math/ndslice.h"

namespace sfc::math {

template <class I, class O>
struct FFTPlan {
  FFTPlan() {}
  virtual ~FFTPlan() {}

  virtual auto len() const -> u32 = 0;
  virtual auto batch() const -> u32 = 0;
  virtual void exec(const I X[], O Y[], int DIR) = 0;
};

template <class I, class O>
class FFT {
  using Inn = FFTPlan<I, O>;
  Inn* _inn;

 public:
  FFT();
  ~FFT();
  FFT(FFT&& other) noexcept;
  FFT& operator=(FFT&& other) noexcept;

  static auto create(u32 len, u32 batch, cuda::MemType mtype) -> FFT;

  void forward(const I in[], O out[]);
  void inverse(const I in[], O out[]);
  void operator()(math::NdSlice<I, 2> in, math::NdSlice<O, 2> out);
};

void fft_c2c(u32 N, const c32 X[], c32 Y[], int direction);
void fft_r2c(u32 N, const f32 X[], c32 Y[]);
void fft_c2r(u32 N, const c32 X[], f32 Y[]);

void fft(math::NdSlice<c32, 1> in, math::NdSlice<c32, 1> out);
void ifft(math::NdSlice<c32, 1> in, math::NdSlice<c32, 1> out);

}  // namespace sfc::math
