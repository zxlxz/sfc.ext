#pragma once

#include "sfc/cuda/memory.h"
#include "sfc/math/complex.h"
#include "sfc/math/ndslice.h"

namespace sfc::math {

template <class I, class O>
class FFT {
  struct Inn;
  u32 _len{0};
  u32 _batch{0};
  Inn* _inn;

 public:
  FFT() noexcept;
  ~FFT();
  FFT(FFT&& other) noexcept;
  auto operator=(FFT&& other) noexcept -> FFT&;
  static auto create(u32 len, u32 batch = 1) -> FFT;

 public:
  auto ilen() const -> u32;
  auto olen() const -> u32;
  void exec(const I in[], O out[], int DIR = +1);

  void operator()(math::NdSlice<I, 1> in, math::NdSlice<O, 1> out, int DIR = +1);
  void operator()(math::NdSlice<I, 2> in, math::NdSlice<O, 2> out, int DIR = +1);
};

void fft(NdSlice<c32, 1> in, NdSlice<c32, 1> out);
void ifft(NdSlice<c32, 1> in, NdSlice<c32, 1> out);

}  // namespace sfc::math
