#pragma once

#include "sfc/cuda/mod.h"
#include "sfc/math/complex.h"
#include "sfc/math/ndslice.h"

namespace sfc::cuda {

template <class I, class O>
class FFT {
  using plan_t = int;
  u32 _len{0};
  u32 _batch{0};
  plan_t _plan{-1};

 public:
  FFT() noexcept;
  ~FFT();
  FFT(FFT&& other) noexcept;
  auto operator=(FFT&& other) noexcept -> FFT&;
  static auto create(u32 len, u32 batch = 1) -> FFT;

 public:
  void exec(const I in[], O out[], int DIR = +1);

  void operator()(math::NdSlice<I, 1> in, math::NdSlice<O, 1> out, int DIR = +1);
  void operator()(math::NdSlice<I, 2> in, math::NdSlice<O, 2> out, int DIR = +1);
};

}  // namespace sfc::cuda
