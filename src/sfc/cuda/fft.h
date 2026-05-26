#pragma once

#include "sfc/core/mod.h"

namespace sfc::math {
template<class T, int N>
struct NdSlice;
}

namespace sfc::cuda {

using fft_plan_t = int;

template <class I, class O>
class FFT {
  u32 _len{0};
  u32 _batch{0};
  fft_plan_t _plan;

 public:
  FFT() noexcept;
  ~FFT();
  FFT(FFT&& other) noexcept;
  auto operator=(FFT&& other) noexcept -> FFT&;
  static auto create(u32 len, u32 batch = 1) -> FFT;

 public:
  void exec(const I in[], O out[], int DIR);

  template <int N>
  void operator()(math::NdSlice<I, N> in, math::NdSlice<O, N> out, int DIR = -1) {
    static_assert(N == 1 || N == 2);
    this->exec(in._data, out._data, DIR);
  }
};

}  // namespace sfc::cuda
