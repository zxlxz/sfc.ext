#pragma once

#include "sfc/math/complex.h"
#include "sfc/math/ndslice.h"

struct fftwf_plan_s;

namespace sfc::math {

template <class I, class O>
class FFT {
  using plan_t = ::fftwf_plan_s*;
  u32 _len{0};
  plan_t _fwd{nullptr};
  plan_t _inv{nullptr};

 public:
  FFT() noexcept;
  ~FFT();
  FFT(FFT&& other) noexcept;
  auto operator=(FFT&& other) noexcept -> FFT&;
  static auto create(u32 len) -> FFT;

 public:
  auto plan(const I in[], O out[], int DIR) -> plan_t;
  void exec(const I in[], O out[], int DIR = -1);

  template <int N>
  void operator()(math::NdSlice<I, N> in, math::NdSlice<O, N> out, int DIR = -1) {
    static_assert(N == 1 || N == 2);
    if constexpr (N == 1) {
      this->exec(in._data, out._data, DIR);
    } else if constexpr (N == 2) {
      for (auto i = 0U; i < in._dims.y; ++i) {
        this->exec(in[i]._data, out[i]._data, DIR);
      }
    }
  }
};

void fft(NdSlice<c32, 1> in, NdSlice<c32, 1> out);
void ifft(NdSlice<c32, 1> in, NdSlice<c32, 1> out);

void fft_r2c(NdSlice<f32, 1> in, NdSlice<c32, 1> out);
void fft_c2r(NdSlice<c32, 1> in, NdSlice<f32, 1> out);

}  // namespace sfc::math
