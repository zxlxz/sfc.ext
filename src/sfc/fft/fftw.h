#pragma once

#include "sfc/math/complex.h"
#include "sfc/math/ndslice.h"

struct fftwf_plan_s;

namespace sfc::fft {

template <class I, class O>
class FFTW {
  using plan_t = ::fftwf_plan_s*;
  u32 _len{0};
  plan_t _fwd{nullptr};
  plan_t _inv{nullptr};

 public:
  FFTW() noexcept;
  ~FFTW();
  FFTW(FFTW&& other) noexcept;
  auto operator=(FFTW&& other) noexcept -> FFTW&;
  static auto create(u32 len) -> FFTW;

 public:
  auto plan(const I in[], O out[], int DIR) -> plan_t;
  void exec(const I in[], O out[], int DIR = -1);

  template <int N>
  void operator()(math::NdView<I, N> in, math::NdView<O, N> out, int DIR = -1) {
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

}  // namespace sfc::fft
