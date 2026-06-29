#pragma once

#include "sfc/math/complex.h"
#include "sfc/math/ndarray.h"

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
  auto in_len() const -> usize;
  auto out_len() const -> usize;
  auto plan(const I in[], O out[], int DIR) -> plan_t;
  void exec(const I in[], O out[], int DIR = -1);

  void operator()(math::NdArray<I, 1>& in, math::NdArray<O, 1>& out, int DIR = -1);
  void operator()(math::NdArray<I, 2>& in, math::NdArray<O, 2>& out, int DIR = -1);
};

template <class I, class O>
auto fftw(u32 len) -> FFTW<I, O> {
  return FFTW<I, O>::create(len);
}

}  // namespace sfc::fft
