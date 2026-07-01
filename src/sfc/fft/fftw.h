#pragma once

#include "sfc/math/complex.h"
#include "sfc/math/ndarray.h"

namespace sfc::fft {

template <class I, class O>
class FFTW {
  struct Inn;
  u32 _len{0};
  Inn* _inn{nullptr};

 public:
  FFTW() noexcept;
  ~FFTW();
  FFTW(FFTW&& other) noexcept;
  auto operator=(FFTW&& other) noexcept -> FFTW&;
  static auto create(u32 len) -> FFTW;

 public:
  auto in_len() const -> usize;
  auto out_len() const -> usize;
  void exec(const I in[], O out[], int DIR = -1);

  void operator()(const math::NdArray<I, 1>& in, math::NdArray<O, 1>& out, int DIR = -1);
  void operator()(const math::NdArray<I, 2>& in, math::NdArray<O, 2>& out, int DIR = -1);
};

template <class I, class O>
auto fftw(u32 len) -> FFTW<I, O> {
  return FFTW<I, O>::create(len);
}

}  // namespace sfc::fft
