#pragma once

#include "sfc/math/complex.h"
#include "sfc/math/ndarray.h"

namespace sfc::math {

template <class I, class O>
class FFTW {
  struct Inn;
  u32 _len{0};
  Inn* _inn{nullptr};

 public:
  FFTW() noexcept;
  ~FFTW();
  FFTW(FFTW&& other) noexcept;
  FFTW& operator=(FFTW&& other) noexcept;

  static auto create(u32 len) -> FFTW;

 public:
  auto ilen() const -> usize;
  auto olen() const -> usize;
  void exec(const I in[], O out[], int DIR = -1);

  void operator()(NdSlice<I, 1> in, NdSlice<O, 1> out, int DIR = -1);
  void operator()(NdSlice<I, 2> in, NdSlice<O, 2> out, int DIR = -1);
};

template <class I, class O>
auto fftw(u32 len) -> FFTW<I, O> {
  return FFTW<I, O>::create(len);
}

}  // namespace sfc::math
