#pragma once

#include "sfc/math/complex.h"
#include "sfc/math/ndslice.h"

namespace sfc::fft {

template <class I, class O>
class CUFFT {
  using plan_t = int;

  u32 _len{0};
  u32 _batch{0};
  plan_t _plan;

 public:
  CUFFT() noexcept;
  ~CUFFT();
  CUFFT(CUFFT&& other) noexcept;
  auto operator=(CUFFT&& other) noexcept -> CUFFT&;
  static auto create(u32 len, u32 batch = 1) -> CUFFT;

 public:
  void exec(const I in[], O out[], int DIR);

  template <int N>
  void operator()(math::NdSlice<I, N> in, math::NdSlice<O, N> out, int DIR = -1) {
    static_assert(N == 1 || N == 2);
    this->exec(in._data, out._data, DIR);
  }
};

}  // namespace sfc::cufft
