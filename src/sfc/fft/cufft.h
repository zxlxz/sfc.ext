#pragma once

#include "sfc/math/complex.h"
#include "sfc/math/ndarray.h"

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
  auto in_len() const -> usize;
  auto out_len() const -> usize;
  void exec(const I in[], O out[], int DIR);

  void operator()(math::NdArray<I, 1>& in, math::NdArray<O, 1>& out, int DIR = -1);
  void operator()(math::NdArray<I, 2>& in, math::NdArray<O, 2>& out, int DIR = -1);
};

template <class I, class O>
auto cufft(u32 len, u32 batch) -> CUFFT<I, O> {
  return CUFFT<I, O>::create(len, batch);
}

}  // namespace sfc::fft
