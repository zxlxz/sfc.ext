#pragma once

#include "sfc/math/complex.h"
#include "sfc/math/ndarray.h"

namespace sfc::cuda {

enum class FFTError;

template <class T = Unit>
using FFTResult = result::Result<T, FFTError>;

auto to_str(FFTError err) -> const char*;

template <class I, class O>
class CUFFT {
  u32 _len{0};
  u32 _batch{0};
  int _plan{-1};

 public:
  CUFFT() noexcept;
  ~CUFFT();
  CUFFT(CUFFT&& other) noexcept;
  CUFFT& operator=(CUFFT&& other) noexcept;

  static auto create(u32 len, u32 batch = 1) -> CUFFT;

 public:
  auto ilen() const -> usize;
  auto olen() const -> usize;
  auto exec(const I in[], O out[], int DIR) -> FFTResult<>;

  auto operator()(math::NdSlice<I, 1> in, math::NdSlice<O, 1> out, int DIR = -1) -> FFTResult<>;
  auto operator()(math::NdSlice<I, 2> in, math::NdSlice<O, 2> out, int DIR = -1) -> FFTResult<>;
};

template<class I, class O>
auto cufft(u32 len, u32 batch = 1) -> CUFFT<I, O> {
  return CUFFT<I, O>::create(len, batch);
}

}  // namespace sfc::cuda
