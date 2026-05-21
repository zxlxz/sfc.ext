#pragma once

#include "sfc/math/complex.h"
#include "sfc/math/ndslice.h"

namespace sfc::cuda {

using fft_plan_t = int;
void fft_drop(fft_plan_t plan);

template <class I, class O>
auto fft_plan(u32 N, u32 batch) -> fft_plan_t;

template <class I, class O>
void fft_exec(fft_plan_t plan, I in[], O out[], int SIGN);

struct FFTError {
  int _code;

 public:
  auto to_str() const -> cstr_t;

  void fmt(auto& f) const {
    const auto s = this->to_str();
    f.write_str(s);
  }
};

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
  void check_size(const u32 (&idim)[2], const u32 (&odim)[2]) const;
  void exec(const I in[], O out[], int DIR = +1);

  void operator()(math::NdSlice<I, 1> in, math::NdSlice<O, 1> out, int DIR = +1);
  void operator()(math::NdSlice<I, 2> in, math::NdSlice<O, 2> out, int DIR = +1);
};

}  // namespace sfc::cuda
