#include <fftw3.h>

#include "sfc/fft/fftw.h"

namespace sfc::fft {

namespace fftw {
using fft_plan_t = fftwf_plan_s*;

template <class T>
static auto fft_cast(T* p) {
  if constexpr (trait::same_<T, c32>) {
    return ptr::cast<fftwf_complex>(p);
  } else if constexpr (trait::same_<T, f32>) {
    return p;
  }
}

static void fft_drop(fft_plan_t p) {
  if (p == nullptr) return;
  ::fftwf_destroy_plan(p);
}

template <class I, class O>
static auto fft_plan(u32 N, const I in[], O out[], int SIGN) -> fft_plan_t {
  const auto len = num::saturating_cast<int>(N);
  const auto idata = fftw::fft_cast(ptr::cast_mut(in));
  const auto odata = fftw::fft_cast(out);

  if constexpr (sizeof(I) == sizeof(O)) {
    return ::fftwf_plan_dft_1d(len, idata, odata, SIGN, FFTW_ESTIMATE);
  } else if constexpr (sizeof(I) < sizeof(O)) {
    return ::fftwf_plan_dft_r2c_1d(len, idata, odata, FFTW_ESTIMATE);
  } else if constexpr (sizeof(I) > sizeof(O)) {
    return ::fftwf_plan_dft_c2r_1d(len, idata, odata, FFTW_ESTIMATE);
  } else {
    static_assert(false, "unsupported type combination");
  }
}

template <class I, class O>
static void fft_exec(fft_plan_t plan, const I in[], O out[]) {
  const auto idata = fftw::fft_cast(ptr::cast_mut(in));
  const auto odata = fftw::fft_cast(out);

  if constexpr (sizeof(I) == sizeof(O)) {
    ::fftwf_execute_dft(plan, idata, odata);
  } else if constexpr (sizeof(I) < sizeof(O)) {
    ::fftwf_execute_dft_r2c(plan, idata, odata);
  } else if constexpr (sizeof(I) > sizeof(O)) {
    ::fftwf_execute_dft_c2r(plan, idata, odata);
  } else {
    static_assert(false, "unsupported type combination");
  }
}

}  // namespace fftw

template <class I, class O>
FFTW<I, O>::FFTW() noexcept {}

template <class I, class O>
FFTW<I, O>::~FFTW() {
  fftw::fft_drop(_fwd);
  fftw::fft_drop(_inv);
}

template <class I, class O>
FFTW<I, O>::FFTW(FFTW&& other) noexcept : _len{other._len}, _fwd{mem::take(other._fwd)}, _inv{mem::take(other._inv)} {}

template <class I, class O>
FFTW<I, O>& FFTW<I, O>::operator=(FFTW&& other) noexcept {
  if (this == &other) return *this;
  mem::swap(_fwd, other._fwd);
  mem::swap(_inv, other._inv);
  return *this;
}

template <class I, class O>
auto FFTW<I, O>::create(u32 len) -> FFTW {
  auto res = FFTW{};
  res._len = len;

  if constexpr (sizeof(I) < sizeof(O)) {
    res._fwd = fftw::fft_plan<I, O>(len, nullptr, nullptr, 0);
  } else if constexpr (sizeof(I) > sizeof(O)) {
    res._inv = fftw::fft_plan<I, O>(len, nullptr, nullptr, 0);
  }
  return res;
}

template <class I, class O>
auto FFTW<I, O>::plan(const I in[], O out[], int DIR) -> plan_t {
  if constexpr (sizeof(I) < sizeof(O)) {
    return _fwd;
  } else if constexpr (sizeof(I) > sizeof(O)) {
    return _inv;
  } else {
    auto& plan = DIR < 0 ? _fwd : _inv;
    if (plan == nullptr) {
      plan = fftw::fft_plan(_len, in, out, DIR);
    }
    return plan;
  }
}

template <class I, class O>
void FFTW<I, O>::exec(const I in[], O out[], int DIR) {
  const auto plan = this->plan(in, out, DIR);
  fftw::fft_exec(plan, in, out);
}

template class FFTW<f32, c32>;
template class FFTW<c32, f32>;
template class FFTW<c32, c32>;

}  // namespace sfc::fft
