#include <fftw3.h>

#include "sfc/core.h"
#include "sfc/math/fft.h"

namespace sfc::math {

using fft_plan_t = fftwf_plan_s*;

template <class T>
auto fft_cast(T* p) -> T* {
  return p;
}

auto fft_cast(c32* p) -> fftwf_complex* {
  return ptr::cast_mut<fftwf_complex>(p);
}

static void fft_drop(fft_plan_t p) {
  if (p == nullptr) return;
  ::fftwf_destroy_plan(p);
}

template <class I, class O>
static auto fft_plan(u32 N, const I in[], O out[], int SIGN) -> fft_plan_t {
  const auto len = static_cast<int>(N);
  const auto idata = math::fft_cast(const_cast<I*>(in));
  const auto odata = math::fft_cast(out);

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
  const auto idata = math::fft_cast(const_cast<I*>(in));
  const auto odata = math::fft_cast(out);

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

template <class I, class O>
FFT<I, O>::FFT() noexcept {}

template <class I, class O>
FFT<I, O>::~FFT() {
  math::fft_drop(_fwd);
  math::fft_drop(_inv);
}

template <class I, class O>
FFT<I, O>::FFT(FFT&& other) noexcept : _len{other._len}, _fwd{mem::take(other._fwd)}, _inv{mem::take(other._inv)} {}

template <class I, class O>
FFT<I, O>& FFT<I, O>::operator=(FFT&& other) noexcept {
  if (this == &other) return *this;
  mem::swap(_fwd, other._fwd);
  mem::swap(_inv, other._inv);
  return *this;
}

template <class I, class O>
auto FFT<I, O>::create(u32 len) -> FFT {
  auto res = FFT{};
  res._len = len;

  if constexpr (sizeof(I) < sizeof(O)) {
    res._fwd = math::fft_plan<I, O>(len, nullptr, nullptr, 0);
  } else if constexpr (sizeof(I) > sizeof(O)) {
    res._inv = math::fft_plan<I, O>(len, nullptr, nullptr, 0);
  }
  return res;
}

template <class I, class O>
auto FFT<I, O>::plan(const I in[], O out[], int DIR) -> plan_t {
  if constexpr (sizeof(I) < sizeof(O)) {
    return _fwd;
  } else if constexpr (sizeof(I) > sizeof(O)) {
    return _inv;
  } else {
    auto& plan = DIR < 0 ? _fwd : _inv;
    if (plan == nullptr) {
      plan = math::fft_plan(_len, in, out, DIR);
    }
    return plan;
  }
}

template <class I, class O>
void FFT<I, O>::exec(const I in[], O out[], int DIR) {
  const auto plan = this->plan(in, out, DIR);
  math::fft_exec(plan, in, out);
}

template class FFT<f32, c32>;
template class FFT<c32, f32>;
template class FFT<c32, c32>;

void fft(NdSlice<c32, 1> in, NdSlice<c32, 1> out) {
  auto plan = FFT<c32, c32>::create(in._dims.x);
  plan(in, out, -1);
}

void ifft(NdSlice<c32, 1> in, NdSlice<c32, 1> out) {
  auto plan = FFT<c32, c32>::create(in._dims.x);
  plan(in, out, +1);
}

void fft_r2c(NdSlice<f32, 1> in, NdSlice<c32, 1> out) {
  auto plan = FFT<f32, c32>::create(in._dims.x);
  plan(in, out, -1);
}

void fft_c2r(NdSlice<c32, 1> in, NdSlice<f32, 1> out) {
  auto plan = FFT<c32, f32>::create(out._dims.x);
  plan(in, out, +1);
}

}  // namespace sfc::math
