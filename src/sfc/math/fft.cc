#include <fftw3.h>

#include "sfc/core.h"
#include "sfc/math/fft.h"

namespace sfc::math {

FFT<f32, c32>::FFT() {}

FFT<f32, c32>::~FFT() {
  if (_plan == nullptr) return;
  ::fftwf_destroy_plan(_plan);
}

FFT<f32, c32>::FFT(FFT&& other) noexcept
    : _plan{other._plan}, _len{other._len}, _batch{other._batch} {
  other._plan = nullptr;
  other._len = 0;
  other._batch = 0;
}

FFT<f32, c32>& FFT<f32, c32>::operator=(FFT&& other) noexcept {
  if (this == &other) return *this;
  mem::swap(_plan, other._plan);
  mem::swap(_len, other._len);
  mem::swap(_batch, other._batch);
  return *this;
}

auto FFT<f32, c32>::create(u32 len, u32 batch) -> FFT {
  auto res = FFT{};
  res._len = len;
  res._batch = batch;
  res._plan = ::fftwf_plan_dft_r2c_1d(len, nullptr, nullptr, FFTW_ESTIMATE);
  return res;
}

void FFT<f32, c32>::exec(f32 X[], c32 Y[]) {
  auto I = X;
  auto O = reinterpret_cast<fftwf_complex*>(Y);
  for (auto k = 0U; k < _batch; ++k) {
    ::fftwf_execute_dft_r2c(_plan, I, O);
    I += _len;
    O += _len / 2 + 1;
  }
}

FFT<c32, f32>::FFT() {}

FFT<c32, f32>::~FFT() {
  if (_plan == nullptr) return;
  ::fftwf_destroy_plan(_plan);
}

FFT<c32, f32>::FFT(FFT&& other) noexcept
    : _plan{other._plan}, _len{other._len}, _batch{other._batch} {
  other._plan = nullptr;
  other._len = 0;
  other._batch = 0;
}

FFT<c32, f32>& FFT<c32, f32>::operator=(FFT&& other) noexcept {
  if (this == &other) return *this;
  mem::swap(_plan, other._plan);
  mem::swap(_len, other._len);
  mem::swap(_batch, other._batch);
  return *this;
}

auto FFT<c32, f32>::create(u32 len, u32 batch) -> FFT {
  auto res = FFT{};
  res._len = len;
  res._batch = batch;
  res._plan = ::fftwf_plan_dft_c2r_1d(len, nullptr, nullptr, FFTW_ESTIMATE);
  return res;
}

void FFT<c32, f32>::exec(c32 X[], f32 Y[]) {
  auto I = reinterpret_cast<fftwf_complex*>(X);
  auto O = Y;
  for (auto k = 0U; k < _batch; ++k) {
    ::fftwf_execute_dft_c2r(_plan, I, O);
    I += _len / 2 + 1;
    O += _len;
  }
}

FFT<c32, c32>::FFT() {}

FFT<c32, c32>::~FFT() {
  if (_inplace_fwd) ::fftwf_destroy_plan(_inplace_fwd);
  if (_inplace_inv) ::fftwf_destroy_plan(_inplace_inv);
  if (_outplace_fwd) ::fftwf_destroy_plan(_outplace_fwd);
  if (_outplace_inv) ::fftwf_destroy_plan(_outplace_inv);
}

FFT<c32, c32>::FFT(FFT&& other) noexcept
    : _len{mem::take(other._len)}
    , _batch{mem::take(other._batch)}
    , _inplace_fwd{mem::take(other._inplace_fwd)}
    , _inplace_inv{mem::take(other._inplace_inv)}
    , _outplace_fwd{mem::take(other._outplace_fwd)}
    , _outplace_inv{mem::take(other._outplace_inv)} {}

FFT<c32, c32>& FFT<c32, c32>::operator=(FFT&& other) noexcept {
  if (this == &other) return *this;
  mem::swap(_len, other._len);
  mem::swap(_batch, other._batch);
  mem::swap(_inplace_fwd, other._inplace_fwd);
  mem::swap(_inplace_inv, other._inplace_inv);
  mem::swap(_outplace_fwd, other._outplace_fwd);
  mem::swap(_outplace_inv, other._outplace_inv);
  return *this;
}

auto FFT<c32, c32>::create(u32 len, u32 batch) -> FFT {
  auto res = FFT{};
  res._len = len;
  res._batch = batch;
  return res;
}

void FFT<c32, c32>::exec(c32 X[], c32 Y[], int DIR) {
  auto I = reinterpret_cast<fftwf_complex*>(X);
  auto O = reinterpret_cast<fftwf_complex*>(Y);

  auto& p = X == Y ? (DIR == -1 ? _inplace_fwd : _inplace_inv)
                   : (DIR == -1 ? _outplace_fwd : _outplace_inv);
  if (p == nullptr) {
    p = ::fftwf_plan_dft_1d(_len, I, O, DIR, FFTW_ESTIMATE);
  }

  for (auto k = 0U; k < _batch; ++k) {
    ::fftwf_execute_dft(p, I, O);
    I += _len;
    O += _len;
  }
}

void fft(math::NdSlice<c32, 1> in, math::NdSlice<c32, 1> out) {
  panic::expect(in.len() == out.len(),
                "input len(=`{}`) and output.len(=`{}`) must equal",
                in.len(),
                out.len());

  auto plan = FFT<c32, c32>::create(in.len(), 1);
  plan.exec(in._data, out._data, -1);
}

void ifft(math::NdSlice<c32, 1> in, math::NdSlice<c32, 1> out) {
  panic::expect(in.len() == out.len(),
                "input len(=`{}`) and output.len(=`{}`) must equal",
                in.len(),
                out.len());

  auto plan = FFT<c32, c32>::create(in.len(), 1);
  plan.exec(in._data, out._data, +1);
}

}  // namespace sfc::math
