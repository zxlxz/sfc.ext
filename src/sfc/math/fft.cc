#include <fftw3.h>
#include "sfc/math/fft.h"

namespace sfc::math::fft {

using fftw_plan_t = fftwf_plan_s*;

template <class T>
static auto fftw_cast(T* p) {
  if constexpr (trait::same_<T, c32>) {
    return ptr::cast<fftwf_complex>(p);
  } else if constexpr (trait::same_<T, f32>) {
    return p;
  }
}

static void fftw_drop(fftw_plan_t p) {
  if (p == nullptr) return;
  ::fftwf_destroy_plan(p);
}

static auto fftw_plan(u32 N, const c32 in[], c32 out[], int SIGN) -> fftw_plan_t {
  const auto idata = fftw_cast(ptr::cast_mut(in));
  const auto odata = fftw_cast(out);
  return ::fftwf_plan_dft_1d(int(N), idata, odata, SIGN, FFTW_ESTIMATE);
}

static auto fftw_plan(u32 N, const f32 in[], c32 out[], [[maybe_unused]] int SIGN) -> fftw_plan_t {
  const auto idata = fftw_cast(ptr::cast_mut(in));
  const auto odata = fftw_cast(out);
  return ::fftwf_plan_dft_r2c_1d(int(N), idata, odata, FFTW_ESTIMATE);
}

static auto fftw_plan(u32 N, const c32 in[], f32 out[], [[maybe_unused]] int SIGN) -> fftw_plan_t {
  const auto idata = fftw_cast(ptr::cast_mut(in));
  const auto odata = fftw_cast(out);
  return ::fftwf_plan_dft_c2r_1d(int(N), idata, odata, FFTW_ESTIMATE);
}

static void fftw_exec(fftw_plan_t plan, const c32 in[], c32 out[]) {
  const auto idata = fftw_cast(ptr::cast_mut(in));
  const auto odata = fftw_cast(out);
  ::fftwf_execute_dft(plan, idata, odata);
}

static void fftw_exec(fftw_plan_t plan, const f32 in[], c32 out[]) {
  const auto idata = fftw_cast(ptr::cast_mut(in));
  const auto odata = fftw_cast(out);
  ::fftwf_execute_dft_r2c(plan, idata, odata);
}

static void fftw_exec(fftw_plan_t plan, const c32 in[], f32 out[]) {
  const auto idata = fftw_cast(ptr::cast_mut(in));
  const auto odata = fftw_cast(out);
  ::fftwf_execute_dft_c2r(plan, idata, odata);
}

FFT::FFT() noexcept {}

FFT::~FFT() {
  fftw_drop(_fwd_inplace);
  fftw_drop(_inv_inplace);
  fftw_drop(_fwd_outplace);
  fftw_drop(_inv_outplace);
}

FFT::FFT(FFT&& other) noexcept
    : _len{mem::take(other._len)}
    , _fwd_inplace{mem::take(other._fwd_inplace)}
    , _inv_inplace{mem::take(other._inv_inplace)}
    , _fwd_outplace{mem::take(other._fwd_outplace)}
    , _inv_outplace{mem::take(other._inv_outplace)} {}

FFT& FFT::operator=(FFT&& other) noexcept {
  if (this == &other) return *this;
  mem::swap(_len, other._len);
  mem::swap(_fwd_inplace, other._fwd_inplace);
  mem::swap(_inv_inplace, other._inv_inplace);
  mem::swap(_fwd_outplace, other._fwd_outplace);
  mem::swap(_inv_outplace, other._inv_outplace);

  return *this;
}

FFT FFT::new_(u32 len) {
  const auto fwd_inplace = fftw_plan(len, ptr::null<c32>(), ptr::null<c32>(), FFTW_FORWARD);
  const auto inv_inplace = fftw_plan(len, ptr::null<c32>(), ptr::null<c32>(), FFTW_BACKWARD);

  auto res = FFT{};
  res._len = len;
  res._fwd_inplace = fwd_inplace;
  res._inv_inplace = inv_inplace;
  return res;
}

auto FFT::len() const -> usize {
  return _len;
}

void FFT::fft(math::NdSlice<c32, 1> in, math::NdSlice<c32, 1> out) {
  sfc::assert_(in.is_contiguous(), "FFT::fft: in is not contiguous");
  sfc::assert_(out.is_contiguous(), "FFT::fft: out is not contiguous");
  sfc::assert_(in._shape[0] == _len, "FFT::fft: in.shape({}) not match len(={})", in._shape, _len);
  sfc::assert_(out._shape[0] == _len, "FFT::fft: out.shape({}) not match len(={})", out._shape, _len);

  auto& plan = in._data == out._data ? _fwd_inplace : _fwd_outplace;
  if (plan == nullptr) {
    plan = fftw_plan(_len, in._data, out._data, FFTW_FORWARD);
  }

  fftw_exec(plan, in._data, out._data);
}

void FFT::ifft(math::NdSlice<c32, 1> in, math::NdSlice<c32, 1> out) {
  sfc::assert_(in.is_contiguous(), "FFT::ifft: in is not contiguous");
  sfc::assert_(out.is_contiguous(), "FFT::ifft: out is not contiguous");
  sfc::assert_(in._shape[0] == _len, "FFT::ifft: in.shape({}) not match len(={})", in._shape, _len);
  sfc::assert_(out._shape[0] == _len, "FFT::ifft: out.shape({}) not match len(={})", out._shape, _len);

  auto& plan = in._data == out._data ? _inv_inplace : _inv_outplace;
  if (plan == nullptr) {
    plan = fftw_plan(_len, in._data, out._data, FFTW_BACKWARD);
  }
  fftw_exec(plan, in._data, out._data);
}

void FFT::fft(math::NdSlice<c32, 2> in, math::NdSlice<c32, 2> out) {
  const auto [ibatch, ilen] = in._shape;
  const auto [obatch, olen] = out._shape;

  sfc::assert_(in.is_contiguous(), "FFT::fft: in is not contiguous");
  sfc::assert_(out.is_contiguous(), "FFT::fft: out is not contiguous");
  sfc::assert_(ilen == _len, "FFT::fft: in.shape({}) not match len(={})", in._shape, _len);
  sfc::assert_(olen == _len, "FFT::fft: out.shape({}) not match len(={})", out._shape, _len);
  sfc::assert_(ibatch == obatch, "FFT::fft: in.shape({}) not match out.shape({})", in._shape, out._shape);

  auto& plan = in._data == out._data ? _fwd_inplace : _fwd_outplace;
  if (plan == nullptr) {
    plan = fftw_plan(_len, in[0]._data, out[0]._data, FFTW_FORWARD);
  }

  for (auto i = 0U; i < ibatch; ++i) {
    auto tmp_in = in[i];
    auto tmp_out = out[i];
    fftw_exec(plan, tmp_in._data, tmp_out._data);
  }
}

void FFT::ifft(math::NdSlice<c32, 2> in, math::NdSlice<c32, 2> out) {
  const auto [ibatch, ilen] = in._shape;
  const auto [obatch, olen] = out._shape;

  sfc::assert_(in.is_contiguous(), "FFT::ifft: in is not contiguous");
  sfc::assert_(out.is_contiguous(), "FFT::ifft: out is not contiguous");
  sfc::assert_(ilen == _len, "FFT::ifft: in.shape({}) not match len(={})", in._shape, _len);
  sfc::assert_(olen == _len, "FFT::ifft: out.shape({}) not match len(={})", out._shape, _len);
  sfc::assert_(ibatch == obatch, "FFT::ifft: in.shape({}) not match out.shape({})", in._shape, out._shape);

  auto& plan = in._data == out._data ? _inv_inplace : _inv_outplace;
  if (plan == nullptr) {
    plan = fftw_plan(_len, in[0]._data, out[0]._data, FFTW_BACKWARD);
  }

  for (auto i = 0U; i < ibatch; ++i) {
    auto tmp_in = in[i];
    auto tmp_out = out[i];
    fftw_exec(plan, tmp_in._data, tmp_out._data);
  }
}

RFFT::RFFT() noexcept {}

RFFT::~RFFT() {
  fftw_drop(_r2c);
  fftw_drop(_c2r);
}

RFFT::RFFT(RFFT&& other) noexcept
    : _len{mem::take(other._len)}, _r2c{mem::take(other._r2c)}, _c2r{mem::take(other._c2r)} {}

RFFT& RFFT::operator=(RFFT&& other) noexcept {
  if (this == &other) return *this;
  mem::swap(_len, other._len);
  mem::swap(_r2c, other._r2c);
  mem::swap(_c2r, other._c2r);
  return *this;
}

auto RFFT::new_(u32 len) -> RFFT {
  const auto r2c = fftw_plan(len, ptr::null<f32>(), ptr::null<c32>(), FFTW_FORWARD);
  const auto c2r = fftw_plan(len, ptr::null<c32>(), ptr::null<f32>(), FFTW_BACKWARD);

  auto res = RFFT{};
  res._len = len;
  res._r2c = r2c;
  res._c2r = c2r;
  return res;
}

void RFFT::fft(math::NdSlice<f32, 1> in, math::NdSlice<c32, 1> out) {
  const auto full_len = _len;
  const auto half_len = _len / 2 + 1;
  sfc::assert_(in.is_contiguous(), "RFFT::fft: in is not contiguous");
  sfc::assert_(out.is_contiguous(), "RFFT::fft: out is not contiguous");
  sfc::assert_(in._shape[0] == full_len, "RFFT::fft: in.shape({}) not match fft.len(={})", in._shape, _len);
  sfc::assert_(out._shape[0] == half_len, "RFFT::fft: out.shape({}) not match fft.len(={})/2+1", out._shape, _len);

  fftw_exec(_r2c, in._data, out._data);
}

void RFFT::ifft(math::NdSlice<c32, 1> in, math::NdSlice<f32, 1> out) {
  const auto full_len = _len;
  const auto half_len = _len / 2 + 1;
  sfc::assert_(in.is_contiguous(), "RFFT::ifft: in is not contiguous");
  sfc::assert_(out.is_contiguous(), "RFFT::ifft: out is not contiguous");
  sfc::assert_(in._shape[0] == half_len, "RFFT::ifft: in.shape({}) not match fft.len(={})/2+1", in._shape, _len);
  sfc::assert_(out._shape[0] == full_len, "RFFT::ifft: out.shape({}) not match fft.len(={})", out._shape, _len);

  fftw_exec(_c2r, in._data, out._data);
}

void RFFT::fft(math::NdSlice<f32, 2> in, math::NdSlice<c32, 2> out) {
  const auto full_len = _len;
  const auto half_len = _len / 2 + 1;
  const auto [ibatch, ilen] = in._shape;
  const auto [obatch, olen] = out._shape;

  sfc::assert_(in.is_contiguous(), "RFFT::fft: in is not contiguous");
  sfc::assert_(out.is_contiguous(), "RFFT::fft: out is not contiguous");
  sfc::assert_(ilen == full_len, "RFFT::fft: in.shape({}) not match fft.len(={})", in._shape, _len);
  sfc::assert_(olen == half_len, "RFFT::fft: out.shape({}) not match fft.len(={})/2+1", out._shape, _len);
  sfc::assert_(ibatch == obatch, "RFFT::fft: in.shape({}) not match out.shape({})", in._shape, out._shape);

  for (auto i = 0U; i < ibatch; ++i) {
    auto tmp_in = in[i];
    auto tmp_out = out[i];
    fftw_exec(_r2c, tmp_in._data, tmp_out._data);
  }
}

void RFFT::ifft(math::NdSlice<c32, 2> in, math::NdSlice<f32, 2> out) {
  const auto full_len = _len;
  const auto half_len = _len / 2 + 1;
  const auto [ibatch, ilen] = in._shape;
  const auto [obatch, olen] = out._shape;

  sfc::assert_(in.is_contiguous(), "RFFT::ifft: in is not contiguous");
  sfc::assert_(out.is_contiguous(), "RFFT::ifft: out is not contiguous");
  sfc::assert_(ilen == half_len, "RFFT::ifft: in.shape({}) not match fft.len(={})/2+1", in._shape, _len);
  sfc::assert_(olen == full_len, "RFFT::ifft: out.shape({}) not match fft.len(={})", out._shape, _len);
  sfc::assert_(ibatch == obatch, "RFFT::ifft: in.shape({}) not match out.shape({})", in._shape, out._shape);

  for (auto i = 0U; i < ibatch; ++i) {
    auto tmp_in = in[i];
    auto tmp_out = out[i];
    fftw_exec(_c2r, tmp_in._data, tmp_out._data);
  }
}

}  // namespace sfc::math::fft
