#include <fftw3.h>
#include "sfc/math/fft.h"
#include "sfc/ffi/library.h"

namespace sfc::math::fft {

using plan_t = fftwf_plan_s*;

class FFTW3F {
  const ffi::Library& _lib;
#define X(f) decltype(f)* _##f = _lib.get_func<decltype(f)*>(#f)
  X(fftwf_destroy_plan);
  X(fftwf_plan_dft_1d);
  X(fftwf_plan_dft_r2c_1d);
  X(fftwf_plan_dft_c2r_1d);
  X(fftwf_execute_dft);
  X(fftwf_execute_dft_r2c);
  X(fftwf_execute_dft_c2r);
#undef X

 public:
  FFTW3F(const ffi::Library& lib) noexcept : _lib{lib} {}
  ~FFTW3F() = default;

  static auto instance() -> FFTW3F& {
    static auto lib = ffi::Library::load("libfftw3f");
    static auto res = FFTW3F{lib};
    return res;
  }

 public:
  template <class T>
  static auto cast(T* p) {
    if constexpr (trait::same_<T, c32>) {
      return ptr::cast<fftwf_complex>(p);
    } else if constexpr (trait::same_<T, f32>) {
      return p;
    }
  }

  void drop(plan_t p) {
    if (p == nullptr) return;
    _fftwf_destroy_plan(p);
  }

  auto plan(int N, const auto in[], auto out[], int SIGN) -> plan_t {
    const auto idata = FFTW3F::cast(ptr::cast_mut(in));
    const auto odata = FFTW3F::cast(out);
    if constexpr (requires { _fftwf_plan_dft_1d(N, idata, odata, SIGN, 0); }) {
      return _fftwf_plan_dft_1d(N, idata, odata, SIGN, FFTW_ESTIMATE);
    } else if constexpr (requires { _fftwf_plan_dft_r2c_1d(N, idata, odata, 0); }) {
      return _fftwf_plan_dft_r2c_1d(N, idata, odata, FFTW_ESTIMATE);
    } else if constexpr (requires { _fftwf_plan_dft_c2r_1d(N, idata, odata, 0); }) {
      return _fftwf_plan_dft_c2r_1d(N, idata, odata, FFTW_ESTIMATE);
    }
    return nullptr;
  }

  void exec(plan_t plan, const auto in[], auto out[]) {
    auto idata = FFTW3F::cast(ptr::cast_mut(in));
    auto odata = FFTW3F::cast(out);
    if constexpr (requires { _fftwf_execute_dft(plan, idata, odata); }) {
      _fftwf_execute_dft(plan, idata, odata);
    } else if constexpr (requires { _fftwf_execute_dft_r2c(plan, idata, odata); }) {
      _fftwf_execute_dft_r2c(plan, idata, odata);
    } else if constexpr (requires { _fftwf_execute_dft_c2r(plan, idata, odata); }) {
      _fftwf_execute_dft_c2r(plan, idata, odata);
    }
  }
};

FFT::FFT() noexcept {}

FFT::~FFT() {
  auto& fftw = FFTW3F::instance();
  fftw.drop(_fwd_inplace);
  fftw.drop(_inv_inplace);
  fftw.drop(_fwd_outplace);
  fftw.drop(_inv_outplace);
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
  auto& fftw = FFTW3F::instance();

  const auto n = num::saturating_cast<int>(len);
  const auto fwd_inplace = fftw.plan(n, ptr::null<c32>(), ptr::null<c32>(), FFTW_FORWARD);
  const auto inv_inplace = fftw.plan(n, ptr::null<c32>(), ptr::null<c32>(), FFTW_BACKWARD);

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
  auto& fftw = FFTW3F::instance();
  sfc::assert_(in.is_contiguous(), "FFT::fft: in is not contiguous");
  sfc::assert_(out.is_contiguous(), "FFT::fft: out is not contiguous");
  sfc::assert_(in._shape[0] == _len, "FFT::fft: in.shape({}) not match len(={})", in._shape, _len);
  sfc::assert_(out._shape[0] == _len, "FFT::fft: out.shape({}) not match len(={})", out._shape, _len);

  auto& plan = in._data == out._data ? _fwd_inplace : _fwd_outplace;
  if (plan == nullptr) {
    plan = fftw.plan(int(_len), in._data, out._data, FFTW_FORWARD);
  }

  fftw.exec(plan, in._data, out._data);
}

void FFT::ifft(math::NdSlice<c32, 1> in, math::NdSlice<c32, 1> out) {
  auto& fftw = FFTW3F::instance();
  sfc::assert_(in.is_contiguous(), "FFT::ifft: in is not contiguous");
  sfc::assert_(out.is_contiguous(), "FFT::ifft: out is not contiguous");
  sfc::assert_(in._shape[0] == _len, "FFT::ifft: in.shape({}) not match len(={})", in._shape, _len);
  sfc::assert_(out._shape[0] == _len, "FFT::ifft: out.shape({}) not match len(={})", out._shape, _len);

  auto& plan = in._data == out._data ? _inv_inplace : _inv_outplace;
  if (plan == nullptr) {
    plan = fftw.plan(int(_len), in._data, out._data, FFTW_BACKWARD);
  }
  fftw.exec(plan, in._data, out._data);
}

void FFT::fft(math::NdSlice<c32, 2> in, math::NdSlice<c32, 2> out) {
  auto& fftw = FFTW3F::instance();
  const auto [ibatch, ilen] = in._shape;
  const auto [obatch, olen] = out._shape;

  sfc::assert_(in.is_contiguous(), "FFT::fft: in is not contiguous");
  sfc::assert_(out.is_contiguous(), "FFT::fft: out is not contiguous");
  sfc::assert_(ilen == _len, "FFT::fft: in.shape({}) not match len(={})", in._shape, _len);
  sfc::assert_(olen == _len, "FFT::fft: out.shape({}) not match len(={})", out._shape, _len);
  sfc::assert_(ibatch == obatch, "FFT::fft: in.shape({}) not match out.shape({})", in._shape, out._shape);

  auto& plan = in._data == out._data ? _fwd_inplace : _fwd_outplace;
  if (plan == nullptr) {
    plan = fftw.plan(int(_len), in[0]._data, out[0]._data, FFTW_FORWARD);
  }

  for (auto i = 0U; i < ibatch; ++i) {
    auto tmp_in = in[i];
    auto tmp_out = out[i];
    fftw.exec(plan, tmp_in._data, tmp_out._data);
  }
}

void FFT::ifft(math::NdSlice<c32, 2> in, math::NdSlice<c32, 2> out) {
  auto& fftw = FFTW3F::instance();
  const auto [ibatch, ilen] = in._shape;
  const auto [obatch, olen] = out._shape;

  sfc::assert_(in.is_contiguous(), "FFT::ifft: in is not contiguous");
  sfc::assert_(out.is_contiguous(), "FFT::ifft: out is not contiguous");
  sfc::assert_(ilen == _len, "FFT::ifft: in.shape({}) not match len(={})", in._shape, _len);
  sfc::assert_(olen == _len, "FFT::ifft: out.shape({}) not match len(={})", out._shape, _len);
  sfc::assert_(ibatch == obatch, "FFT::ifft: in.shape({}) not match out.shape({})", in._shape, out._shape);

  auto& plan = in._data == out._data ? _inv_inplace : _inv_outplace;
  if (plan == nullptr) {
    plan = fftw.plan(int(_len), in[0]._data, out[0]._data, FFTW_BACKWARD);
  }

  for (auto i = 0U; i < ibatch; ++i) {
    auto tmp_in = in[i];
    auto tmp_out = out[i];
    fftw.exec(plan, tmp_in._data, tmp_out._data);
  }
}

RFFT::RFFT() noexcept {}

RFFT::~RFFT() {
  auto& fftw = FFTW3F::instance();
  fftw.drop(_r2c);
  fftw.drop(_c2r);
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
  auto& fftw = FFTW3F::instance();
  const auto n = num::saturating_cast<int>(len);
  const auto r2c = fftw.plan(n, ptr::null<f32>(), ptr::null<c32>(), FFTW_FORWARD);
  const auto c2r = fftw.plan(n, ptr::null<c32>(), ptr::null<f32>(), FFTW_BACKWARD);

  auto res = RFFT{};
  res._len = len;
  res._r2c = r2c;
  res._c2r = c2r;
  return res;
}

void RFFT::fft(math::NdSlice<f32, 1> in, math::NdSlice<c32, 1> out) {
  auto& fftw = FFTW3F::instance();
  const auto full_len = _len;
  const auto half_len = _len / 2 + 1;
  sfc::assert_(in.is_contiguous(), "RFFT::fft: in is not contiguous");
  sfc::assert_(out.is_contiguous(), "RFFT::fft: out is not contiguous");
  sfc::assert_(in._shape[0] == full_len, "RFFT::fft: in.shape({}) not match fft.len(={})", in._shape, _len);
  sfc::assert_(out._shape[0] == half_len, "RFFT::fft: out.shape({}) not match fft.len(={})/2+1", out._shape, _len);

  fftw.exec(_r2c, in._data, out._data);
}

void RFFT::ifft(math::NdSlice<c32, 1> in, math::NdSlice<f32, 1> out) {
  auto& fftw = FFTW3F::instance();
  const auto full_len = _len;
  const auto half_len = _len / 2 + 1;
  sfc::assert_(in.is_contiguous(), "RFFT::ifft: in is not contiguous");
  sfc::assert_(out.is_contiguous(), "RFFT::ifft: out is not contiguous");
  sfc::assert_(in._shape[0] == half_len, "RFFT::ifft: in.shape({}) not match fft.len(={})/2+1", in._shape, _len);
  sfc::assert_(out._shape[0] == full_len, "RFFT::ifft: out.shape({}) not match fft.len(={})", out._shape, _len);

  fftw.exec(_c2r, in._data, out._data);
}

void RFFT::fft(math::NdSlice<f32, 2> in, math::NdSlice<c32, 2> out) {
  auto& fftw = FFTW3F::instance();
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
    fftw.exec(_r2c, tmp_in._data, tmp_out._data);
  }
}

void RFFT::ifft(math::NdSlice<c32, 2> in, math::NdSlice<f32, 2> out) {
  auto& fftw = FFTW3F::instance();
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
    fftw.exec(_c2r, tmp_in._data, tmp_out._data);
  }
}

}  // namespace sfc::math::fft
