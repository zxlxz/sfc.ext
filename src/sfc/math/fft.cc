#include <fftw3.h>
#include "sfc/math/fft.h"
#include "sfc/ffi/library.h"

namespace sfc::math {

namespace detail {

using fft_plan_t = fftwf_plan;

class FFTW3F {
  decltype(fftwf_destroy_plan)* _fftwf_destroy_plan = nullptr;
  decltype(fftwf_plan_dft_1d)* _fftwf_plan_dft_1d = nullptr;
  decltype(fftwf_plan_dft_r2c_1d)* _fftwf_plan_dft_r2c_1d = nullptr;
  decltype(fftwf_plan_dft_c2r_1d)* _fftwf_plan_dft_c2r_1d = nullptr;
  decltype(fftwf_execute_dft)* _fftwf_execute_dft = nullptr;
  decltype(fftwf_execute_dft_r2c)* _fftwf_execute_dft_r2c = nullptr;
  decltype(fftwf_execute_dft_c2r)* _fftwf_execute_dft_c2r = nullptr;

 public:
  static auto from(ffi::Library& lib) {
    auto res = FFTW3F{};
    res._fftwf_destroy_plan = lib.func("fftwf_destroy_plan");
    res._fftwf_plan_dft_1d = lib.func("fftwf_plan_dft_1d");
    res._fftwf_plan_dft_r2c_1d = lib.func("fftwf_plan_dft_r2c_1d");
    res._fftwf_plan_dft_c2r_1d = lib.func("fftwf_plan_dft_c2r_1d");
    res._fftwf_execute_dft = lib.func("fftwf_execute_dft");
    res._fftwf_execute_dft_r2c = lib.func("fftwf_execute_dft_r2c");
    res._fftwf_execute_dft_c2r = lib.func("fftwf_execute_dft_c2r");
    return res;
  }

 public:
  void destroy(fftwf_plan p) {
    _fftwf_destroy_plan(p);
  }

  auto plan_1d_c2c(int len, c32 in[], c32 out[], int sign) -> fftwf_plan {
    const auto idata = ptr::cast<fftwf_complex>(in);
    const auto odata = ptr::cast<fftwf_complex>(out);
    return _fftwf_plan_dft_1d(len, idata, odata, sign, FFTW_ESTIMATE);
  }

  auto plan_1d_r2c(int len, f32 in[], c32 out[]) -> fftwf_plan {
    const auto idata = in;
    const auto odata = ptr::cast<fftwf_complex>(out);
    return _fftwf_plan_dft_r2c_1d(len, idata, odata, FFTW_ESTIMATE);
  }

  auto plan_1d_c2r(int len, c32 in[], f32 out[]) -> fftwf_plan {
    const auto idata = ptr::cast<fftwf_complex>(in);
    const auto odata = out;
    return _fftwf_plan_dft_c2r_1d(len, idata, odata, FFTW_ESTIMATE);
  }

  void exec_c2c(fftwf_plan plan, c32 in[], c32 out[]) {
    const auto idata = ptr::cast<fftwf_complex>(in);
    const auto odata = ptr::cast<fftwf_complex>(out);
    _fftwf_execute_dft(plan, idata, odata);
  }

  void exec_r2c(fftwf_plan plan, f32 in[], c32 out[]) {
    const auto idata = in;
    const auto odata = ptr::cast<fftwf_complex>(out);
    _fftwf_execute_dft_r2c(plan, idata, odata);
  }

  void exec_c2r(fftwf_plan plan, c32 in[], f32 out[]) {
    const auto idata = ptr::cast<fftwf_complex>(in);
    const auto odata = out;
    _fftwf_execute_dft_c2r(plan, idata, odata);
  }
};

auto fft_lib() -> FFTW3F& {
#ifdef _WIN32
  const auto path = Str{"libfftw3f-3.dll"};
#else
  const auto path = Str{"libfftw3f"};
#endif
  static auto lib = ffi::Library::load(path).unwrap();
  static auto fft = FFTW3F::from(lib);
  return fft;
}

void fft_destroy(fftwf_plan plan) {
  if (plan == nullptr) {
    return;
  }

  auto& lib = fft_lib();
  lib.destroy(plan);
}

template <class I, class O>
auto fft_plan_1d(u32 len, I in[], O out[], int sign) -> fftwf_plan {
  auto& lib = fft_lib();

  if constexpr (trait::same_<I, c32> && trait::same_<O, c32>) {
    return lib.plan_1d_c2c(int(len), in, out, sign);
  } else if constexpr (trait::same_<I, f32> && trait::same_<O, c32>) {
    return lib.plan_1d_r2c(int(len), in, out);
  } else if constexpr (trait::same_<I, c32> && trait::same_<O, f32>) {
    return lib.plan_1d_c2r(int(len), in, out);
  } else {
    static_assert(false, "unsupported type combination");
  }
}

template <class I, class O>
void fft_exec(fftwf_plan plan, I in[], O out[]) {
  auto& lib = fft_lib();
  if constexpr (trait::same_<I, c32> && trait::same_<O, c32>) {
    lib.exec_c2c(plan, in, out);
  } else if constexpr (trait::same_<I, f32> && trait::same_<O, c32>) {
    lib.exec_r2c(plan, in, out);
  } else if constexpr (trait::same_<I, c32> && trait::same_<O, f32>) {
    lib.exec_c2r(plan, in, out);
  } else {
    static_assert(false, "unsupported type combination");
  }
}

template <class I, class O>
void fft_exec_batch(fftwf_plan plan, math::NdView<I, 2> in, math::NdView<O, 2> out, u32 batch) {
  for (auto i = 0U; i < batch; ++i) {
    auto idata = in[i]._data;
    auto odata = out[i]._data;
    detail::fft_exec(plan, idata, odata);
  }
}

}  // namespace detail

CFFT::CFFT() noexcept {}

CFFT::~CFFT() {
  detail::fft_destroy(_fwd_inplace);
  detail::fft_destroy(_inv_inplace);
  detail::fft_destroy(_fwd_outplace);
  detail::fft_destroy(_inv_outplace);
}

CFFT::CFFT(CFFT&& other) noexcept
    : _len{mem::take(other._len)}
    , _fwd_inplace{mem::take(other._fwd_inplace)}
    , _inv_inplace{mem::take(other._inv_inplace)}
    , _fwd_outplace{mem::take(other._fwd_outplace)}
    , _inv_outplace{mem::take(other._inv_outplace)} {}

CFFT& CFFT::operator=(CFFT&& other) noexcept {
  if (this == &other) return *this;
  mem::swap(_len, other._len);
  mem::swap(_fwd_inplace, other._fwd_inplace);
  mem::swap(_inv_inplace, other._inv_inplace);
  mem::swap(_fwd_outplace, other._fwd_outplace);
  mem::swap(_inv_outplace, other._inv_outplace);

  return *this;
}

CFFT CFFT::new_(u32 len) {
  const auto fwd_inplace = detail::fft_plan_1d(len, ptr::null<c32>(), ptr::null<c32>(), FFTW_FORWARD);
  const auto inv_inplace = detail::fft_plan_1d(len, ptr::null<c32>(), ptr::null<c32>(), FFTW_BACKWARD);

  auto res = CFFT{};
  res._len = len;
  res._fwd_inplace = fwd_inplace;
  res._inv_inplace = inv_inplace;
  return res;
}

auto CFFT::len() const -> usize {
  return _len;
}

void CFFT::fft(math::NdView<c32, 1> in, math::NdView<c32, 1> out) {
  sfc::assert_(in.is_contiguous(), "CFFT::fft: in is not contiguous");
  sfc::assert_(out.is_contiguous(), "CFFT::fft: out is not contiguous");
  sfc::assert_(in._shape[0] == _len, "CFFT::fft: in.shape({}) not match len(={})", in._shape, _len);
  sfc::assert_(out._shape[0] == _len, "CFFT::fft: out.shape({}) not match len(={})", out._shape, _len);

  auto& plan = in._data == out._data ? _fwd_inplace : _fwd_outplace;
  if (plan == nullptr) {
    plan = detail::fft_plan_1d(_len, in._data, out._data, FFTW_FORWARD);
  }

  detail::fft_exec(plan, in._data, out._data);
}

void CFFT::ifft(math::NdView<c32, 1> in, math::NdView<c32, 1> out) {
  sfc::assert_(in.is_contiguous(), "CFFT::ifft: in is not contiguous");
  sfc::assert_(out.is_contiguous(), "CFFT::ifft: out is not contiguous");
  sfc::assert_(in._shape[0] == _len, "CFFT::ifft: in.shape({}) not match len(={})", in._shape, _len);
  sfc::assert_(out._shape[0] == _len, "CFFT::ifft: out.shape({}) not match len(={})", out._shape, _len);

  auto& plan = in._data == out._data ? _inv_inplace : _inv_outplace;
  if (plan == nullptr) {
    plan = detail::fft_plan_1d(_len, in._data, out._data, FFTW_BACKWARD);
  }
  detail::fft_exec(plan, in._data, out._data);
}

void CFFT::fft(math::NdView<c32, 2> in, math::NdView<c32, 2> out) {
  const auto [ibatch, ilen] = in._shape;
  const auto [obatch, olen] = out._shape;

  sfc::assert_(in.is_contiguous(), "CFFT::fft: in is not contiguous");
  sfc::assert_(out.is_contiguous(), "CFFT::fft: out is not contiguous");
  sfc::assert_(ilen == _len, "CFFT::fft: in.shape({}) not match len(={})", in._shape, _len);
  sfc::assert_(olen == _len, "CFFT::fft: out.shape({}) not match len(={})", out._shape, _len);
  sfc::assert_(ibatch == obatch, "CFFT::fft: in.shape({}) not match out.shape({})", in._shape, out._shape);

  auto& plan = in._data == out._data ? _fwd_inplace : _fwd_outplace;
  if (plan == nullptr) {
    plan = detail::fft_plan_1d(_len, in[0]._data, out[0]._data, FFTW_FORWARD);
  }

  detail::fft_exec_batch(plan, in, out, u32(ibatch));
}

void CFFT::ifft(math::NdView<c32, 2> in, math::NdView<c32, 2> out) {
  const auto [ibatch, ilen] = in._shape;
  const auto [obatch, olen] = out._shape;

  sfc::assert_(in.is_contiguous(), "CFFT::ifft: in is not contiguous");
  sfc::assert_(out.is_contiguous(), "CFFT::ifft: out is not contiguous");
  sfc::assert_(ilen == _len, "CFFT::ifft: in.shape({}) not match len(={})", in._shape, _len);
  sfc::assert_(olen == _len, "CFFT::ifft: out.shape({}) not match len(={})", out._shape, _len);
  sfc::assert_(ibatch == obatch, "CFFT::ifft: in.shape({}) not match out.shape({})", in._shape, out._shape);

  auto& plan = in._data == out._data ? _inv_inplace : _inv_outplace;
  if (plan == nullptr) {
    plan = detail::fft_plan_1d(_len, in[0]._data, out[0]._data, FFTW_BACKWARD);
  }

  detail::fft_exec_batch(plan, in, out, u32(ibatch));
}

RFFT::RFFT() noexcept {}

RFFT::~RFFT() {
  detail::fft_destroy(_r2c);
  detail::fft_destroy(_c2r);
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
  const auto r2c = detail::fft_plan_1d(len, ptr::null<f32>(), ptr::null<c32>(), FFTW_FORWARD);
  const auto c2r = detail::fft_plan_1d(len, ptr::null<c32>(), ptr::null<f32>(), FFTW_BACKWARD);

  auto res = RFFT{};
  res._len = len;
  res._r2c = r2c;
  res._c2r = c2r;
  return res;
}

void RFFT::fft(math::NdView<f32, 1> in, math::NdView<c32, 1> out) {
  const auto full_len = _len;
  const auto half_len = _len / 2 + 1;
  sfc::assert_(in.is_contiguous(), "RFFT::fft: in is not contiguous");
  sfc::assert_(out.is_contiguous(), "RFFT::fft: out is not contiguous");
  sfc::assert_(in._shape[0] == full_len, "RFFT::fft: in.shape({}) not match fft.len(={})", in._shape, _len);
  sfc::assert_(out._shape[0] == half_len, "RFFT::fft: out.shape({}) not match fft.len(={})/2+1", out._shape, _len);

  detail::fft_exec(_r2c, in._data, out._data);
}

void RFFT::ifft(math::NdView<c32, 1> in, math::NdView<f32, 1> out) {
  const auto full_len = _len;
  const auto half_len = _len / 2 + 1;
  sfc::assert_(in.is_contiguous(), "RFFT::ifft: in is not contiguous");
  sfc::assert_(out.is_contiguous(), "RFFT::ifft: out is not contiguous");
  sfc::assert_(in._shape[0] == half_len, "RFFT::ifft: in.shape({}) not match fft.len(={})/2+1", in._shape, _len);
  sfc::assert_(out._shape[0] == full_len, "RFFT::ifft: out.shape({}) not match fft.len(={})", out._shape, _len);

  detail::fft_exec(_c2r, in._data, out._data);
}

void RFFT::fft(math::NdView<f32, 2> in, math::NdView<c32, 2> out) {
  const auto full_len = _len;
  const auto half_len = _len / 2 + 1;
  const auto [ibatch, ilen] = in._shape;
  const auto [obatch, olen] = out._shape;

  sfc::assert_(in.is_contiguous(), "RFFT::fft: in is not contiguous");
  sfc::assert_(out.is_contiguous(), "RFFT::fft: out is not contiguous");
  sfc::assert_(ilen == full_len, "RFFT::fft: in.shape({}) not match fft.len(={})", in._shape, _len);
  sfc::assert_(olen == half_len, "RFFT::fft: out.shape({}) not match fft.len(={})/2+1", out._shape, _len);
  sfc::assert_(ibatch == obatch, "RFFT::fft: in.shape({}) not match out.shape({})", in._shape, out._shape);

  detail::fft_exec_batch(_r2c, in, out, ibatch);
}

void RFFT::ifft(math::NdView<c32, 2> in, math::NdView<f32, 2> out) {
  const auto full_len = _len;
  const auto half_len = _len / 2 + 1;
  const auto [ibatch, ilen] = in._shape;
  const auto [obatch, olen] = out._shape;

  sfc::assert_(in.is_contiguous(), "RFFT::ifft: in is not contiguous");
  sfc::assert_(out.is_contiguous(), "RFFT::ifft: out is not contiguous");
  sfc::assert_(ilen == half_len, "RFFT::ifft: in.shape({}) not match fft.len(={})/2+1", in._shape, _len);
  sfc::assert_(olen == full_len, "RFFT::ifft: out.shape({}) not match fft.len(={})", out._shape, _len);
  sfc::assert_(ibatch == obatch, "RFFT::ifft: in.shape({}) not match out.shape({})", in._shape, out._shape);

  detail::fft_exec_batch(_c2r, in, out, ibatch);
}

}  // namespace sfc::math
