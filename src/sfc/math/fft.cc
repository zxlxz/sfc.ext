#include <fftw3.h>

#include "sfc/core.h"
#include "sfc/math/fft.h"

namespace sfc::math {

using fft_plan_t = fftwf_plan_s*;

static void fft_drop(fft_plan_t p) {
  if (p == nullptr) return;
  ::fftwf_destroy_plan(p);
}

static auto fft_plan(int N, c32* I, c32* O, int SIGN) -> fft_plan_t {
  return ::fftwf_plan_dft_1d(N, (fftwf_complex*)I, (fftwf_complex*)O, SIGN, FFTW_ESTIMATE);
}

static auto fft_plan(int N, f32* I, c32* O, [[maybe_unused]] int SIGN = 0) -> fft_plan_t {
  return ::fftwf_plan_dft_r2c_1d(N, I, (fftwf_complex*)O, FFTW_ESTIMATE);
}

static auto fft_plan(int N, c32* I, f32* O, [[maybe_unused]] int SIGN = 0) -> fft_plan_t {
  return ::fftwf_plan_dft_c2r_1d(N, (fftwf_complex*)I, (f32*)O, FFTW_ESTIMATE);
}

static void fft_exec(fft_plan_t plan, c32 I[], c32 O[]) {
  ::fftwf_execute_dft(plan, (fftwf_complex*)I, (fftwf_complex*)O);
}

static void fft_exec(fft_plan_t plan, f32 I[], c32 O[]) {
  ::fftwf_execute_dft_r2c(plan, (f32*)I, (fftwf_complex*)O);
}

static void fft_exec(fft_plan_t plan, c32 I[], f32 O[]) {
  ::fftwf_execute_dft_c2r(plan, (fftwf_complex*)I, (f32*)O);
}

template <class I, class O>
struct FFT<I, O>::Inn {
  fft_plan_t _plan{nullptr};

 public:
  explicit Inn(u32 N) {
    _plan = math::fft_plan(static_cast<int>(N), (I*)nullptr, (O*)nullptr, 0);
  }

  ~Inn() {
    math::fft_drop(_plan);
  }

  void exec(const I in[], O out[], [[maybe_unused]] int SIGN = 0) {
    math::fft_exec(_plan, const_cast<I*>(in), out);
  }
};

template <>
struct FFT<c32, c32>::Inn {
  int _len;
  fft_plan_t _inplace[2] = {nullptr, nullptr};
  fft_plan_t _outplace[2] = {nullptr, nullptr};

 public:
  Inn(u32 len) : _len{static_cast<int>(len)} {}

  ~Inn() {
    math::fft_drop(_inplace[0]);
    math::fft_drop(_inplace[1]);
    math::fft_drop(_outplace[0]);
    math::fft_drop(_outplace[1]);
  }

  void exec(const c32 Ic[], c32 Oc[], int SIGN) {
    const auto id = SIGN < 0 ? 0 : 1;
    auto& p = Ic == Oc ? _inplace[id] : _outplace[id];
    if (p == nullptr) {
      p = math::fft_plan(_len, const_cast<c32*>(Ic), Oc, SIGN);
    }
    math::fft_exec(p, const_cast<c32*>(Ic), Oc);
  }
};

template <class I, class O>
FFT<I, O>::FFT() noexcept : _inn{nullptr} {}

template <class I, class O>
FFT<I, O>::~FFT() {
  if (!_inn) return;
  delete _inn;
}

template <class I, class O>
FFT<I, O>::FFT(FFT&& other) noexcept : _inn{other._inn}, _len{other._len}, _batch{other._batch} {
  other._inn = nullptr;
  other._len = 0;
  other._batch = 0;
}

template <class I, class O>
FFT<I, O>& FFT<I, O>::operator=(FFT&& other) noexcept {
  if (this == &other) return *this;
  mem::swap(_inn, other._inn);
  mem::swap(_len, other._len);
  mem::swap(_batch, other._batch);
  return *this;
}

template <class I, class O>
auto FFT<I, O>::create(u32 len, u32 batch) -> FFT {
  auto res = FFT{};
  res._len = len;
  res._batch = batch;
  res._inn = new Inn{len};
  return res;
}

template <class I, class O>
void FFT<I, O>::exec(const I in[], O out[], int DIR) {
  panic::expect(_inn, "FFT not initialized");

  const auto SIGN = DIR <= 0 ? -1 : +1;
  const auto ilen = trait::same_<O, c32> ? _len : _len / 2 + 1;
  const auto olen = trait::same_<O, c32> ? _len : _len / 2 + 1;

  auto pi = in;
  auto po = out;
  for (auto k = 0U; k < _batch; ++k) {
    _inn->exec(pi, po, SIGN);
    pi += _len;
    po += _len / 2 + 1;
  }
}

template <class I, class O>
void FFT<I, O>::operator()(math::NdSlice<I, 1> i, math::NdSlice<O, 1> o, int DIR) {
  panic::expect(_batch == 1, "batch size must be 1 for 1D slice");

  const auto ilen = trait::same_<O, c32> ? _len : _len / 2 + 1;
  const auto olen = trait::same_<I, c32> ? _len : _len / 2 + 1;
  panic::expect(i._dims.x == ilen, "in.shape(=`{}`) not match plan(=`{}`)", i._dims, ilen);
  panic::expect(o._dims.x == olen, "out.shape(=`{}`) not match plan(=`{}`)", o._dims, olen);
  return this->exec(i._data, o._data, DIR);
}

template <class I, class O>
void FFT<I, O>::operator()(math::NdSlice<I, 2> i, math::NdSlice<O, 2> o, int DIR) {
  using math::vec2u;

  const auto ilen = trait::same_<O, c32> ? _len : _len / 2 + 1;
  const auto olen = trait::same_<I, c32> ? _len : _len / 2 + 1;
  const auto idim = vec2u{ilen, _batch};
  const auto odim = vec2u{olen, _batch};
  panic::expect(i._dims == idim, "in.shape(=`{}`) not match plan(=`{}`)", i._dims, idim);
  panic::expect(o._dims == odim, "out.shape(=`{}`) not match plan(=`{}`)", o._dims, odim);
  return this->exec(i._data, o._data, DIR);
}

template class FFT<f32, c32>;
template class FFT<c32, f32>;
template class FFT<c32, c32>;

void fft(math::NdSlice<c32, 1> in, math::NdSlice<c32, 1> out) {
  const auto is = in._dims.x;
  const auto os = out._dims.x;
  panic::expect(is == os, "input len(=`{}`) and output.len(=`{}`) must equal", is, os);
  if (is == 0) {
    return;
  }

  auto plan = FFT<c32, c32>::create(is, 1);
  plan.exec(in._data, out._data, -1);
}

void ifft(math::NdSlice<c32, 1> in, math::NdSlice<c32, 1> out) {
  const auto is = in._dims.x;
  const auto os = out._dims.x;
  panic::expect(is == os, "input(len=`{}`) and output(len=`{}`) must equal", is, os);
  if (is == 0) {
    return;
  }

  auto plan = FFT<c32, c32>::create(is, 1);
  plan.exec(in._data, out._data, +1);
}

}  // namespace sfc::math
