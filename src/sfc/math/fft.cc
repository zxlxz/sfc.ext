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
    _plan = fft_plan(static_cast<int>(N), (I*)nullptr, (O*)nullptr, 0);
  }

  ~Inn() {
    fft_drop(_plan);
  }

  void exec(const I in[], O out[], [[maybe_unused]] int SIGN = 0) {
    fft_exec(_plan, const_cast<I*>(in), out);
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
    fft_drop(_inplace[0]);
    fft_drop(_inplace[1]);
    fft_drop(_outplace[0]);
    fft_drop(_outplace[1]);
  }

  void exec(const c32 Ic[], c32 Oc[], int SIGN) {
    const auto id = SIGN < 0 ? 0 : 1;
    auto& p = Ic == Oc ? _inplace[id] : _outplace[id];
    if (p == nullptr) {
      p = fft_plan(_len, const_cast<c32*>(Ic), Oc, SIGN);
    }
    fft_exec(p, const_cast<c32*>(Ic), Oc);
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
FFT<I, O>::FFT(FFT&& other) noexcept : _len{other._len}, _batch{other._batch}, _inn{other._inn} {
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
void FFT<I, O>::check_size(const u32 (&idim)[2], const u32 (&odim)[2]) const {
  const auto full_len = _len;
  const auto half_len = full_len / 2 + 1;
  const auto ilen = trait::same_<O, c32> ? full_len : half_len;
  const auto olen = trait::same_<I, c32> ? full_len : half_len;

  panic::expect(idim[0] == ilen, "idim(={}) not match plan(in_len={})", idim, ilen);
  panic::expect(odim[0] == olen, "odim(={}) not match plan(out_len={})", odim, olen);

  panic::expect(idim[1] == _batch, "idim(={}) not match plan(batch={})", idim, _batch);
  panic::expect(odim[1] == _batch, "odim(={}) not match plan(batch={})", odim, _batch);
}

template <class I, class O>
void FFT<I, O>::exec(const I in[], O out[], int DIR) {
  panic::expect(_inn, "FFT not initialized");
  const auto SIGN = DIR <= 0 ? -1 : +1;

  const auto ilen = trait::same_<O, c32> ? _len : _len / 2 + 1;
  const auto olen = trait::same_<I, c32> ? _len : _len / 2 + 1;

  auto idata = in;
  auto odata = out;
  for (auto k = 0U; k < _batch; ++k) {
    _inn->exec(idata, odata, SIGN);
    idata += ilen;
    odata += olen;
  }
}

template <class I, class O>
void FFT<I, O>::operator()(NdSlice<I, 1> in, NdSlice<O, 1> out, int DIR) {
  this->check_size({in._dims.x, 1}, {out._dims.x, 1});
  return this->exec(in._data, out._data, DIR);
}

template <class I, class O>
void FFT<I, O>::operator()(NdSlice<I, 2> in, NdSlice<O, 2> out, int DIR) {
  this->check_size({in._dims.x, in._dims.y}, {out._dims.x, out._dims.y});
  return this->exec(in._data, out._data, DIR);
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
