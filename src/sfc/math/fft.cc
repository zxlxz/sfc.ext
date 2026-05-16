#include <fftw3.h>
#include "sfc/core.h"
#include "sfc/math/fft.h"
#include "sfc/cuda/fft.h"
#include "sfc/cuda/memory.h"

namespace sfc::math {

namespace fftw {

using fft_plan_t = fftwf_plan;

static auto fft_plan_c2c(u32 N, const c32* in, c32* out, int SIGN) -> fft_plan_t {
  auto I = reinterpret_cast<fftwf_complex*>(const_cast<c32*>(in));
  auto O = reinterpret_cast<fftwf_complex*>(out);
  auto res = fftwf_plan_dft_1d(N, I, O, SIGN, FFTW_ESTIMATE);
  return res;
}

static auto fft_plan_r2c(u32 N) -> fft_plan_t {
  auto res = fftwf_plan_dft_r2c_1d(N, nullptr, nullptr, FFTW_ESTIMATE);
  return res;
}

static auto fft_plan_c2r(u32 N) -> fft_plan_t {
  auto res = fftwf_plan_dft_c2r_1d(N, nullptr, nullptr, FFTW_ESTIMATE);
  return res;
}

static auto fft_drop(fft_plan_t plan) -> void {
  if (plan == nullptr) return;
  fftwf_destroy_plan(plan);
}

static void fft_exec_c2c(fftwf_plan plan, const c32 X[], c32 Y[]) {
  auto I = reinterpret_cast<fftwf_complex*>(const_cast<c32*>(X));
  auto O = reinterpret_cast<fftwf_complex*>(Y);
  fftwf_execute_dft(plan, I, O);
}

void fft_exec_r2c(fftwf_plan plan, const f32 X[], c32 Y[]) {
  auto I = const_cast<f32*>(X);
  auto O = reinterpret_cast<fftwf_complex*>(Y);
  fftwf_execute_dft_r2c(plan, I, O);
}

void fft_exec_c2r(fftwf_plan plan, const c32 X[], f32 Y[]) {
  auto I = reinterpret_cast<fftwf_complex*>(const_cast<c32*>(X));
  auto O = Y;
  fftwf_execute_dft_c2r(plan, I, O);
}

template <class I, class O>
struct Plan;

template <>
struct Plan<f32, c32> : FFTPlan<f32, c32> {
  u32 _len;
  u32 _batch;
  fftw::fft_plan_t _plan = nullptr;

 public:
  Plan(u32 len, u32 batch) : _len{len}, _batch{batch} {
    _plan = fftw::fft_plan_r2c(_len);
  }

  ~Plan() {
    fftw::fft_drop(_plan);
  }

  Plan(const Plan&) = delete;

 public:
  auto len() const -> u32 override {
    return _len;
  }

  auto batch() const -> u32 override {
    return _batch;
  }

  void exec(const f32 input[], c32 output[], int DIR) override {
    auto I = input;
    auto O = output;
    for (auto k = 0U; k < _batch; ++k) {
      fftw::fft_exec_r2c(_plan, I, O);
      I += _len;
      O += _len / 2 + 1;
    }
  }
};

template <>
struct Plan<c32, f32> : FFTPlan<c32, f32> {
  u32 _len;
  u32 _batch;
  fftw::fft_plan_t _plan = nullptr;

 public:
  Plan(u32 len, u32 batch) : _len{len}, _batch{batch} {
    _plan = fftw::fft_plan_r2c(_len);
  }

  ~Plan() {
    fftw::fft_drop(_plan);
  }

  Plan(const Plan&) = delete;

 public:
  auto len() const -> u32 override {
    return _len;
  }

  auto batch() const -> u32 override {
    return _batch;
  }

  void exec(const c32 input[], f32 output[], int DIR) override {
    auto I = input;
    auto O = output;
    for (auto k = 0U; k < _batch; ++k) {
      fftw::fft_exec_c2r(_plan, I, O);
      I += _len / 2 + 1;
      O += _len;
    }
  }
};

template <>
struct Plan<c32, c32> : FFTPlan<c32, c32> {
  u32 _len;
  u32 _batch;
  fftw::fft_plan_t _inplace_fwd = nullptr;
  fftw::fft_plan_t _inplace_inv = nullptr;
  fftw::fft_plan_t _outplace_fwd = nullptr;
  fftw::fft_plan_t _outplace_inv = nullptr;

 public:
  Plan(u32 len, u32 batch) : _len{len}, _batch{batch} {}

  ~Plan() {
    fftw::fft_drop(_inplace_fwd);
    fftw::fft_drop(_inplace_inv);
    fftw::fft_drop(_outplace_fwd);
    fftw::fft_drop(_outplace_inv);
  }

  Plan(const Plan&) = delete;

 public:
  auto len() const -> u32 override {
    return _len;
  }

  auto batch() const -> u32 override {
    return _batch;
  }

  void exec(const c32 input[], c32 output[], int DIR) override {
    auto p = this->get_plan(input, output, DIR);

    auto X = input;
    auto Y = output;
    for (auto k = 0U; k < _batch; ++k) {
      fftw::fft_exec_c2c(p, X, Y);
      X += _len;
      Y += _len;
    }
  }

  auto get_plan(const c32 I[], c32 O[], int DIR) -> fftw::fft_plan_t {
    auto& p = I == O ? (DIR == -1 ? _inplace_fwd : _inplace_inv)
                     : (DIR == -1 ? _outplace_fwd : _outplace_inv);
    if (p == nullptr) {
      p = fftw::fft_plan_c2c(_len, I, O, DIR);
    }
    return p;
  }
};

}  // namespace fftw

namespace cufft {
template <class I, class O>
struct Plan : FFTPlan<I, O> {
  u32 _len;
  u32 _batch;
  cuda::fft_plan_t _plan = -1;

 public:
  Plan(u32 len, u32 batch) : _len{len}, _batch{batch} {
    if constexpr (sfc::same_<I, c32> && sfc::same_<O, c32>) {
      _plan = cuda::fft_plan_c2c(_len, _batch);
    } else if constexpr (sfc::same_<I, f32> && sfc::same_<O, c32>) {
      _plan = cuda::fft_plan_r2c(_len, _batch);
    } else if constexpr (sfc::same_<I, c32> && sfc::same_<O, f32>) {
      _plan = cuda::fft_plan_c2r(_len, _batch);
    }
  }

  ~Plan() {
    if (_plan == -1) return;
    cuda::fft_drop(_plan);
  }

  Plan(const Plan&) = delete;

  auto len() const -> u32 override {
    return _len;
  }

  auto batch() const -> u32 override {
    return _batch;
  }

  void exec(const I X[], O Y[], int DIR) override {
    if constexpr (sfc::same_<I, c32> && sfc::same_<O, c32>) {
      cuda::fft_exec_c2c(_plan, X, Y, DIR);
    } else if constexpr (sfc::same_<I, f32> && sfc::same_<O, c32>) {
      cuda::fft_exec_r2c(_plan, X, Y);
    } else if constexpr (sfc::same_<I, c32> && sfc::same_<O, f32>) {
      cuda::fft_exec_c2r(_plan, X, Y);
    }
  }
};
}  // namespace cufft

template <class I, class O>
FFT<I, O>::FFT() : _inn{nullptr} {}

template <class I, class O>
FFT<I, O>::~FFT() {
  if (_inn == nullptr) return;
  delete _inn;
}

template <class I, class O>
FFT<I, O>::FFT(FFT&& other) noexcept : _inn{other._inn} {
  other._inn = nullptr;
}

template <class I, class O>
auto FFT<I, O>::operator=(FFT&& other) noexcept -> FFT& {
  if (this == &other) return *this;
  mem::swap(_inn, other._inn);
  return *this;
}

template <class I, class O>
auto FFT<I, O>::create(u32 len, u32 batch, cuda::MemType mtype) -> FFT {
  const auto use_gpu = mtype == cuda::MemType::Device || mtype == cuda::MemType::UVA;

  auto res = FFT{};
  if (use_gpu) {
    res._inn = new cufft::Plan<I, O>(len, batch);
  } else {
    res._inn = new fftw::Plan<I, O>(len, batch);
  }
  return res;
}

template <class I, class O>
void FFT<I, O>::forward(const I in[], O out[]) {
  return _inn->exec(in, out, -1);
}

template <class I, class O>
void FFT<I, O>::inverse(const I in[], O out[]) {
  return _inn->exec(in, out, +1);
}

template <class I, class O>
void FFT<I, O>::operator()(math::NdSlice<I, 2> in, math::NdSlice<O, 2> out) {
  return _inn->exec(in._data, out._data, -1);
}

template class FFT<c32, c32>;
template class FFT<f32, c32>;
template class FFT<c32, f32>;

void fft_c2c(u32 N, const c32 X[], c32 Y[], int direction) {
  auto plan = fftw::Plan<c32, c32>(N, 1);
  plan.exec(X, Y, direction);
}

void fft_r2c(u32 N, const f32 X[], c32 Y[]) {
  auto plan = fftw::Plan<f32, c32>(N, 1);
  plan.exec(X, Y, -1);
}

void fft_c2r(u32 N, const c32 X[], f32 Y[]) {
  auto plan = fftw::Plan<c32, f32>(N, 1);
  plan.exec(X, Y, +1);
}

void fft(math::NdSlice<c32, 1> in, math::NdSlice<c32, 1> out) {
  panic::expect(in.len() == out.len(), "input len(=`{}`) and output.len(=`{}`) must equal",
                in.len(), out.len());

  auto plan = fftw::Plan<c32, c32>{in.len(), 1};
  plan.exec(in._data, out._data, -1);
}

void ifft(math::NdSlice<c32, 1> in, math::NdSlice<c32, 1> out) {
  panic::expect(in.len() == out.len(), "input len(=`{}`) and output.len(=`{}`) must equal",
                in.len(), out.len());

  auto plan = fftw::Plan<c32, c32>{in.len(), 1};
  plan.exec(in._data, out._data, +1);
}

}  // namespace sfc::math
