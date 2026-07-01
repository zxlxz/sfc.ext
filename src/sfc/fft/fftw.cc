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

template <>
struct FFTW<c32, c32>::Inn {
  u32 _len;
  fftw::fft_plan_t _c2c_fwd{nullptr};
  fftw::fft_plan_t _c2c_inv{nullptr};
  fftw::fft_plan_t _c2c_inplace_fwd{nullptr};
  fftw::fft_plan_t _c2c_inplace_inv{nullptr};

 public:
  explicit Inn(u32 len) : _len{len} {
    _c2c_inplace_fwd = fftw::fft_plan<c32, c32>(len, nullptr, nullptr, -1);
    _c2c_inplace_inv = fftw::fft_plan<c32, c32>(len, nullptr, nullptr, +1);
  }

  ~Inn() {
    fftw::fft_drop(_c2c_fwd);
    fftw::fft_drop(_c2c_inv);
    fftw::fft_drop(_c2c_inplace_fwd);
    fftw::fft_drop(_c2c_inplace_inv);
  }

  Inn(const Inn&) = delete;
  Inn& operator=(const Inn&) = delete;

  auto plan(const c32 in[], c32 out[], int DIR) -> fftw::fft_plan_t {
    auto& fwd_plan = in == out ? _c2c_inplace_fwd : _c2c_fwd;
    auto& inv_plan = in == out ? _c2c_inplace_inv : _c2c_inv;
    auto& plan = DIR < 0 ? fwd_plan : inv_plan;

    if (plan == nullptr) {
      plan = fftw::fft_plan<c32, c32>(_len, in, out, DIR);
    }

    return plan;
  }
};

template <>
struct FFTW<c32, f32>::Inn {
  using plan_t = fftw::fft_plan_t;
  plan_t _c2r{nullptr};

 public:
  explicit Inn(u32 len) {
    _c2r = fftw::fft_plan<c32, f32>(len, nullptr, nullptr, -1);
  }

  ~Inn() {
    fftw::fft_drop(_c2r);
  }

  Inn(const Inn&) = delete;
  Inn& operator=(const Inn&) = delete;

  auto plan([[maybe_unused]] const c32 in[], [[maybe_unused]] f32 out[], [[maybe_unused]] int DIR) -> plan_t {
    return _c2r;
  }
};

template <>
struct FFTW<f32, c32>::Inn {
  using plan_t = fftw::fft_plan_t;
  plan_t _r2c{nullptr};

 public:
  explicit Inn(u32 len) {
    _r2c = fftw::fft_plan<f32, c32>(len, nullptr, nullptr, -1);
  }

  ~Inn() {
    fftw::fft_drop(_r2c);
  }

  Inn(const Inn&) = delete;
  Inn& operator=(const Inn&) = delete;

  auto plan([[maybe_unused]] const f32 in[], [[maybe_unused]] c32 out[], [[maybe_unused]] int DIR) -> plan_t {
    return _r2c;
  }
};

template <class I, class O>
FFTW<I, O>::FFTW() noexcept {}

template <class I, class O>
FFTW<I, O>::~FFTW() {
  if (_inn == nullptr) {
    return;
  }
  delete _inn;
}

template <class I, class O>
FFTW<I, O>::FFTW(FFTW&& other) noexcept : _len{other._len}, _inn{other._inn} {
  other._len = 0;
  other._inn = nullptr;
}

template <class I, class O>
FFTW<I, O>& FFTW<I, O>::operator=(FFTW&& other) noexcept {
  if (this != &other) {
    mem::swap(_len, other._len);
    mem::swap(_inn, other._inn);
  }
  return *this;
}

template <class I, class O>
auto FFTW<I, O>::create(u32 len) -> FFTW {
  auto res = FFTW{};
  res._len = len;
  res._inn = new Inn(len);
  return res;
}

template <class I, class O>
auto FFTW<I, O>::in_len() const -> usize {
  const auto half_len = _len / 2 + 1;
  return sizeof(I) <= sizeof(O) ? _len : half_len;
}

template <class I, class O>
auto FFTW<I, O>::out_len() const -> usize {
  const auto half_len = _len / 2 + 1;
  return sizeof(O) <= sizeof(I) ? _len : half_len;
}

template <class I, class O>
void FFTW<I, O>::exec(const I in[], O out[], int DIR) {
  const auto plan = _inn->plan(in, out, DIR);
  fftw::fft_exec(plan, in, out);
}

template <class I, class O>
void FFTW<I, O>::operator()(math::NdArray<I, 1>& in, math::NdArray<O, 1>& out, int DIR) {
  const auto ilen = this->in_len();
  const auto olen = this->out_len();
  sfc::assert_eq(in.shape()[0], ilen);
  sfc::assert_eq(out.shape()[0], olen);
  this->exec(in.data(), out.data(), DIR);
}

template <class I, class O>
void FFTW<I, O>::operator()(math::NdArray<I, 2>& in, math::NdArray<O, 2>& out, int DIR) {
  const auto in_batch = in.shape()[0];
  const auto out_batch = out.shape()[0];
  sfc::assert_eq(in_batch, out_batch);

  const auto ilen = this->in_len();
  const auto olen = this->out_len();
  sfc::assert_eq(in.shape()[1], ilen);
  sfc::assert_eq(out.shape()[1], olen);

  for (auto b = 0U; b < in_batch; ++b) {
    auto in_col = in[b];
    auto out_col = out[b];
    this->exec(in_col._data, out_col._data, DIR);
  }
}

template class FFTW<f32, c32>;
template class FFTW<c32, f32>;
template class FFTW<c32, c32>;

}  // namespace sfc::fft
