#include <fftw3.h>
#include "sfc/math/fft.h"

namespace sfc::math {

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

template <>
struct FFTW<c32, c32>::Inn {
  u32 _len;
  fftw_plan_t _c2c_fwd{nullptr};
  fftw_plan_t _c2c_inv{nullptr};
  fftw_plan_t _c2c_inplace_fwd{nullptr};
  fftw_plan_t _c2c_inplace_inv{nullptr};

 public:
  explicit Inn(u32 len) : _len{len} {
    _c2c_inplace_fwd = fftw_plan(len, ptr::null<c32>(), ptr::null<c32>(), -1);
    _c2c_inplace_inv = fftw_plan(len, ptr::null<c32>(), ptr::null<c32>(), +1);
  }

  ~Inn() {
    fftw_drop(_c2c_fwd);
    fftw_drop(_c2c_inv);
    fftw_drop(_c2c_inplace_fwd);
    fftw_drop(_c2c_inplace_inv);
  }

  Inn(const Inn&) = delete;
  Inn& operator=(const Inn&) = delete;

  auto plan(const c32 in[], c32 out[], int DIR) -> fftw_plan_t {
    auto& fwd_plan = in == out ? _c2c_inplace_fwd : _c2c_fwd;
    auto& inv_plan = in == out ? _c2c_inplace_inv : _c2c_inv;
    auto& plan = DIR < 0 ? fwd_plan : inv_plan;

    if (plan == nullptr) {
      plan = fftw_plan(_len, in, out, DIR);
    }

    return plan;
  }
};

template <>
struct FFTW<c32, f32>::Inn {
  fftw_plan_t _c2r{nullptr};

 public:
  explicit Inn(u32 len) {
    _c2r = fftw_plan(len, ptr::null<c32>(), ptr::null<f32>(), -1);
  }

  ~Inn() {
    fftw_drop(_c2r);
  }

  Inn(const Inn&) = delete;
  Inn& operator=(const Inn&) = delete;

  auto plan([[maybe_unused]] const c32 in[], [[maybe_unused]] f32 out[], [[maybe_unused]] int DIR) -> fftw_plan_t {
    return _c2r;
  }
};

template <>
struct FFTW<f32, c32>::Inn {
  fftw_plan_t _r2c{nullptr};

 public:
  explicit Inn(u32 len) {
    _r2c = fftw_plan(len, ptr::null<f32>(), ptr::null<c32>(), -1);
  }

  ~Inn() {
    fftw_drop(_r2c);
  }

  Inn(const Inn&) = delete;
  Inn& operator=(const Inn&) = delete;

  auto plan([[maybe_unused]] const f32 in[], [[maybe_unused]] c32 out[], [[maybe_unused]] int DIR) -> fftw_plan_t {
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
auto FFTW<I, O>::ilen() const -> usize {
  const auto half_len = _len / 2 + 1;
  return sizeof(I) <= sizeof(O) ? _len : half_len;
}

template <class I, class O>
auto FFTW<I, O>::olen() const -> usize {
  const auto half_len = _len / 2 + 1;
  return sizeof(O) <= sizeof(I) ? _len : half_len;
}

template <class I, class O>
void FFTW<I, O>::exec(const I in[], O out[], int DIR) {
  const auto plan = _inn->plan(in, out, DIR);
  fftw_exec(plan, in, out);
}

template <class I, class O>
void FFTW<I, O>::operator()(NdSlice<I, 1> in, NdSlice<O, 1> out, int DIR) {
  const auto ilen = this->ilen();
  const auto olen = this->olen();
  const auto [src_len] = in._shape;
  const auto [dst_len] = out._shape;
  sfc::assert_(in.is_contiguous(), "FFTW::exec: src is not contiguous");
  sfc::assert_(out.is_contiguous(), "FFTW::exec: dst is not contiguous");
  sfc::assert_(src_len == ilen, "FFTW::exec: src.shape({}) != ilen{}", src_len, ilen);
  sfc::assert_(dst_len == olen, "FFTW::exec: dst.shape({}) != olen{}", dst_len, olen);
  this->exec(in._data, out._data, DIR);
}

template <class I, class O>
void FFTW<I, O>::operator()(NdSlice<I, 2> src, NdSlice<O, 2> dst, int DIR) {
  const auto ilen = this->ilen();
  const auto olen = this->olen();
  const auto [src_batch, src_len] = src._shape;
  const auto [dst_batch, dst_len] = dst._shape;

  sfc::assert_(src.is_contiguous(), "FFTW::exec: src is not contiguous");
  sfc::assert_(dst.is_contiguous(), "FFTW::exec: dst is not contiguous");
  sfc::assert_(src_len == ilen, "FFTW::exec: src.shape({}) != ilen{}", src_len, ilen);
  sfc::assert_(dst_len == olen, "FFTW::exec: dst.shape({}) != olen{}", dst_len, olen);
  sfc::assert_(src_batch == dst_batch, "FFTW::exec: src.batch({}) != dst.batch({})", src_batch, dst_batch);

  for (auto i = 0U; i < src_batch; ++i) {
    auto src_col = src[i];
    auto dst_col = dst[i];
    this->exec(src_col._data, dst_col._data, DIR);
  }
}

template class FFTW<f32, c32>;
template class FFTW<c32, f32>;
template class FFTW<c32, c32>;

}  // namespace sfc::math
