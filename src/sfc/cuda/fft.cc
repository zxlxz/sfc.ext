#include <cuda.h>
#include <cufft.h>

#include "sfc/core.h"
#include "sfc/cuda/fft.h"
#include "sfc/cuda/error.h"
#include "sfc/cuda/stream.h"

namespace sfc::cuda {

using fft_plan_t = int;

struct FFTError {
  cufftResult _code;

 public:
  auto to_str() const -> const char* {
    switch (_code) {
      case CUFFT_SUCCESS:            return "CUFFT_SUCCESS";
      case CUFFT_INVALID_PLAN:       return "CUFFT_INVALID_PLAN";
      case CUFFT_ALLOC_FAILED:       return "CUFFT_ALLOC_FAILED";
      case CUFFT_INVALID_TYPE:       return "CUFFT_INVALID_TYPE";
      case CUFFT_INVALID_VALUE:      return "CUFFT_INVALID_VALUE";
      case CUFFT_INTERNAL_ERROR:     return "CUFFT_INTERNAL_ERROR";
      case CUFFT_EXEC_FAILED:        return "CUFFT_EXEC_FAILED";
      case CUFFT_SETUP_FAILED:       return "CUFFT_SETUP_FAILED";
      case CUFFT_INVALID_SIZE:       return "CUFFT_INVALID_SIZE";
      case CUFFT_UNALIGNED_DATA:     return "CUFFT_UNALIGNED_DATA";
      case CUFFT_INVALID_DEVICE:     return "CUFFT_INVALID_DEVICE";
      case CUFFT_NO_WORKSPACE:       return "CUFFT_NO_WORKSPACE";
      case CUFFT_NOT_IMPLEMENTED:    return "CUFFT_NOT_IMPLEMENTED";
      case CUFFT_NOT_SUPPORTED:      return "CUFFT_NOT_SUPPORTED";
      case CUFFT_MISSING_DEPENDENCY: return "CUFFT_MISSING_DEPENDENCY";
      case CUFFT_NVRTC_FAILURE:      return "CUFFT_NVRTC_FAILURE";
      case CUFFT_NVJITLINK_FAILURE:  return "CUFFT_NVJITLINK_FAILURE";
      case CUFFT_NVSHMEM_FAILURE:    return "CUFFT_NVSHMEM_FAILURE";
      default:                       return "CUFFT_ERROR_UNKNOWN";
    }
  }

  void fmt(auto& f) const {
    const auto s = this->to_str();
    f.write_str(s);
  }
};

void fft_drop(fft_plan_t plan) {
  if (plan == CUFFT_PLAN_NULL) {
    return;
  }

  if (auto err = ::cufftDestroy(plan)) {
    panic::panic_fmt("cufftDestroy failed: {}", FFTError{err});
  }
}

auto fft_plan(i32 N, c32 I[], c32 O[], i32 batch) -> fft_plan_t {
  auto plan = CUFFT_PLAN_NULL;
  if (auto err = ::cufftPlan1d(&plan, N, CUFFT_C2C, batch)) {
    panic::panic_fmt("cufftPlan1d failed, err={}", FFTError{err});
  }
  return plan;
}

auto fft_plan(i32 N, f32 I[], c32 O[], i32 batch) -> fft_plan_t {
  auto plan = CUFFT_PLAN_NULL;
  if (auto err = ::cufftPlan1d(&plan, N, CUFFT_R2C, batch)) {
    panic::panic_fmt("cufftPlan1d failed, err={}", FFTError{err});
  }
  return plan;
}

auto fft_plan(i32 N, c32 I[], f32 O[], i32 batch) -> fft_plan_t {
  auto plan = CUFFT_PLAN_NULL;
  if (auto err = ::cufftPlan1d(&plan, N, CUFFT_C2R, batch)) {
    panic::panic_fmt("cufftPlan1d failed, err={}", FFTError{err});
  }
  return plan;
}

void fft_exec(fft_plan_t plan, c32 Ic[], c32 Oc[], int SIGN) {
  const auto idata = reinterpret_cast<cufftComplex*>(Ic);
  const auto odata = reinterpret_cast<cufftComplex*>(Oc);
  if (auto err = ::cufftExecC2C(plan, idata, odata, SIGN)) {
    panic::panic_fmt("cufftExecC2C failed, err={}", FFTError{err});
  }
}

void fft_exec(fft_plan_t plan, f32 Ir[], c32 Oc[], [[maybe_unused]] int SIGN) {
  const auto idata = reinterpret_cast<cufftReal*>(Ir);
  const auto odata = reinterpret_cast<cufftComplex*>(Oc);
  if (auto err = ::cufftExecR2C(plan, idata, odata)) {
    panic::panic_fmt("cufftExecR2C failed, err={}", FFTError{err});
  }
}

void fft_exec(fft_plan_t plan, c32 Ic[], f32 Or[], [[maybe_unused]] int SIGN) {
  const auto idata = reinterpret_cast<cufftComplex*>(Ic);
  const auto odata = reinterpret_cast<cufftReal*>(Or);
  if (auto err = ::cufftExecC2R(plan, idata, odata)) {
    panic::panic_fmt("cufftExecC2R failed, err={}", FFTError{err});
  }
}

template <class I, class O>
FFT<I, O>::FFT() noexcept : _plan{-1} {}

template <class I, class O>
FFT<I, O>::FFT(FFT&& other) noexcept : _plan{other._plan} {
  other._plan = -1;
}

template <class I, class O>
FFT<I, O>::~FFT() {
  cuda::fft_drop(_plan);
}

template <class I, class O>
auto FFT<I, O>::operator=(FFT&& other) noexcept -> FFT& {
  if (this == &other) return *this;
  mem::swap(_plan, other._plan);
  return *this;
}

template <class I, class O>
auto FFT<I, O>::create(u32 len, u32 batch) -> FFT {
  auto res = FFT{};
  res._len = len;
  res._batch = batch;
  res._plan = cuda::fft_plan(len, (I*)nullptr, (O*)nullptr, batch);
  return res;
}

template <class I, class O>
void FFT<I, O>::exec(const I idata[], O odata[], int DIR) {
  return cuda::fft_exec(_plan, const_cast<I*>(idata), odata, DIR);
}

template <class I, class O>
void FFT<I, O>::operator()(math::NdSlice<I, 1> i, math::NdSlice<O, 1> o, int DIR) {
  panic::expect(_batch == 1, "batch size must be 1 for 1D slice");

  const auto ilen = trait::same_<O, c32> ? _len : _len / 2 + 1;
  const auto olen = trait::same_<I, c32> ? _len : _len / 2 + 1;
  panic::expect(i._dims.x == ilen, "in.shape(=`{}`) not match plan(=`{}`)", i._dims, ilen);
  panic::expect(o._dims.x == olen, "out.shape(=`{}`) not match plan(=`{}`)", o._dims, olen);
  return cuda::fft_exec(_plan, i._data, o._data, DIR);
}

template <class I, class O>
void FFT<I, O>::operator()(math::NdSlice<I, 2> i, math::NdSlice<O, 2> o, int DIR) {
  const auto ilen = trait::same_<O, c32> ? _len : _len / 2 + 1;
  const auto olen = trait::same_<I, c32> ? _len : _len / 2 + 1;
  const auto idim = math::vec2u{ilen, _batch};
  const auto odim = math::vec2u{olen, _batch};

  panic::expect(i._dims == idim, "in.shape(=`{}`) not match plan(=`{}`)", i._dims, idim);
  panic::expect(o._dims == odim, "out.shape(=`{}`) not match plan(=`{}`)", o._dims, odim);
  return cuda::fft_exec(_plan, i._data, o._data, DIR);
}

template struct FFT<c32, c32>;
template struct FFT<f32, c32>;
template struct FFT<c32, f32>;

}  // namespace sfc::cuda
