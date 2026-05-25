#include <cuda.h>
#include <cufft.h>

#include "sfc/core.h"
#include "sfc/cuda/fft.h"
#include "sfc/cuda/error.h"
#include "sfc/cuda/stream.h"

namespace sfc::cuda {

using fft_plan_t = int;

struct FFTError {
  int _code;

 public:
  auto to_str() const -> cstr_t {
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

template <class T>
auto fft_cast(T* p) -> T* {
  return p;
}

auto fft_cast(c32* p) -> cufftComplex* {
  return ptr::cast_mut<cufftComplex>(p);
}

void fft_drop(fft_plan_t plan) {
  if (plan == CUFFT_PLAN_NULL) {
    return;
  }

  if (auto err = ::cufftDestroy(plan)) {
    panic::panic_fmt("cufftDestroy failed: {}", FFTError{err});
  }
}

template <class I, class O>
auto fft_plan(u32 N, u32 batch) -> fft_plan_t {
  const auto nx = static_cast<int>(N);
  const auto ny = static_cast<int>(batch);

  auto plan = fft_plan_t{CUFFT_PLAN_NULL};

  auto err = CUFFT_SUCCESS;
  if constexpr (trait::same_<I, c32> && trait::same_<O, c32>) {
    err = ::cufftPlan1d(&plan, nx, CUFFT_C2C, ny);
  } else if constexpr (trait::same_<I, f32> && trait::same_<O, c32>) {
    err = ::cufftPlan1d(&plan, nx, CUFFT_R2C, ny);
  } else if constexpr (trait::same_<I, c32> && trait::same_<O, f32>) {
    err = ::cufftPlan1d(&plan, nx, CUFFT_C2R, ny);
  } else {
    static_assert(false, "unsupported type combination");
  }

  if (err != CUFFT_SUCCESS) {
    panic::panic_fmt("cufftPlan1d failed, err={}", FFTError{err});
  }
  return plan;
}

template <class I, class O>
void fft_exec(fft_plan_t plan, const I in[], O out[], int SIGN) {
  const auto idata = cuda::fft_cast(const_cast<I*>(in));
  const auto odata = cuda::fft_cast(out);

  auto err = CUFFT_SUCCESS;
  if constexpr (sizeof(I) == sizeof(O)) {
    err = ::cufftExecC2C(plan, idata, odata, SIGN);
  } else if constexpr (sizeof(I) < sizeof(O)) {
    err = ::cufftExecR2C(plan, idata, odata);
  } else if constexpr (sizeof(I) > sizeof(O)) {
    err = ::cufftExecC2R(plan, idata, odata);
  } else {
    static_assert(false, "unsupported type combination");
  }

  if (err != CUFFT_SUCCESS) {
    panic::panic_fmt("cufftExecC2C failed, err={}", FFTError{err});
  }
}

template <class I, class O>
FFT<I, O>::FFT() noexcept : _plan{-1} {}

template <class I, class O>
FFT<I, O>::~FFT() {
  cuda::fft_drop(_plan);
}

template <class I, class O>
FFT<I, O>::FFT(FFT&& other) noexcept : _len{other._len}, _batch{other._batch}, _plan{other._plan} {
  other._len = 0;
  other._batch = 0;
  other._plan = -1;
}

template <class I, class O>
auto FFT<I, O>::operator=(FFT&& other) noexcept -> FFT& {
  if (this == &other) return *this;
  mem::swap(_len, other._len);
  mem::swap(_plan, other._plan);
  mem::swap(_batch, other._batch);
  return *this;
}

template <class I, class O>
auto FFT<I, O>::create(u32 len, u32 batch) -> FFT {
  auto res = FFT{};
  res._len = len;
  res._batch = batch;
  res._plan = cuda::fft_plan<I, O>(len, batch);
  return res;
}

template <class I, class O>
void FFT<I, O>::exec(const I in[], O out[], int DIR) {
  cuda::fft_exec(_plan, in, out, DIR);
}

template class FFT<f32, c32>;
template class FFT<c32, f32>;
template class FFT<c32, c32>;

}  // namespace sfc::cuda
