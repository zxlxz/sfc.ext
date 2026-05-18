#include <cuda.h>
#include <cufft.h>

#include "sfc/core.h"
#include "sfc/cuda/fft.h"
#include "sfc/cuda/error.h"
#include "sfc/cuda/stream.h"

namespace sfc::cuda {

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

auto fft_plan_c2c(u32 nx, u32 batch) -> fft_plan_t {
  const auto fft_nx = static_cast<int>(nx);
  const auto fft_batch = static_cast<int>(batch);

  auto plan = CUFFT_PLAN_NULL;
  if (auto err = ::cufftPlan1d(&plan, fft_nx, CUFFT_C2C, fft_batch)) {
    panic::panic_fmt("cufftPlan1d failed, err={}", FFTError{err});
  }
  return plan;
}

auto fft_plan_r2c(u32 nx, u32 batch) -> fft_plan_t {
  const auto fft_nx = static_cast<int>(nx);
  const auto fft_batch = static_cast<int>(batch);

  auto plan = CUFFT_PLAN_NULL;
  if (auto err = ::cufftPlan1d(&plan, fft_nx, CUFFT_R2C, fft_batch)) {
    panic::panic_fmt("cufftPlan1d failed, err={}", FFTError{err});
  }
  return plan;
}

auto fft_plan_c2r(u32 nx, u32 batch) -> fft_plan_t {
  const auto fft_nx = static_cast<int>(nx);
  const auto fft_batch = static_cast<int>(batch);

  auto plan = CUFFT_PLAN_NULL;
  if (auto err = ::cufftPlan1d(&plan, fft_nx, CUFFT_C2R, fft_batch)) {
    panic::panic_fmt("cufftPlan1d failed, err={}", FFTError{err});
  }
  return plan;
}

void fft_exec_c2c(fft_plan_t plan, const c32* in, c32* out, int direction) {
  const auto fft_in = reinterpret_cast<cufftComplex*>(const_cast<c32*>(in));
  const auto fft_out = reinterpret_cast<cufftComplex*>(out);

  if (auto err = ::cufftExecC2C(plan, fft_in, fft_out, direction)) {
    panic::panic_fmt("cufftExecC2C failed, err={}", FFTError{err});
  }
}

void fft_exec_r2c(fft_plan_t plan, const f32* in, c32* out) {
  const auto fft_in = reinterpret_cast<cufftReal*>(const_cast<f32*>(in));
  const auto fft_out = reinterpret_cast<cufftComplex*>(out);
  if (auto err = ::cufftExecR2C(plan, fft_in, fft_out)) {
    panic::panic_fmt("cufftExecR2C failed, err={}", FFTError{err});
  }
}

void fft_exec_c2r(fft_plan_t plan, const c32* in, f32* out) {
  const auto fft_in = reinterpret_cast<cufftComplex*>(const_cast<c32*>(in));
  const auto fft_out = reinterpret_cast<cufftReal*>(out);
  if (auto err = ::cufftExecC2R(plan, fft_in, fft_out)) {
    panic::panic_fmt("cufftExecC2R failed, err={}", FFTError{err});
  }
}

template <class I, class O>
FFT<I, O>::FFT() : _len{0}, _batch{0}, _plan{-1} {}

template <class I, class O>
FFT<I, O>::~FFT() {
  if (_plan == -1) return;
  cuda::fft_drop(_plan);
}

template <class I, class O>
FFT<I, O>::FFT(FFT&& other) noexcept : _len{other._len}, _batch{other._batch}, _plan{other._plan} {
  other._len = 0;
  other._batch = 0;
  other._plan = -1;
}

template <class I, class O>
FFT<I, O>& FFT<I, O>::operator=(FFT&& other) noexcept {
  if (this == &other) return;
  mem::swap(_len, other._len);
  mem::swap(_batch, other._batch);
  mem::swap(_plan, other._plan);
  return *this;
}

template <class I, class O>
auto FFT<I, O>::create(u32 len, u32 batch) -> FFT<I, O> {
  auto plan = FFT<I, O>{};
  plan._len = len;
  plan._batch = batch;

  if constexpr (trait::same_<I, c32> && trait::same_<O, c32>) {
    plan._plan = cuda::fft_plan_c2c(plan._len, plan._batch);
  } else if constexpr (trait::same_<I, f32> && trait::same_<O, c32>) {
    plan._plan = cuda::fft_plan_r2c(plan._len, plan._batch);
  } else if constexpr (trait::same_<I, c32> && trait::same_<O, f32>) {
    plan._plan = cuda::fft_plan_c2r(plan._len, plan._batch);
  }
  return plan;
}

template <class I, class O>
void FFT<I, O>::exec(const I X[], O Y[], int DIR) {
  if constexpr (trait::same_<I, c32> && trait::same_<O, c32>) {
    cuda::fft_exec_c2c(_plan, X, Y, DIR);
  } else if constexpr (trait::same_<I, f32> && trait::same_<O, c32>) {
    cuda::fft_exec_r2c(_plan, X, Y);
  } else if constexpr (trait::same_<I, c32> && trait::same_<O, f32>) {
    cuda::fft_exec_c2r(_plan, X, Y);
  }
}

}  // namespace sfc::cuda
