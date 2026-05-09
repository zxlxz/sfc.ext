#include <cuda.h>
#include <cufft.h>

#include "sfc/core.h"
#include "sfc/cuda/fft.h"
#include "sfc/cuda/error.h"
#include "sfc/cuda/stream.h"

namespace sfc::cuda {

struct CuFFTError {
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
    panic::panic_fmt("cufftDestroy failed: {}", CuFFTError{err});
  }
}

auto fft_plan_c2c(u32 nx, u32 batch) -> fft_plan_t {
  const auto fft_nx = static_cast<int>(nx);
  const auto fft_batch = static_cast<int>(batch);

  auto plan = CUFFT_PLAN_NULL;
  if (auto err = ::cufftPlan1d(&plan, fft_nx, CUFFT_C2C, fft_batch)) {
    panic::panic_fmt("cufftPlan1d failed, err={}", CuFFTError{err});
  }
  return plan;
}

auto fft_plan_r2c(u32 nx, u32 batch) -> fft_plan_t {
  const auto fft_nx = static_cast<int>(nx);
  const auto fft_batch = static_cast<int>(batch);

  auto plan = CUFFT_PLAN_NULL;
  if (auto err = ::cufftPlan1d(&plan, fft_nx, CUFFT_R2C, fft_batch)) {
    panic::panic_fmt("cufftPlan1d failed, err={}", CuFFTError{err});
  }
  return plan;
}

auto fft_plan_c2r(u32 nx, u32 batch) -> fft_plan_t {
  const auto fft_nx = static_cast<int>(nx);
  const auto fft_batch = static_cast<int>(batch);

  auto plan = CUFFT_PLAN_NULL;
  if (auto err = ::cufftPlan1d(&plan, fft_nx, CUFFT_C2R, fft_batch)) {
    panic::panic_fmt("cufftPlan1d failed, err={}", CuFFTError{err});
  }
  return plan;
}

void fft_exec_c2c(fft_plan_t plan, const c32* in, c32* out, int direction) {
  const auto fft_in = reinterpret_cast<cufftComplex*>(const_cast<c32*>(in));
  const auto fft_out = reinterpret_cast<cufftComplex*>(out);

  if (auto err = ::cufftExecC2C(plan, fft_in, fft_out, direction)) {
    panic::panic_fmt("cufftExecC2C failed, err={}", CuFFTError{err});
  }
}

void fft_exec_r2c(fft_plan_t plan, const f32* in, c32* out) {
  const auto fft_in = reinterpret_cast<cufftReal*>(const_cast<f32*>(in));
  const auto fft_out = reinterpret_cast<cufftComplex*>(out);
  if (auto err = ::cufftExecR2C(plan, fft_in, fft_out)) {
    panic::panic_fmt("cufftExecR2C failed, err={}", CuFFTError{err});
  }
}

void fft_exec_c2r(fft_plan_t plan, const c32* in, f32* out) {
  const auto fft_in = reinterpret_cast<cufftComplex*>(const_cast<c32*>(in));
  const auto fft_out = reinterpret_cast<cufftReal*>(out);
  if (auto err = ::cufftExecC2R(plan, fft_in, fft_out)) {
    panic::panic_fmt("cufftExecC2R failed, err={}", CuFFTError{err});
  }
}

template <class I, class O>
FFT<I, O>::FFT() : _plan{-1} {}

template <class I, class O>
FFT<I, O>::~FFT() {
  if (_plan == -1) return;
  cuda::fft_drop(_plan);
}

template <class I, class O>
FFT<I, O>::FFT(FFT&& other) noexcept : _plan{other._plan} {
  other._plan = -1;
}

template <class I, class O>
FFT<I, O>& FFT<I, O>::operator=(FFT&& other) noexcept {
  if (this == &other) return *this;
  cuda::fft_drop(_plan);
  _plan = other._plan;
  other._plan = -1;
  return *this;
}

template <>
auto FFT<c32, c32>::create(u32 nx, u32 batch) -> FFT {
  auto res = FFT{};
  res._plan = cuda::fft_plan_c2c(nx, batch);
  return res;
}

template <>
auto FFT<f32, c32>::create(u32 nx, u32 batch) -> FFT {
  auto res = FFT{};
  res._plan = cuda::fft_plan_r2c(nx, batch);
  return res;
}

template <>
auto FFT<c32, f32>::create(u32 nx, u32 batch) -> FFT {
  auto res = FFT{};
  res._plan = cuda::fft_plan_c2r(nx, batch);
  return res;
}

template <>
void FFT<c32, c32>::operator()(NdSlice<c32, 2> in, NdSlice<c32, 2> out, int dir) {
  cuda::fft_exec_c2c(_plan, in._data, out._data, dir);
}

template <>
void FFT<f32, c32>::operator()(NdSlice<f32, 2> in, NdSlice<c32, 2> out, int dir) {
  cuda::fft_exec_r2c(_plan, in._data, out._data);
}

template <>
void FFT<c32, f32>::operator()(NdSlice<c32, 2> in, NdSlice<f32, 2> out, int dir) {
  cuda::fft_exec_c2r(_plan, in._data, out._data);
}

}  // namespace sfc::cuda
