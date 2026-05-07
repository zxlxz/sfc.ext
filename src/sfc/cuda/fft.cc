#include <cuda.h>
#include <cufft.h>

#include "sfc/cuda/fft.h"
#include "sfc/cuda/error.h"
#include "sfc/cuda/stream.h"

#define CUFFT_TRY(expr)              \
  if (auto err = (expr)) {           \
    throw Error{cuda::fft_err(err)}; \
  }

namespace sfc::cuda {

static auto fft_err(cufftResult res) -> CUresult {
  switch (res) {
    case CUFFT_SUCCESS:        return CUDA_SUCCESS;
    case CUFFT_INVALID_PLAN:   return CUDA_ERROR_INVALID_VALUE;
    case CUFFT_ALLOC_FAILED:   return CUDA_ERROR_OUT_OF_MEMORY;
    case CUFFT_INVALID_TYPE:   return CUDA_ERROR_INVALID_VALUE;
    case CUFFT_INVALID_VALUE:  return CUDA_ERROR_INVALID_VALUE;
    case CUFFT_INTERNAL_ERROR: return CUDA_ERROR_UNKNOWN;
    case CUFFT_EXEC_FAILED:    return CUDA_ERROR_LAUNCH_FAILED;
    case CUFFT_SETUP_FAILED:   return CUDA_ERROR_INVALID_VALUE;
    case CUFFT_INVALID_SIZE:   return CUDA_ERROR_INVALID_VALUE;
    case CUFFT_UNALIGNED_DATA: return CUDA_ERROR_INVALID_VALUE;
    case CUFFT_INVALID_DEVICE: return CUDA_ERROR_INVALID_DEVICE;
    case CUFFT_NO_WORKSPACE:   return CUDA_ERROR_OUT_OF_MEMORY;
    default:                   return CUDA_ERROR_UNKNOWN;
  }
}

void fft_drop(fft_plan_t plan) {
  if (plan == CUFFT_PLAN_NULL) {
    return;
  }

  CUFFT_TRY(::cufftDestroy(plan));
}

auto fft_plan_c2c(u32 nx, u32 batch) -> fft_plan_t {
  const auto fft_nx = static_cast<int>(nx);
  const auto fft_batch = static_cast<int>(batch);

  auto plan = CUFFT_PLAN_NULL;
  CUFFT_TRY(::cufftPlan1d(&plan, fft_nx, CUFFT_C2C, fft_batch));
  return plan;
}

auto fft_plan_r2c(u32 nx, u32 batch) -> fft_plan_t {
  const auto fft_nx = static_cast<int>(nx);
  const auto fft_batch = static_cast<int>(batch);

  auto plan = CUFFT_PLAN_NULL;
  CUFFT_TRY(::cufftPlan1d(&plan, fft_nx, CUFFT_R2C, fft_batch));
  return plan;
}

auto fft_plan_c2r(u32 nx, u32 batch) -> fft_plan_t {
  const auto fft_nx = static_cast<int>(nx);
  const auto fft_batch = static_cast<int>(batch);

  auto plan = CUFFT_PLAN_NULL;
  CUFFT_TRY(::cufftPlan1d(&plan, fft_nx, CUFFT_C2R, fft_batch));
  return plan;
}

void fft_exec_c2c(fft_plan_t plan, const c32* in, c32* out, int direction) {
  const auto fft_in = reinterpret_cast<cufftComplex*>(const_cast<c32*>(in));
  const auto fft_out = reinterpret_cast<cufftComplex*>(out);

  CUFFT_TRY(::cufftExecC2C(plan, fft_in, fft_out, direction));
}

void fft_exec_r2c(fft_plan_t plan, const f32* in, c32* out) {
  const auto fft_in = reinterpret_cast<cufftReal*>(const_cast<f32*>(in));
  const auto fft_out = reinterpret_cast<cufftComplex*>(out);
  CUFFT_TRY(::cufftExecR2C(plan, fft_in, fft_out));
}

void fft_exec_c2r(fft_plan_t plan, const c32* in, f32* out) {
  const auto fft_in = reinterpret_cast<cufftComplex*>(const_cast<c32*>(in));
  const auto fft_out = reinterpret_cast<cufftReal*>(out);
  CUFFT_TRY(::cufftExecC2R(plan, fft_in, fft_out));
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
