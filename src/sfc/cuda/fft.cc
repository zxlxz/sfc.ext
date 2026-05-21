#include <cuda.h>
#include <cufft.h>

#include "sfc/core.h"
#include "sfc/cuda/fft.h"
#include "sfc/cuda/error.h"
#include "sfc/cuda/stream.h"

namespace sfc::cuda {

using fft_plan_t = int;

auto FFTError::to_str() const -> cstr_t {
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

void fft_drop(fft_plan_t plan) {
  if (plan == CUFFT_PLAN_NULL) {
    return;
  }

  if (auto err = ::cufftDestroy(plan)) {
    panic::panic_fmt("cufftDestroy failed: {}", FFTError{err});
  }
}

template <class I, class O>
auto fft_plan(i32 N, I in[], O out[], i32 batch) -> fft_plan_t {
  auto plan = fft_plan_t{CUFFT_PLAN_NULL};

  auto err = CUFFT_SUCCESS;
  if constexpr (trait::same_<I, c32> && trait::same_<O, c32>) {
    err = ::cufftPlan1d(&plan, N, CUFFT_C2C, batch);
  } else if constexpr (trait::same_<I, f32> && trait::same_<O, c32>) {
    err = ::cufftPlan1d(&plan, N, CUFFT_R2C, batch);
  } else if constexpr (trait::same_<I, c32> && trait::same_<O, f32>) {
    err = ::cufftPlan1d(&plan, N, CUFFT_C2R, batch);
  } else {
    static_assert(false, "unsupported type combination");
  }

  if (err != CUFFT_SUCCESS) {
    panic::panic_fmt("cufftPlan1d failed, err={}", FFTError{err});
  }
  return plan;
}

template <class I, class O>
void fft_exec(fft_plan_t plan, I in[], O out[], int SIGN) {
  using R = cufftReal;
  using C = cufftComplex;

  auto err = CUFFT_SUCCESS;

  if constexpr (trait::same_<I, c32> && trait::same_<O, c32>) {
    err = ::cufftExecC2C(plan, reinterpret_cast<C*>(in), reinterpret_cast<C*>(out), SIGN);
  } else if constexpr (trait::same_<I, f32> && trait::same_<O, c32>) {
    err = ::cufftExecR2C(plan, reinterpret_cast<R*>(in), reinterpret_cast<C*>(out));
  } else if constexpr (trait::same_<I, c32> && trait::same_<O, f32>) {
    err = ::cufftExecC2R(plan, reinterpret_cast<C*>(in), reinterpret_cast<R*>(out));
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
void FFT<I, O>::check_size(const u32 (&idim)[2], const u32 (&odim)[2]) const {
  const auto full_len = _len;
  const auto half_len = _len / 2 + 1;
  const auto ilen = trait::same_<O, c32> ? full_len : half_len;
  const auto olen = trait::same_<I, c32> ? full_len : half_len;

  panic::expect(idim[0] == ilen, "idim(={}) not match plan(ilen={})", idim, ilen);
  panic::expect(odim[0] == olen, "odim(={}) not match plan(olen={})", odim, olen);

  panic::expect(idim[1] == _batch, "idim(={}) not match plan(batch={})", idim, _batch);
  panic::expect(odim[1] == _batch, "odim(={}) not match plan(batch={})", odim, _batch);
}

template <class I, class O>
void FFT<I, O>::operator()(math::NdSlice<I, 1> i, math::NdSlice<O, 1> o, int DIR) {
  this->check_size({i._dims.x, 1}, {o._dims.x, 1});
  return cuda::fft_exec(_plan, i._data, o._data, DIR);
}

template <class I, class O>
void FFT<I, O>::operator()(math::NdSlice<I, 2> i, math::NdSlice<O, 2> o, int DIR) {
  this->check_size({i._dims.x, i._dims.y}, {o._dims.x, o._dims.y});
  return cuda::fft_exec(_plan, i._data, o._data, DIR);
}

template struct FFT<c32, c32>;
template struct FFT<f32, c32>;
template struct FFT<c32, f32>;

}  // namespace sfc::cuda
