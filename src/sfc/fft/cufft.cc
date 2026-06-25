#include <cufft.h>

#include "sfc/math.h"
#include "sfc/cuda/stream.h"
#include "sfc/fft/cufft.h"

#if !defined(__INTELLISENSE__) && !defined(__clang_analyzer__)
#define CHECK_RET(func, ...) cufft::check_ret(func(__VA_ARGS__), #func)
#else
#define CHECK_RET(func, ...) func(__VA_ARGS__)
#endif

namespace sfc::fft {

using panic::SourceLoc;

namespace cufft {

static auto error_name(cufftResult code) -> cstr_t {
  switch (code) {
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

inline void check_ret(cufftResult ret, const char* func, SourceLoc loc = SourceLoc::current()) {
  if (ret == 0) {
    return;
  }

  const auto err_name = cufft::error_name(ret);
  panic::panic_fmt(fmt::Args{"cufft API(`{}`) called failed, err=`{}`", func, err_name}, loc);
}

template <class T>
static auto fft_cast(T* p) {
  if constexpr (trait::same_<T, c32>) {
    return ptr::cast<cufftComplex>(p);
  } else if constexpr (trait::same_<T, f32>) {
    return p;
  }
}

static void fft_drop(cufftHandle plan) {
  if (plan == CUFFT_PLAN_NULL) {
    return;
  }

  CHECK_RET(cufftDestroy, plan);
}

template <class I, class O>
static auto fft_plan(u32 N, u32 batch) -> cufftHandle {
  const auto nx = num::saturating_cast<int>(N);
  const auto ny = num::saturating_cast<int>(batch);

  auto plan = cufftHandle{CUFFT_PLAN_NULL};

  if constexpr (trait::same_<I, c32> && trait::same_<O, c32>) {
    CHECK_RET(cufftPlan1d, &plan, nx, CUFFT_C2C, ny);
  } else if constexpr (trait::same_<I, f32> && trait::same_<O, c32>) {
    CHECK_RET(cufftPlan1d, &plan, nx, CUFFT_R2C, ny);
  } else if constexpr (trait::same_<I, c32> && trait::same_<O, f32>) {
    CHECK_RET(cufftPlan1d, &plan, nx, CUFFT_C2R, ny);
  } else {
    static_assert(false, "unsupported type combination");
  }

  return plan;
}

template <class I, class O>
static void fft_exec(cufftHandle plan, const I in[], O out[], int SIGN) {
  const auto idata = cufft::fft_cast(const_cast<I*>(in));
  const auto odata = cufft::fft_cast(out);

  if constexpr (sizeof(I) == sizeof(O)) {
    CHECK_RET(cufftExecC2C, plan, idata, odata, SIGN);
  } else if constexpr (sizeof(I) < sizeof(O)) {
    CHECK_RET(cufftExecR2C, plan, idata, odata);
  } else if constexpr (sizeof(I) > sizeof(O)) {
    CHECK_RET(cufftExecC2R, plan, idata, odata);
  } else {
    static_assert(false, "unsupported type combination");
  }
}
}  // namespace cufft

template <class I, class O>
CUFFT<I, O>::CUFFT() noexcept : _plan{-1} {}

template <class I, class O>
CUFFT<I, O>::~CUFFT() {
  cufft::fft_drop(_plan);
}

template <class I, class O>
CUFFT<I, O>::CUFFT(CUFFT&& other) noexcept : _len{other._len}, _batch{other._batch}, _plan{other._plan} {
  other._len = 0;
  other._batch = 0;
  other._plan = -1;
}

template <class I, class O>
auto CUFFT<I, O>::operator=(CUFFT&& other) noexcept -> CUFFT& {
  if (this == &other) return *this;
  mem::swap(_len, other._len);
  mem::swap(_plan, other._plan);
  mem::swap(_batch, other._batch);
  return *this;
}

template <class I, class O>
auto CUFFT<I, O>::create(u32 len, u32 batch) -> CUFFT {
  auto res = CUFFT{};
  res._len = len;
  res._batch = batch;
  res._plan = cufft::fft_plan<I, O>(len, batch);
  return res;
}

template <class I, class O>
auto CUFFT<I, O>::in_len() const -> usize {
  const auto half_len = _len / 2 + 1;
  return sizeof(I) <= sizeof(O) ? _len : half_len;
}

template <class I, class O>
auto CUFFT<I, O>::out_len() const -> usize {
  const auto half_len = _len / 2 + 1;
  return sizeof(O) <= sizeof(I) ? _len : half_len;
}

template <class I, class O>
void CUFFT<I, O>::exec(const I in[], O out[], int DIR) {
  cufft::fft_exec(_plan, in, out, DIR);
}

template <class I, class O>
void CUFFT<I, O>::operator()(math::NdArray<I, 1>& in, math::NdArray<O, 1>& out, int DIR) {
  const auto ilen = this->in_len();
  const auto olen = this->out_len();
  sfc::assert_(in.shape()[0] == ilen, "in.shape({}) != {}", in.shape(), ilen);
  sfc::assert_(out.shape()[0] == olen, "out.shape({}) != {}", out.shape(), olen);

  this->exec(in.data(), out.data(), DIR);
}

template <class I, class O>
void CUFFT<I, O>::operator()(math::NdArray<I, 2>& in, math::NdArray<O, 2>& out, int DIR) {
  const auto ilen = this->in_len();
  const auto olen = this->out_len();
  sfc::assert_(in.shape()[0] == _batch, "in.shape({}) != batch({})", in.shape(), _batch);
  sfc::assert_(in.shape()[1] == ilen, "in.shape({}) != {}", in.shape(), ilen);

  sfc::assert_(out.shape()[0] == _batch, "out.shape({}) != batch({})", out.shape(), _batch);
  sfc::assert_(out.shape()[1] == olen, "out.shape({}) != {}", out.shape(), olen);

  this->exec(in.data(), out.data(), DIR);
}

template class CUFFT<f32, c32>;
template class CUFFT<c32, f32>;
template class CUFFT<c32, c32>;

}  // namespace sfc::fft
