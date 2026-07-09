#include <cufft.h>

#include "sfc/core.h"
#include "sfc/cuda/stream.h"
#include "sfc/cuda/fft.h"

namespace sfc::cuda {

auto to_str(FFTError e) -> cstr_t {
  const auto err_code = cufftResult(e);
  switch (err_code) {
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

template <class T>
static auto fft_cast(T* p) {
  if constexpr (trait::same_<T, c32>) {
    return ptr::cast<cufftComplex>(p);
  } else if constexpr (trait::same_<T, f32>) {
    return p;
  }
}

template <class I, class O>
static auto fft_type() -> cufftType {
  if constexpr (trait::same_<I, c32> && trait::same_<O, c32>) {
    return CUFFT_C2C;
  } else if constexpr (trait::same_<I, f32> && trait::same_<O, c32>) {
    return CUFFT_R2C;
  } else if constexpr (trait::same_<I, c32> && trait::same_<O, f32>) {
    return CUFFT_C2R;
  } else {
    static_assert(false, "unsupported type combination");
  }
}

static auto fft_plan(u32 nx, u32 batch, cufftType type) -> FFTResult<cufftHandle> {
  auto plan = cufftHandle{CUFFT_PLAN_NULL};
  if (auto err = cufftPlan1d(&plan, int(nx), type, int(batch))) {
    return FFTError(err);
  }
  return Ok(plan);
}

static auto fft_drop(cufftHandle plan) -> FFTResult<> {
  if (plan == CUFFT_PLAN_NULL) return Ok{};
  if (auto err = cufftDestroy(plan)) {
    return FFTError(err);
  }
  return Ok{};
}

static auto fft_exec(cufftHandle plan, c32* in, c32* out, int direction) -> FFTResult<> {
  const auto idata = fft_cast(in);
  const auto odata = fft_cast(out);
  if (auto err = cufftExecC2C(plan, idata, odata, direction)) {
    return FFTError(err);
  }
  return Ok{};
}

static auto fft_exec(cufftHandle plan, f32* in, c32* out, [[maybe_unused]] int direction) -> FFTResult<> {
  const auto idata = fft_cast(in);
  const auto odata = fft_cast(out);
  if (auto err = cufftExecR2C(plan, idata, odata)) {
    return FFTError(err);
  }
  return Ok{};
}

static auto fft_exec(cufftHandle plan, c32* in, f32* out, [[maybe_unused]] int direction) -> FFTResult<> {
  const auto idata = fft_cast(in);
  const auto odata = fft_cast(out);
  if (auto err = cufftExecC2R(plan, idata, odata)) {
    return FFTError(err);
  }
  return Ok{};
}

template <class I, class O>
CUFFT<I, O>::CUFFT() noexcept : _plan{-1} {}

template <class I, class O>
CUFFT<I, O>::~CUFFT() {
  if (_plan == CUFFT_PLAN_NULL) {
    return;
  }
  cuda::fft_drop(_plan).unwrap();
}

template <class I, class O>
CUFFT<I, O>::CUFFT(CUFFT&& other) noexcept : _len{other._len}, _batch{other._batch}, _plan{other._plan} {
  other._len = 0;
  other._batch = 0;
  other._plan = CUFFT_PLAN_NULL;
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
  const auto type = cuda::fft_type<I, O>();
  const auto plan = cuda::fft_plan(len, batch, type).unwrap();

  auto res = CUFFT{};
  res._len = len;
  res._batch = batch;
  res._plan = plan;
  return res;
}

template <class I, class O>
auto CUFFT<I, O>::ilen() const -> usize {
  const auto half_len = _len / 2 + 1;
  return sizeof(I) <= sizeof(O) ? _len : half_len;
}

template <class I, class O>
auto CUFFT<I, O>::olen() const -> usize {
  const auto half_len = _len / 2 + 1;
  return sizeof(O) <= sizeof(I) ? _len : half_len;
}

template <class I, class O>
auto CUFFT<I, O>::exec(const I in[], O out[], int DIR) -> FFTResult<> {
  return cuda::fft_exec(_plan, (I*)in, out, DIR);
}

template <class I, class O>
auto CUFFT<I, O>::operator()(math::NdSlice<I, 1> src, math::NdSlice<O, 1> dst, int DIR) -> FFTResult<> {
  const auto ilen = this->ilen();
  const auto olen = this->olen();
  const auto [src_len] = src._shape;
  const auto [dst_len] = dst._shape;
  sfc::assert_(src.is_contiguous(), "CUFFT::exec: src is not contiguous");
  sfc::assert_(dst.is_contiguous(), "CUFFT::exec: dst is not contiguous");
  sfc::assert_(src_len == ilen, "CUFFT::exec: src.shape({}) != ilen{}", src_len, ilen);
  sfc::assert_(dst_len == olen, "CUFFT::exec: dst.shape({}) != olen{}", dst_len, olen);

  sfc::assert_(_batch == 1, "CUFFT::exec: batch({}) != 1", _batch);
  _TRY(this->exec(src._data, dst._data, DIR));

  return Ok{};
}

template <class I, class O>
auto CUFFT<I, O>::operator()(math::NdSlice<I, 2> src, math::NdSlice<O, 2> dst, int DIR) -> FFTResult<> {
  const auto ilen = this->ilen();
  const auto olen = this->olen();
  const auto [src_batch, src_len] = src._shape;
  const auto [dst_batch, dst_len] = dst._shape;

  sfc::assert_(src.is_contiguous(), "CUFFT::exec: src is not contiguous");
  sfc::assert_(dst.is_contiguous(), "CUFFT::exec: dst is not contiguous");
  sfc::assert_(src_len == ilen, "CUFFT::exec: src.shape({}) not match ilen{}", src._shape, ilen);
  sfc::assert_(dst_len == olen, "CUFFT::exec: dst.shape({}) not match olen{}", dst._shape, olen);
  sfc::assert_(src_batch == dst_batch, "CUFFT::exec: src.batch({}) not match dst.batch({})", src_batch, dst_batch);
  sfc::assert_(src_batch % _batch == 0, "CUFFT::exec: src.batch({}) not multiple of batch({})", src_batch, _batch);

  const auto loop_cnt = src_batch / _batch;
  for (auto i = 0U; i < loop_cnt; ++i) {
    auto src_col = src[i * _batch];
    auto dst_col = dst[i * _batch];
    _TRY(this->exec(src_col._data, dst_col._data, DIR));
  }

  return Ok{};
}

template class CUFFT<f32, c32>;
template class CUFFT<c32, f32>;
template class CUFFT<c32, c32>;

}  // namespace sfc::cuda
