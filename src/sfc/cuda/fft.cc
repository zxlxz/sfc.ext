#include <cufft.h>

#include "sfc/core.h"
#include "sfc/cuda/stream.h"
#include "sfc/cuda/fft.h"
#include "sfc/math/vec.h"

namespace sfc::cuda::fft {

auto to_str(Error e) -> cstr_t {
  switch (e) {
    case Error::Success:           return "cuda::fft::Success";
    case Error::InvalidPlan:       return "cuda::fft::InvalidPlan";
    case Error::AllocFailed:       return "cuda::fft::AllocFailed";
    case Error::InvalidType:       return "cuda::fft::InvalidType";
    case Error::InvalidValue:      return "cuda::fft::InvalidValue";
    case Error::InternalError:     return "cuda::fft::InternalError";
    case Error::ExecFailed:        return "cuda::fft::ExecFailed";
    case Error::SetupFailed:       return "cuda::fft::SetupFailed";
    case Error::InvalidSize:       return "cuda::fft::InvalidSize";
    case Error::UnalignedData:     return "cuda::fft::UnalignedData";
    case Error::InvalidDevice:     return "cuda::fft::InvalidDevice";
    case Error::NoWorkspace:       return "cuda::fft::NoWorkspace";
    case Error::NotImplemented:    return "cuda::fft::NotImplemented";
    case Error::NotSupported:      return "cuda::fft::NotSupported";
    case Error::MissingDependency: return "cuda::fft::MissingDependency";
    case Error::NVRTCFailure:      return "cuda::fft::NVRTCFailure";
    case Error::NVJITLinkFailure:  return "cuda::fft::NVJITLinkFailure";
    case Error::NVSHMEMFailure:    return "cuda::fft::NVSHMEMFailure";
    default:                       return "cuda::fft::Unknown";
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

static auto fft_plan(u32 nx, u32 batch, cufftType type) -> Result<cufftHandle> {
  auto plan = cufftHandle{};
  if (auto err = cufftPlan1d(&plan, int(nx), type, int(batch))) {
    return Error(err);
  }
  return Ok(plan);
}

static auto fft_drop(cufftHandle plan) -> Result<> {
  if (plan == 0) return Ok{};
  if (auto err = cufftDestroy(plan)) {
    return Error(err);
  }
  return Ok{};
}

static auto fft_exec(cufftHandle plan, c32* in, c32* out, int direction) -> Result<> {
  const auto idata = fft_cast(in);
  const auto odata = fft_cast(out);
  if (auto err = cufftExecC2C(plan, idata, odata, direction); err != CUFFT_SUCCESS) {
    return Error(err);
  }
  return Ok{};
}

static auto fft_exec(cufftHandle plan, f32* in, c32* out, [[maybe_unused]] int direction) -> Result<> {
  const auto idata = fft_cast(in);
  const auto odata = fft_cast(out);
  if (auto err = cufftExecR2C(plan, idata, odata)) {
    return Error(err);
  }
  return Ok{};
}

static auto fft_exec(cufftHandle plan, c32* in, f32* out, [[maybe_unused]] int direction) -> Result<> {
  const auto idata = fft_cast(in);
  const auto odata = fft_cast(out);
  if (auto err = cufftExecC2R(plan, idata, odata)) {
    return Error(err);
  }
  return Ok{};
}

FFT::FFT() noexcept : _plan{0} {}

FFT::~FFT() {
  fft_drop(_plan).unwrap();
}

FFT::FFT(FFT&& other) noexcept
    : _len{mem::take(other._len)}, _batch{mem::take(other._batch)}, _plan{mem::take(other._plan)} {}

auto FFT::operator=(FFT&& other) noexcept -> FFT& {
  if (this == &other) return *this;
  mem::swap(_len, other._len);
  mem::swap(_plan, other._plan);
  mem::swap(_batch, other._batch);
  return *this;
}

auto FFT::new_(u32 len, u32 batch) -> FFT {
  const auto type = fft_type<c32, c32>();
  const auto plan = fft_plan(len, batch, type).unwrap();

  auto res = FFT{};
  res._len = len;
  res._batch = batch;
  res._plan = plan;
  return res;
}

auto FFT::len() const -> usize {
  return _len;
}

auto FFT::batch() const -> usize {
  return _batch;
}

auto FFT::fft(math::NdSlice<c32, 1> in, math::NdSlice<c32, 1> out) -> Result<> {
  const auto [ilen] = in._shape;
  const auto [olen] = out._shape;

  sfc::assert_(in.is_contiguous(), "FFT::fft: in is not contiguous");
  sfc::assert_(out.is_contiguous(), "FFT::fft: out is not contiguous");
  sfc::assert_(ilen == _len, "FFT::fft: in.shape({}) not match fft.len(={})", ilen, _len);
  sfc::assert_(olen == _len, "FFT::fft: out.shape({}) not match fft.len(={})", olen, _len);
  sfc::assert_(_batch == 1, "FFT::fft: batch({}) != 1", _batch);

  auto ret = fft_exec(_plan, in._data, out._data, CUFFT_FORWARD);
  return ret;
}

auto FFT::ifft(math::NdSlice<c32, 1> in, math::NdSlice<c32, 1> out) -> Result<> {
  const auto [ilen] = in._shape;
  const auto [olen] = out._shape;

  sfc::assert_(in.is_contiguous(), "FFT::ifft: in is not contiguous");
  sfc::assert_(out.is_contiguous(), "FFT::ifft: out is not contiguous");

  sfc::assert_(ilen == _len, "FFT::ifft: in.shape({}) not match fft.len(={})", ilen, _len);
  sfc::assert_(olen == _len, "FFT::ifft: out.shape({}) not match fft.len(={})", olen, _len);
  sfc::assert_(_batch == 1, "FFT::ifft: batch({}) != 1", _batch);

  auto ret = fft_exec(_plan, in._data, out._data, CUFFT_INVERSE);
  return ret;
}

auto FFT::fft(math::NdSlice<c32, 2> in, math::NdSlice<c32, 2> out) -> Result<> {
  const auto [ibatch, ilen] = in._shape;
  const auto [obatch, olen] = out._shape;

  sfc::assert_(in.is_contiguous(), "FFT::fft: in is not contiguous");
  sfc::assert_(out.is_contiguous(), "FFT::fft: out is not contiguous");
  sfc::assert_(ilen == _len, "FFT::fft: in.shape({}) not match fft.len(={})", ilen, _len);
  sfc::assert_(olen == _len, "FFT::fft: out.shape({}) not match fft.len(={})", olen, _len);
  sfc::assert_(ibatch == obatch, "FFT::fft: in.batch({}) not match out.batch(={})", ibatch, obatch);
  sfc::assert_(ibatch % _batch == 0, "FFT::fft: in.batch({}) not multiple of batch(={})", ibatch, _batch);
  sfc::assert_(obatch % _batch == 0, "FFT::fft: out.batch({}) not multiple of batch(={})", obatch, _batch);

  for (auto i = 0U; i < ibatch; i += _batch) {
    auto s = in[i];
    auto d = out[i];
    _TRY(fft_exec(_plan, s._data, d._data, CUFFT_FORWARD));
  }
  return Ok{};
}

auto FFT::ifft(math::NdSlice<c32, 2> in, math::NdSlice<c32, 2> out) -> Result<> {
  const auto [ibatch, ilen] = in._shape;
  const auto [obatch, olen] = out._shape;

  sfc::assert_(in.is_contiguous(), "FFT::ifft: in is not contiguous");
  sfc::assert_(out.is_contiguous(), "FFT::ifft: out is not contiguous");
  sfc::assert_(ibatch == obatch, "FFT::ifft: in.batch({}) not match out.batch(={})", ibatch, obatch);
  sfc::assert_(ibatch % _batch == 0, "FFT::ifft: in.batch({}) not multiple of batch(={})", ibatch, _batch);
  sfc::assert_(obatch % _batch == 0, "FFT::ifft: out.batch({}) not multiple of batch(={})", obatch, _batch);

  for (auto i = 0U; i < ibatch; i += _batch) {
    auto s = in[i];
    auto d = out[i];
    _TRY(fft_exec(_plan, s._data, d._data, CUFFT_INVERSE));
  }
  return Ok{};
}

RFFT::RFFT() noexcept {}

RFFT::~RFFT() {
  fft_drop(_plan_r2c).unwrap();
  fft_drop(_plan_c2r).unwrap();
}

RFFT::RFFT(RFFT&& other) noexcept
    : _len{mem::take(other._len)}
    , _batch{mem::take(other._batch)}
    , _plan_r2c{mem::take(other._plan_r2c)}
    , _plan_c2r{mem::take(other._plan_c2r)} {}

RFFT& RFFT::operator=(RFFT&& other) noexcept {
  if (this == &other) return *this;
  mem::swap(_len, other._len);
  mem::swap(_batch, other._batch);
  mem::swap(_plan_r2c, other._plan_r2c);
  mem::swap(_plan_c2r, other._plan_c2r);
  return *this;
}

auto RFFT::create(u32 len, u32 batch) -> RFFT {
  const auto plan_r2c = fft_plan(len, batch, fft_type<f32, c32>()).unwrap();
  const auto plan_c2r = fft_plan(len, batch, fft_type<c32, f32>()).unwrap();

  auto res = RFFT{};
  res._len = len;
  res._batch = batch;
  res._plan_r2c = plan_r2c;
  res._plan_c2r = plan_c2r;
  return res;
}

auto RFFT::fft(math::NdSlice<f32, 1> in, math::NdSlice<c32, 1> out) -> Result<> {
  const auto full_len = _len;
  const auto half_len = _len / 2 + 1;

  const auto [src_len] = in._shape;
  const auto [dst_len] = out._shape;

  sfc::assert_(in.is_contiguous(), "RFFT::fft: in is not contiguous");
  sfc::assert_(out.is_contiguous(), "RFFT::fft: out is not contiguous");
  sfc::assert_(src_len == full_len, "RFFT::fft: in.shape({}) not match len(={})", src_len, _len);
  sfc::assert_(dst_len == half_len, "RFFT::fft: out.shape({}) not match len(={}/2+1)", dst_len, _len);
  sfc::assert_(_batch == 1, "RFFT::fft: batch({}) != 1", _batch);

  auto ret = fft_exec(_plan_r2c, in._data, out._data, CUFFT_FORWARD);
  return ret;
}

auto RFFT::ifft(math::NdSlice<c32, 1> in, math::NdSlice<f32, 1> out) -> Result<> {
  const auto full_len = _len;
  const auto half_len = _len / 2 + 1;

  const auto [ilen] = in._shape;
  const auto [olen] = out._shape;

  sfc::assert_(in.is_contiguous(), "RFFT::ifft: in is not contiguous");
  sfc::assert_(out.is_contiguous(), "RFFT::ifft: out is not contiguous");
  sfc::assert_(ilen == half_len, "RFFT::ifft: in.shape({}) not match fft.len(={})/2+1", ilen, _len);
  sfc::assert_(olen == full_len, "RFFT::ifft: out.shape({}) not match fft.len(={})", olen, _len);
  sfc::assert_(_batch == 1, "RFFT::ifft: batch({}) != 1", _batch);

  auto ret = fft_exec(_plan_c2r, in._data, out._data, CUFFT_INVERSE);
  return ret;
}

auto RFFT::fft(math::NdSlice<f32, 2> in, math::NdSlice<c32, 2> out) -> Result<> {
  const auto full_len = _len;
  const auto half_len = _len / 2 + 1;

  const auto [ibatch, ilen] = in._shape;
  const auto [obatch, olen] = out._shape;

  sfc::assert_(in.is_contiguous(), "RFFT::fft: in is not contiguous");
  sfc::assert_(out.is_contiguous(), "RFFT::fft: out is not contiguous");
  sfc::assert_(ilen == full_len, "RFFT::fft: in.shape({}) not match fft.len(={})", in._shape, _len);
  sfc::assert_(olen == half_len, "RFFT::fft: out.shape({}) not match fft.len(={})/2+)", out._shape, _len);
  sfc::assert_(ibatch == obatch, "RFFT::fft: in.batch({}) not match out.batch(={})", ibatch, obatch);
  sfc::assert_(ibatch % _batch == 0, "RFFT::fft: in.batch({}) not multiple of batch(={})", ibatch, _batch);
  sfc::assert_(obatch % _batch == 0, "RFFT::fft: out.batch({}) not multiple of batch(={})", obatch, _batch);

  for (auto i = 0U; i < ibatch; i += _batch) {
    auto s = in[i];
    auto d = out[i];
    _TRY(fft_exec(_plan_r2c, s._data, d._data, CUFFT_FORWARD));
  }
  return Ok{};
}

auto RFFT::ifft(math::NdSlice<c32, 2> in, math::NdSlice<f32, 2> out) -> Result<> {
  const auto full_len = _len;
  const auto half_len = _len / 2 + 1;

  const auto [ibatch, ilen] = in._shape;
  const auto [obatch, olen] = out._shape;

  sfc::assert_(in.is_contiguous(), "RFFT::ifft: in is not contiguous");
  sfc::assert_(out.is_contiguous(), "RFFT::ifft: out is not contiguous");
  sfc::assert_(ilen == half_len, "RFFT::ifft: in.shape({}) not match fft.len(={})/2+1", ilen, _len);
  sfc::assert_(olen == full_len, "RFFT::ifft: out.shape({}) not match fft.len(={})", olen, _len);
  sfc::assert_(ibatch == obatch, "RFFT::ifft: in.batch({}) not match out.batch(={})", ibatch, obatch);
  sfc::assert_(ibatch % _batch == 0, "RFFT::ifft: in.batch({}) not multiple of batch(={})", ibatch, _batch);
  sfc::assert_(obatch % _batch == 0, "RFFT::ifft: out.batch({}) not multiple of batch(={})", obatch, _batch);

  for (auto i = 0U; i < ibatch; i += _batch) {
    auto s = in[i];
    auto d = out[i];
    _TRY(fft_exec(_plan_c2r, s._data, d._data, CUFFT_INVERSE));
  }
  return Ok{};
}

}  // namespace sfc::cuda::fft
