#include <cufft.h>

#include "sfc/ffi/library.h"
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

using plan_t = cufftHandle;

class CUFFT {
  const ffi::Library& _lib;
#define X(f) decltype(f)* _##f = _lib.get_func<decltype(f)*>(#f)
  X(cufftPlan1d);
  X(cufftDestroy);
  X(cufftExecC2C);
  X(cufftExecR2C);
  X(cufftExecC2R);
#undef X

 public:
  CUFFT(const ffi::Library& lib) noexcept : _lib{lib} {}
  ~CUFFT() = default;

  static auto instance() -> CUFFT& {
#ifdef _WIN32
    static auto lib = ffi::Library::load("cufft64_12.dll");
#else
    static auto lib = ffi::Library::load("cufft64");
#endif
    static auto res = CUFFT{lib};
    return res;
  }

 public:
  template <class T>
  static auto cast(T* p) {
    if constexpr (trait::same_<T, c32>) {
      return ptr::cast<cufftComplex>(p);
    } else if constexpr (trait::same_<T, f32>) {
      return p;
    }
  }

  template <class I, class O>
  static auto type() -> cufftType {
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

  template <class I, class O>
  auto plan(u32 nx, u32 batch) -> Result<plan_t> {
    const auto type = CUFFT::type<I, O>();
    auto plan = plan_t{};
    if (auto err = _cufftPlan1d(&plan, int(nx), type, int(batch))) {
      return Error(err);
    }
    return Ok(plan);
  }

  auto drop(plan_t p) -> Result<> {
    if (p == 0) {
      return Ok{};
    }
    const auto err = _cufftDestroy(p);
    if (err != CUFFT_SUCCESS) {
      return Error(err);
    }
    return Ok{};
  }

  auto exec(cufftHandle plan, const auto in[], auto out[], int dir) -> Result<> {
    const auto idata = CUFFT::cast(ptr::cast_mut(in));
    const auto odata = CUFFT::cast(out);

    auto err = CUFFT_SUCCESS;
    if constexpr (requires { _cufftExecC2C(plan, idata, odata, dir); }) {
      err = _cufftExecC2C(plan, idata, odata, dir);
    } else if constexpr (requires { _cufftExecR2C(plan, idata, odata); }) {
      err = _cufftExecR2C(plan, idata, odata);
    } else if constexpr (requires { _cufftExecC2R(plan, idata, odata); }) {
      err = _cufftExecC2R(plan, idata, odata);
    }

    if (err != CUFFT_SUCCESS) {
      return Error(err);
    }
    return Ok{};
  }
};

FFT::FFT() noexcept : _plan{0} {}

FFT::~FFT() {
  auto& cufft = CUFFT::instance();
  cufft.drop(_plan).unwrap();
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
  auto& cufft = CUFFT::instance();
  const auto plan = cufft.plan<c32, c32>(len, batch).unwrap();

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

  auto& cufft = CUFFT::instance();
  auto ret = cufft.exec(_plan, in._data, out._data, CUFFT_FORWARD);
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

  auto& cufft = CUFFT::instance();
  auto ret = cufft.exec(_plan, in._data, out._data, CUFFT_INVERSE);
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

  auto& cufft = CUFFT::instance();
  for (auto i = 0U; i < ibatch; i += _batch) {
    auto s = in[i];
    auto d = out[i];
    _TRY(cufft.exec(_plan, s._data, d._data, CUFFT_FORWARD));
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

  auto& cufft = CUFFT::instance();
  for (auto i = 0U; i < ibatch; i += _batch) {
    auto s = in[i];
    auto d = out[i];
    _TRY(cufft.exec(_plan, s._data, d._data, CUFFT_INVERSE));
  }
  return Ok{};
}

RFFT::RFFT() noexcept {}

RFFT::~RFFT() {
  auto& cufft = CUFFT::instance();
  cufft.drop(_plan_r2c).unwrap();
  cufft.drop(_plan_c2r).unwrap();
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
  auto& cufft = CUFFT::instance();
  const auto plan_r2c = cufft.plan<f32, c32>(len, batch).unwrap();
  const auto plan_c2r = cufft.plan<c32, f32>(len, batch).unwrap();

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

  auto& cufft = CUFFT::instance();
  auto ret = cufft.exec(_plan_r2c, in._data, out._data, CUFFT_FORWARD);
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

  auto& cufft = CUFFT::instance();
  auto ret = cufft.exec(_plan_c2r, in._data, out._data, CUFFT_INVERSE);
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

  auto& cufft = CUFFT::instance();
  for (auto i = 0U; i < ibatch; i += _batch) {
    auto s = in[i];
    auto d = out[i];
    _TRY(cufft.exec(_plan_r2c, s._data, d._data, CUFFT_FORWARD));
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

  auto& cufft = CUFFT::instance();
  for (auto i = 0U; i < ibatch; i += _batch) {
    auto s = in[i];
    auto d = out[i];
    _TRY(cufft.exec(_plan_c2r, s._data, d._data, CUFFT_INVERSE));
  }
  return Ok{};
}

}  // namespace sfc::cuda::fft
