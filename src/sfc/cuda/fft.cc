#include <cuda.h>
#include <cufft.h>

#include "sfc/ffi/library.h"
#include "sfc/cuda/stream.h"
#include "sfc/cuda/fft.h"
#include "sfc/math/vec.h"

namespace sfc::cuda {

namespace detail {

auto fft_err(cufftResult fft_err) -> Error {
  switch (fft_err) {
    case CUFFT_SUCCESS:            return Error(CUDA_SUCCESS);
    case CUFFT_INVALID_PLAN:       return Error(CUDA_ERROR_INVALID_HANDLE);
    case CUFFT_ALLOC_FAILED:       return Error(CUDA_ERROR_OUT_OF_MEMORY);
    case CUFFT_INVALID_TYPE:       return Error(CUDA_ERROR_INVALID_VALUE);
    case CUFFT_INVALID_VALUE:      return Error(CUDA_ERROR_INVALID_VALUE);
    case CUFFT_INTERNAL_ERROR:     return Error(CUDA_ERROR_UNKNOWN);
    case CUFFT_EXEC_FAILED:        return Error(CUDA_ERROR_LAUNCH_FAILED);
    case CUFFT_SETUP_FAILED:       return Error(CUDA_ERROR_NOT_INITIALIZED);
    case CUFFT_INVALID_SIZE:       return Error(CUDA_ERROR_INVALID_VALUE);
    case CUFFT_UNALIGNED_DATA:     return Error(CUDA_ERROR_MISALIGNED_ADDRESS);
    case CUFFT_INVALID_DEVICE:     return Error(CUDA_ERROR_INVALID_DEVICE);
    case CUFFT_NO_WORKSPACE:       return Error(CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES);
    case CUFFT_NOT_IMPLEMENTED:    return Error(CUDA_ERROR_NOT_SUPPORTED);
    case CUFFT_NOT_SUPPORTED:      return Error(CUDA_ERROR_NOT_SUPPORTED);
    case CUFFT_MISSING_DEPENDENCY: return Error(CUDA_ERROR_UNKNOWN);
    case CUFFT_NVRTC_FAILURE:      return Error(CUDA_ERROR_UNKNOWN);
    case CUFFT_NVJITLINK_FAILURE:  return Error(CUDA_ERROR_UNKNOWN);
    case CUFFT_NVSHMEM_FAILURE:    return Error(CUDA_ERROR_UNKNOWN);
    default:                       return Error(CUDA_ERROR_UNKNOWN);
  }
}

class CUFFT {
  decltype(cufftDestroy)* _cufftDestroy = nullptr;
  decltype(cufftPlan1d)* _cufftPlan1d = nullptr;
  decltype(cufftExecC2C)* _cufftExecC2C = nullptr;
  decltype(cufftExecR2C)* _cufftExecR2C = nullptr;
  decltype(cufftExecC2R)* _cufftExecC2R = nullptr;
  decltype(cufftSetStream)* _cufftSetStream = nullptr;

 public:
  static auto from(ffi::Library& lib) {
    auto res = CUFFT{};
    res._cufftDestroy = lib.func("cufftDestroy");
    res._cufftPlan1d = lib.func("cufftPlan1d");
    res._cufftExecC2C = lib.func("cufftExecC2C");
    res._cufftExecR2C = lib.func("cufftExecR2C");
    res._cufftExecC2R = lib.func("cufftExecC2R");
    res._cufftSetStream = lib.func("cufftSetStream");
    return res;
  }

 public:
  auto set_stream(cufftHandle plan, CUstream stream) const -> Result<> {
    if (auto err = _cufftSetStream(plan, stream)) {
      return fft_err(err);
    }
    return Ok{};
  }

  auto destroy(cufftHandle plan) const -> Result<> {
    if (auto err = _cufftDestroy(plan)) {
      return fft_err(err);
    }
    return Ok{};
  }

  auto plan_1d_c2c(int nx, int batch) const -> Result<cufftHandle> {
    auto plan = cufftHandle{};
    if (auto err = _cufftPlan1d(&plan, nx, CUFFT_C2C, batch)) {
      return fft_err(err);
    }
    return Ok(plan);
  }

  auto plan_1d_c2r(int nx, int batch) const -> Result<cufftHandle> {
    auto plan = cufftHandle{};
    if (auto err = _cufftPlan1d(&plan, nx, CUFFT_C2R, batch)) {
      return fft_err(err);
    }
    return Ok(plan);
  }

  auto plan_1d_r2c(int nx, int batch) const -> Result<cufftHandle> {
    auto plan = cufftHandle{};
    if (auto err = _cufftPlan1d(&plan, nx, CUFFT_R2C, batch)) {
      return fft_err(err);
    }
    return Ok(plan);
  }

  auto exec_c2c(cufftHandle plan, c32* in, c32* out, int dir) const -> Result<> {
    const auto idata = ptr::cast<cufftComplex>(in);
    const auto odata = ptr::cast<cufftComplex>(out);
    if (auto err = _cufftExecC2C(plan, idata, odata, dir)) {
      return fft_err(err);
    }
    return Ok{};
  }

  auto exec_r2c(cufftHandle plan, f32* in, c32* out) const -> Result<> {
    const auto idata = in;
    const auto odata = ptr::cast<cufftComplex>(out);
    if (auto err = _cufftExecR2C(plan, idata, odata)) {
      return fft_err(err);
    }
    return Ok{};
  }

  auto exec_c2r(cufftHandle plan, c32* in, f32* out) const -> Result<> {
    const auto idata = ptr::cast<cufftComplex>(in);
    const auto odata = out;
    if (auto err = _cufftExecC2R(plan, idata, odata)) {
      return fft_err(err);
    }
    return Ok{};
  }
};

auto fft_lib() -> CUFFT& {
  cuda::init().unwrap();

#ifdef _WIN32
  const auto path = Str{"cufft64_12.dll"};
#else
  const auto path = Str{"libcufft"};
#endif
  static auto lib = ffi::Library::load(path).unwrap();
  static auto fft = CUFFT::from(lib);
  return fft;
}

auto fft_destroy(cufftHandle plan) -> Result<> {
  if (plan == 0) {
    return Ok{};
  }

  auto& lib = fft_lib();
  return lib.destroy(plan);
}

template <class I, class O>
auto fft_plan_1d(u32 len, u32 batch) -> Result<cufftHandle> {
  auto& lib = fft_lib();
  if constexpr (trait::same_<I, c32> && trait::same_<O, c32>) {
    return lib.plan_1d_c2c(int(len), int(batch));
  } else if constexpr (trait::same_<I, f32> && trait::same_<O, c32>) {
    return lib.plan_1d_r2c(int(len), int(batch));
  } else if constexpr (trait::same_<I, c32> && trait::same_<O, f32>) {
    return lib.plan_1d_c2r(int(len), int(batch));
  } else {
    static_assert(false, "unsupported type combination");
  }
}

template <class I, class O>
auto fft_exec(cufftHandle plan, I in[], O out[], int dir) -> Result<> {
  auto& lib = detail::fft_lib();

  _TRY(lib.set_stream(plan, cuda::stream_current()));

  if constexpr (trait::same_<I, c32> && trait::same_<O, c32>) {
    return lib.exec_c2c(plan, in, out, dir);
  } else if constexpr (trait::same_<I, f32> && trait::same_<O, c32>) {
    return lib.exec_r2c(plan, in, out);
  } else if constexpr (trait::same_<I, c32> && trait::same_<O, f32>) {
    return lib.exec_c2r(plan, in, out);
  } else {
    static_assert(false, "unsupported type combination");
  }
}

}  // namespace detail

CFFT::CFFT() noexcept : _plan{0} {}

CFFT::~CFFT() {
  if (_plan == 0) {
    return;
  }
  detail::fft_destroy(_plan).unwrap();
}

CFFT::CFFT(CFFT&& other) noexcept
    : _len{mem::take(other._len)}, _batch{mem::take(other._batch)}, _plan{mem::take(other._plan)} {}

auto CFFT::operator=(CFFT&& other) noexcept -> CFFT& {
  if (this != &other) {
    mem::swap(_len, other._len);
    mem::swap(_batch, other._batch);
    mem::swap(_plan, other._plan);
  }
  return *this;
}

auto CFFT::new_(u32 len, u32 batch) -> CFFT {
  const auto plan = detail::fft_plan_1d<c32, c32>(len, batch).unwrap();

  auto res = CFFT{};
  res._len = len;
  res._batch = batch;
  res._plan = plan;
  return res;
}

auto CFFT::len() const -> usize {
  return _len;
}

auto CFFT::batch() const -> usize {
  return _batch;
}

auto CFFT::fft(math::NdView<c32, 1> in, math::NdView<c32, 1> out) -> Result<> {
  const auto [ilen] = in._shape;
  const auto [olen] = out._shape;

  sfc::assert_(in.is_contiguous(), "CFFT::fft: in is not contiguous");
  sfc::assert_(out.is_contiguous(), "CFFT::fft: out is not contiguous");
  sfc::assert_(ilen == _len, "CFFT::fft: in.shape({}) not match fft.len(={})", ilen, _len);
  sfc::assert_(olen == _len, "CFFT::fft: out.shape({}) not match fft.len(={})", olen, _len);
  sfc::assert_(_batch == 1, "CFFT::fft: batch({}) != 1", _batch);

  auto ret = detail::fft_exec(_plan, in._data, out._data, CUFFT_FORWARD);
  return ret;
}

auto CFFT::ifft(math::NdView<c32, 1> in, math::NdView<c32, 1> out) -> Result<> {
  const auto [ilen] = in._shape;
  const auto [olen] = out._shape;

  sfc::assert_(in.is_contiguous(), "CFFT::ifft: in is not contiguous");
  sfc::assert_(out.is_contiguous(), "CFFT::ifft: out is not contiguous");

  sfc::assert_(ilen == _len, "CFFT::ifft: in.shape({}) not match fft.len(={})", ilen, _len);
  sfc::assert_(olen == _len, "CFFT::ifft: out.shape({}) not match fft.len(={})", olen, _len);
  sfc::assert_(_batch == 1, "CFFT::ifft: batch({}) != 1", _batch);

  auto ret = detail::fft_exec(_plan, in._data, out._data, CUFFT_INVERSE);
  return ret;
}

auto CFFT::fft(math::NdView<c32, 2> in, math::NdView<c32, 2> out) -> Result<> {
  const auto [ibatch, ilen] = in._shape;
  const auto [obatch, olen] = out._shape;

  sfc::assert_(in.is_contiguous(), "CFFT::fft: in is not contiguous");
  sfc::assert_(out.is_contiguous(), "CFFT::fft: out is not contiguous");
  sfc::assert_(ilen == _len, "CFFT::fft: in.shape({}) not match fft.len(={})", ilen, _len);
  sfc::assert_(olen == _len, "CFFT::fft: out.shape({}) not match fft.len(={})", olen, _len);
  sfc::assert_(ibatch == obatch, "CFFT::fft: in.batch({}) not match out.batch(={})", ibatch, obatch);
  sfc::assert_(ibatch % _batch == 0, "CFFT::fft: in.batch({}) not multiple of batch(={})", ibatch, _batch);
  sfc::assert_(obatch % _batch == 0, "CFFT::fft: out.batch({}) not multiple of batch(={})", obatch, _batch);

  for (auto i = 0U; i < ibatch; i += _batch) {
    auto s = in[i];
    auto d = out[i];
    _TRY(detail::fft_exec(_plan, s._data, d._data, CUFFT_FORWARD));
  }
  return Ok{};
}

auto CFFT::ifft(math::NdView<c32, 2> in, math::NdView<c32, 2> out) -> Result<> {
  const auto [ibatch, ilen] = in._shape;
  const auto [obatch, olen] = out._shape;

  sfc::assert_(in.is_contiguous(), "CFFT::ifft: in is not contiguous");
  sfc::assert_(out.is_contiguous(), "CFFT::ifft: out is not contiguous");
  sfc::assert_(ilen == _len, "CFFT::ifft: in.shape({}) not match fft.len(={})", ilen, _len);
  sfc::assert_(olen == _len, "CFFT::ifft: out.shape({}) not match fft.len(={})", olen, _len);
  sfc::assert_(ibatch == obatch, "CFFT::ifft: in.batch({}) not match out.batch(={})", ibatch, obatch);
  sfc::assert_(ibatch % _batch == 0, "CFFT::ifft: in.batch({}) not multiple of batch(={})", ibatch, _batch);
  sfc::assert_(obatch % _batch == 0, "CFFT::ifft: out.batch({}) not multiple of batch(={})", obatch, _batch);

  for (auto i = 0U; i < ibatch; i += _batch) {
    auto s = in[i];
    auto d = out[i];
    _TRY(detail::fft_exec(_plan, s._data, d._data, CUFFT_INVERSE));
  }

  return Ok{};
}

RFFT::RFFT() noexcept {}

RFFT::~RFFT() {
  if (_plan_r2c != 0) detail::fft_destroy(_plan_r2c).unwrap();
  if (_plan_c2r != 0) detail::fft_destroy(_plan_c2r).unwrap();
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

auto RFFT::new_(u32 len, u32 batch) -> RFFT {
  const auto plan_r2c = detail::fft_plan_1d<f32, c32>(len, batch).unwrap();
  const auto plan_c2r = detail::fft_plan_1d<c32, f32>(len, batch).unwrap();

  auto res = RFFT{};
  res._len = len;
  res._batch = batch;
  res._plan_r2c = plan_r2c;
  res._plan_c2r = plan_c2r;
  return res;
}

auto RFFT::fft(math::NdView<f32, 1> in, math::NdView<c32, 1> out) -> Result<> {
  const auto full_len = _len;
  const auto half_len = _len / 2 + 1;

  const auto [src_len] = in._shape;
  const auto [dst_len] = out._shape;

  sfc::assert_(in.is_contiguous(), "RFFT::fft: in is not contiguous");
  sfc::assert_(out.is_contiguous(), "RFFT::fft: out is not contiguous");
  sfc::assert_(src_len == full_len, "RFFT::fft: in.shape({}) not match len(={})", src_len, _len);
  sfc::assert_(dst_len == half_len, "RFFT::fft: out.shape({}) not match len(={}/2+1)", dst_len, _len);
  sfc::assert_(_batch == 1, "RFFT::fft: batch({}) != 1", _batch);

  auto ret = detail::fft_exec(_plan_r2c, in._data, out._data, CUFFT_FORWARD);
  return ret;
}

auto RFFT::ifft(math::NdView<c32, 1> in, math::NdView<f32, 1> out) -> Result<> {
  const auto full_len = _len;
  const auto half_len = _len / 2 + 1;

  const auto [ilen] = in._shape;
  const auto [olen] = out._shape;

  sfc::assert_(in.is_contiguous(), "RFFT::ifft: in is not contiguous");
  sfc::assert_(out.is_contiguous(), "RFFT::ifft: out is not contiguous");
  sfc::assert_(ilen == half_len, "RFFT::ifft: in.shape({}) not match fft.len(={})/2+1", ilen, _len);
  sfc::assert_(olen == full_len, "RFFT::ifft: out.shape({}) not match fft.len(={})", olen, _len);
  sfc::assert_(_batch == 1, "RFFT::ifft: batch({}) != 1", _batch);

  const auto ret = detail::fft_exec(_plan_c2r, in._data, out._data, CUFFT_INVERSE);
  return ret;
}

auto RFFT::fft(math::NdView<f32, 2> in, math::NdView<c32, 2> out) -> Result<> {
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
    _TRY(detail::fft_exec(_plan_r2c, s._data, d._data, CUFFT_FORWARD));
  }
  return Ok{};
}

auto RFFT::ifft(math::NdView<c32, 2> in, math::NdView<f32, 2> out) -> Result<> {
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
    _TRY(detail::fft_exec(_plan_c2r, s._data, d._data, CUFFT_INVERSE));
  }
  return Ok{};
}

}  // namespace sfc::cuda
