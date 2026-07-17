#pragma once

#include "sfc/math/complex.h"
#include "sfc/math/ndarray.h"

namespace sfc::cuda::fft {

enum class Error {
  Success = 0x0,
  InvalidPlan = 0x1,
  AllocFailed = 0x2,
  InvalidType = 0x3,
  InvalidValue = 0x4,
  InternalError = 0x5,
  ExecFailed = 0x6,
  SetupFailed = 0x7,
  InvalidSize = 0x8,
  UnalignedData = 0x9,
  InvalidDevice = 0xB,
  NoWorkspace = 0xD,
  NotImplemented = 0xE,
  NotSupported = 0x10,
  MissingDependency = 0x11,
  NVRTCFailure = 0x12,
  NVJITLinkFailure = 0x13,
  NVSHMEMFailure = 0x14,
};

template <class T = Unit>
using Result = result::Result<T, Error>;

auto to_str(Error err) -> const char*;

class FFT {
  u32 _len{0};
  u32 _batch{0};
  int _plan{0};

 public:
  FFT() noexcept;
  ~FFT();
  FFT(FFT&& other) noexcept;
  FFT& operator=(FFT&& other) noexcept;

  static auto new_(u32 len, u32 batch = 1) -> FFT;

 public:
  auto len() const -> usize;
  auto batch() const -> usize;

  auto fft(math::NdSlice<c32, 1> in, math::NdSlice<c32, 1> out) -> Result<>;
  auto ifft(math::NdSlice<c32, 1> in, math::NdSlice<c32, 1> out) -> Result<>;

  auto fft(math::NdSlice<c32, 2> in, math::NdSlice<c32, 2> out) -> Result<>;
  auto ifft(math::NdSlice<c32, 2> in, math::NdSlice<c32, 2> out) -> Result<>;
};

class RFFT {
  u32 _len{0};
  u32 _batch{0};
  int _plan_r2c{0};
  int _plan_c2r{0};

 public:
  RFFT() noexcept;
  ~RFFT();
  RFFT(RFFT&& other) noexcept;
  RFFT& operator=(RFFT&& other) noexcept;

  static auto create(u32 len, u32 batch = 1) -> RFFT;

 public:
  auto fft(math::NdSlice<f32, 1> in, math::NdSlice<c32, 1> out) -> Result<>;
  auto ifft(math::NdSlice<c32, 1> in, math::NdSlice<f32, 1> out) -> Result<>;

  auto fft(math::NdSlice<f32, 2> in, math::NdSlice<c32, 2> out) -> Result<>;
  auto ifft(math::NdSlice<c32, 2> in, math::NdSlice<f32, 2> out) -> Result<>;
};

}  // namespace sfc::cuda::fft
