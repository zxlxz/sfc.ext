#pragma once

#include "sfc/cuda/memory.h"
#include "sfc/math/complex.h"
#include "sfc/math/ndslice.h"

struct fftwf_plan_s;

namespace sfc::math {

template <class I, class O>
struct FFT;

template <>
struct FFT<f32, c32> {
  using plan_t = struct fftwf_plan_s*;
  u32 _len{0};
  u32 _batch{0};
  plan_t _plan{nullptr};

 public:
  FFT();
  ~FFT();
  FFT(FFT&& other) noexcept;
  FFT& operator=(FFT&& other) noexcept;

  static auto create(u32 len, u32 batch) -> FFT;

 public:
  void exec(f32 X[], c32 Y[]);

  void operator()(NdSlice<f32, 2> in, NdSlice<c32, 2> out) {
    this->exec(in._data, out._data);
  }
};

template <>
struct FFT<c32, f32> {
  using plan_t = struct fftwf_plan_s*;
  u32 _len{0};
  u32 _batch{0};
  plan_t _plan{nullptr};

 public:
  FFT();
  ~FFT();
  FFT(FFT&& other) noexcept;
  FFT& operator=(FFT&& other) noexcept;

  static auto create(u32 len, u32 batch) -> FFT;

 public:
  void exec(c32 X[], f32 Y[]);

  void operator()(NdSlice<c32, 2> in, NdSlice<f32, 2> out) {
    this->exec(in._data, out._data);
  }
};

template <>
struct FFT<c32, c32> {
  using plan_t = struct fftwf_plan_s*;
  u32 _len{0};
  u32 _batch{0};
  plan_t _inplace_fwd = nullptr;
  plan_t _inplace_inv = nullptr;
  plan_t _outplace_fwd = nullptr;
  plan_t _outplace_inv = nullptr;

 public:
  FFT();
  ~FFT();
  FFT(FFT&& other) noexcept;
  FFT& operator=(FFT&& other) noexcept;

  static auto create(u32 len, u32 batch) -> FFT;

 public:
  void exec(c32 X[], c32 Y[], int DIR);

  void operator()(NdSlice<c32, 2> in, NdSlice<c32, 2> out, int DIR) {
    this->exec(in._data, out._data, DIR);
  }
};

void fft(NdSlice<c32, 1> in, NdSlice<c32, 1> out);
void ifft(NdSlice<c32, 1> in, NdSlice<c32, 1> out);

}  // namespace sfc::math
