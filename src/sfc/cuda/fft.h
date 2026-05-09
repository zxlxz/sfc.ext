#pragma once

#include "sfc/cuda/mod.h"

namespace sfc::math {
struct c32;
}

namespace sfc::cuda {

using fft_plan_t = int;
using math::c32;

auto fft_plan_c2c(u32 nx, u32 batch) -> fft_plan_t;
auto fft_plan_r2c(u32 nx, u32 batch) -> fft_plan_t;
auto fft_plan_c2r(u32 nx, u32 batch) -> fft_plan_t;

void fft_drop(fft_plan_t plan);

/// execute C2C FFT
/// @param plan create by fft_plan_c2c
/// @param in c32[n]
/// @param out c32[n]
/// @param direction -1 for forward, 1 for inverse
void fft_exec_c2c(fft_plan_t plan, const c32* in, c32* out, int direction);

/// execute R2C FFT (forward only)
/// @param plan create by fft_plan_r2c
/// @param in f32[n]
/// @param out c32[n/2+1]
void fft_exec_r2c(fft_plan_t plan, const f32* in, c32* out);

/// execute C2R FFT (inverse only)
/// @param plan create by fft_plan_c2r
/// @param in c32[n/2+1]
/// @param out f32[n]
void fft_exec_c2r(fft_plan_t plan, const c32* in, f32* out);

}  // namespace sfc::cuda
