#pragma once

#include "sfc/math/complex.h"
#include "sfc/math/ndslice.h"

namespace sfc::math {

auto fft(NdSlice<c32, 1> x, NdSlice<c32, 1> y)-> bool;
auto ifft(NdSlice<c32, 1> x, NdSlice<c32, 1> y)-> bool;

auto fft_batch(NdSlice<c32, 2> x, NdSlice<c32, 2> y)-> bool;
auto ifft_batch(NdSlice<c32, 2> x, NdSlice<c32, 2> y)-> bool;

}  // namespace sfc::math
