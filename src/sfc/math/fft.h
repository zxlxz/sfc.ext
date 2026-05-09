#pragma once

#include "sfc/math/complex.h"
#include "sfc/math/ndslice.h"

namespace sfc::math {

void fft(NdSlice<c32, 1> x, NdSlice<c32, 1> y);
void ifft(NdSlice<c32, 1> x, NdSlice<c32, 1> y);

void fft_batch(NdSlice<c32, 2> x, NdSlice<c32, 2> y);
void ifft_batch(NdSlice<c32, 2> x, NdSlice<c32, 2> y);

}  // namespace sfc::math
