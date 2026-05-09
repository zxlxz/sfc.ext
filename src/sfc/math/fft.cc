#include "sfc/math/fft.h"
#include "sfc/math/detail/fft.inl"
#include "sfc/cuda/fft.h"

namespace sfc::math {

static void fft_c2c_cpu(u32 N, c32 X[], c32 Y[], int DIR, int BATCH = 1) {
  for (auto k = 0U; k < BATCH; ++k) {
    auto pX = X + k * N;
    auto pY = Y + k * N;
    detail::fft_c2c(pX, pY, N, DIR);
  }
}

void fft(NdSlice<c32, 1> x, NdSlice<c32, 1> y) {
  if (x._dims.x != y._dims.x) {
    return;
  }

  const auto n = x._dims.x;
  fft_c2c_cpu(n, x._data, y._data, -1);
}

void ifft(NdSlice<c32, 1> x, NdSlice<c32, 1> y) {
  if (x._dims.x != y._dims.x) {
    return;
  }

  const auto n = x._dims.x;
  fft_c2c_cpu(n, x._data, y._data, +1);
}

void fft_batch(NdSlice<c32, 2> x, NdSlice<c32, 2> y) {
  if (x._dims.x != y._dims.x) {
    return;
  }

  if (x._dims.y != y._dims.y) {
    return;
  }

  const auto n = x._dims.x;
  const auto batch = x._dims.y;
  fft_c2c_cpu(n, x._data, y._data, -1, batch);
}

void ifft_batch(NdSlice<c32, 2> x, NdSlice<c32, 2> y) {
  if (x._dims.x != y._dims.x) {
    return;
  }
  if (x._dims.y != y._dims.y) {
    return;
  }

  const auto n = x._dims.x;
  const auto batch = x._dims.y;
  fft_c2c_cpu(n, x._data, y._data, +1, batch);
}

}  // namespace sfc::math
