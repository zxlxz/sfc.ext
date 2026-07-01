#include "sfc/test.h"
#include "sfc/io.h"
#include "sfc/fft/fftw.h"
#include "sfc/math/ndarray.h"

namespace sfc::fft::test {

using math::NdArray;

SFC_TEST(fft_c2c_outplace) {
  const u32 lens[] = {2, 4, 8, 16, 32, 40};
  for (auto n : lens) {
    auto X = math::array<c32>({n});
    auto Y = math::array<c32>({n});
    auto fft = fft::FFTW<c32, c32>::create(n);
    X.imap_mut([&](u32 i, auto& val) { val = c32{float(i), 0}; });
    fft(X, Y, -1);
    io::println("fft_c2[{}] Y={-5.2}", n, Y);
    fft(Y, X, +1);
    io::println("fft_c2[{}] Y={-5.2}", n, X);
  }
}

SFC_TEST(fft_c2c_inplace) {
  const u32 lens[] = {2, 4, 8, 16, 32, 40};
  for (auto n : lens) {
    auto X = math::array<c32>({n});
    auto fft = fft::FFTW<c32, c32>::create(n);
    X.imap_mut([&](u32 i, auto& val) { val = c32{float(i), 0}; });
    fft(X, X, -1);
    fft(X, X, +1);
    io::println("fft_c2[{}] Y={-5.2}", n, X);
  }
}

SFC_TEST(fft_r2c) {
  const u32 lens[] = {2, 4, 8, 16, 32, 40};
  for (auto N : lens) {
    auto R = math::array<f32>({N});
    auto C = math::array<c32>({N / 2 + 1});
    auto fft_r2c = fft::FFTW<f32, c32>::create(N);
    auto fft_c2r = fft::FFTW<c32, f32>::create(N);
    R.imap_mut([&](u32 i, auto& val) { val = float(i); });
    fft_r2c(R, C);
    io::println("fft_r2c[{}] C={-5.2}", N, C);
    fft_c2r(C, R);
    io::println("fft_c2r[{}] R={-5.2}", N, R);
  }
}

}  // namespace sfc::fft::test
