#include "sfc/test.h"
#include "sfc/io.h"
#include "sfc/math/ndarray.h"
#include "sfc/math/fft.h"

namespace sfc::math::fft::test {

SFC_TEST(fft_c2c_outplace) {
  const u32 lens[] = {2, 4, 8, 16, 32, 40};
  for (auto n : lens) {
    auto X = math::array<c32>({n});
    auto Y = math::array<c32>({n});
    auto fft = FFT::new_(n);
    X.for_each([&](u32 i, c32& val) { val = c32{float(i), 0}; });
    fft.fft(X, Y);
    io::println("fft_c2[{}] Y={-5.2}", n, Y);
    fft.ifft(Y, X);
    io::println("fft_c2[{}] Y={-5.2}", n, X);
  }
}

SFC_TEST(fft_c2c_inplace) {
  const u32 lens[] = {2, 4, 8, 16, 32, 40};
  for (auto n : lens) {
    auto X = math::array<c32>({n});
    auto fft = FFT::new_(n);
    X.for_each([&](u32 i, c32& val) { val = c32{float(i), 0}; });
    fft.fft(X, X);
    fft.ifft(X, X);
    io::println("fft_c2[{}] Y={-5.2}", n, X);
  }
}

SFC_TEST(fft_r2c) {
  const u32 lens[] = {2, 4, 8, 16, 32, 40};
  for (auto N : lens) {
    auto R = math::array<f32>({N});
    auto C = math::array<c32>({N / 2 + 1});
    auto fft_r2c = RFFT::new_(N);
    auto fft_c2r = RFFT::new_(N);
    R.for_each([&](u32 i, f32& val) { val = float(i); });
    fft_r2c.fft(R, C);
    io::println("fft_r2c[{}] C={-5.2}", N, C);
    fft_c2r.ifft(C, R);
    io::println("fft_c2r[{}] R={-5.2}", N, R);
  }
}

}  // namespace sfc::math::fft::test
