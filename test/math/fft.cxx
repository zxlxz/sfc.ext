#include "sfc/test.h"
#include "sfc/io.h"
#include "sfc/math/fft.h"
#include "sfc/math/ndarray.h"

namespace sfc::math::test {

SFC_TEST(fft_c2c_outplace) {
  const u32 lens[] = {2, 4, 8, 16, 32, 40};
  for (auto n : lens) {
    auto X = NdArray<c32, 1>::with_shape({n});
    auto Y = NdArray<c32, 1>::with_shape({n});
    auto fft = FFT<c32, c32>::create(n, 1);
    X.imap_mut([&](u32 i, auto& val) { val = c32{float(i), 0}; });
    fft.exec(X->_data, Y->_data, -1);
    io::println("fft_c2[{}] Y={-5.2}", n, Y);
    fft.exec(Y->_data, X->_data, +1);
    io::println("fft_c2[{}] Y={-5.2}", n, X);
  }
}

SFC_TEST(fft_c2c_inplace) {
  const u32 lens[] = {2, 4, 8, 16, 32, 40};
  for (auto n : lens) {
    auto X = NdArray<c32, 1>::with_shape({n});
    auto fft = FFT<c32, c32>::create(n, 1);
    X.imap_mut([&](u32 i, auto& val) { val = c32{float(i), 0}; });
    fft.exec(X->_data, X->_data, -1);
    fft.exec(X->_data, X->_data, +1);
    io::println("fft_c2[{}] Y={-5.2}", n, X);
  }
}

SFC_TEST(fft_r2c) {
  const u32 lens[] = {2, 4, 8, 16, 32, 40};
  for (auto N : lens) {
    auto R = NdArray<f32, 1>::with_shape({N});
    auto C = NdArray<c32, 1>::with_shape({N / 2 + 1});
    auto fft_r2c = FFT<f32, c32>::create(N, 1);
    auto fft_c2r = FFT<c32, f32>::create(N, 1);
    R.imap_mut([&](u32 i, auto& val) { val = float(i); });
    fft_r2c(R, C);
    io::println("fft_r2c[{}] C={-5.2}", N, C);
    fft_c2r(C, R);
    io::println("fft_c2r[{}] R={-5.2}", N, R);
  }
}

}  // namespace sfc::math::test
