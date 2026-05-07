#include "sfc/test.h"
#include "sfc/io.h"
#include "sfc/math/fft.h"
#include "sfc/math/ndarray.h"

namespace sfc::math::test {

SFC_TEST(fft_16) {
  const auto N = 16U;
  for (auto n = 2U; n <= N; ++n) {
    auto X = NdArray<c32, 1>::with_shape({n});
    auto Y = NdArray<c32, 1>::with_shape({n});
    X->imap([&](u32 i, auto& val) { val = c32{float(i), 0}; });
    const auto ret = math::fft(X, Y);
    if (ret) {
      io::println("fft[{3}] = \n{.2f}\n", n, *Y);
    }
  }
}

SFC_TEST(ifft_16) {
  const auto N = 16U;
  for (auto n = 2U; n <= N; ++n) {
    auto X = NdArray<c32, 1>::with_shape({n});
    auto Y = NdArray<c32, 1>::with_shape({n});
    X->imap([&](u32 i, auto& val) { val = c32{float(i), 0}; });
    const auto ret = math::ifft(X, Y);
    if (ret) {
      io::println("ifft[{}] = \n{.2f}\n", n, *Y);
    }
  }
}

SFC_TEST(fft_outplace) {
  const auto N = 32U;
  auto X = NdArray<c32, 1>::with_shape({N});
  auto Y = NdArray<c32, 1>::with_shape({N});
  X->imap([&](u32 i, auto& val) { val = c32{float(i), 0}; });

  const auto ret = math::fft(X, Y);
  if (ret) {
    io::println("fft[{}] = \n{.2f}\n", N, *Y);
  }
}

SFC_TEST(ifft_outplace) {
  const auto N = 32U;
  auto X = NdArray<c32, 1>::with_shape({N});
  auto Y = NdArray<c32, 1>::with_shape({N});
  X->imap([&](u32 i, auto& val) { val = c32{float(i), 0}; });

  const auto ret = math::ifft(X, Y);
  if (ret) {
    io::println("ifft[{}] = \n{.2f}\n", N, *Y);
  }
}

SFC_TEST(fft_inplace) {
  const auto N = 64U;

  auto X = NdArray<c32, 1>::with_shape({N});
  X->imap([&](u32 i, auto& val) { val = c32{float(i), 0}; });

  const auto ret = math::fft(X, X);
  if (ret) {
    io::println("fft[{}] = \n{.2f}\n", N, *X);
  }
}

SFC_TEST(ifft_inplace) {
  const auto N = 64U;

  auto X = NdArray<c32, 1>::with_shape({N});
  X->imap([&](u32 i, auto& val) { val = c32{float(i), 0}; });

  const auto ret = math::ifft(X, X);
  if (ret) {
    io::println("ifft[{}] = \n{}", N, *X);
  }
}

}  // namespace sfc::math::test
