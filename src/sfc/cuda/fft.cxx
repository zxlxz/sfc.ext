#include "sfc/test.h"
#include "sfc/time.h"
#include "sfc/io.h"

#include "sfc/math/ndarray.h"
#include "sfc/cuda/device.h"
#include "sfc/cuda/fft.h"

namespace sfc::cuda::fft::test {

using cuda::CUFFT;
using math::NdArray;
using math::MemKind;

SFC_TEST(fft_c2c_perf) {
  const u32 lens[] = {
      200,   // 2^3 * 5^2
      256,   // 2^8
      500,   // 2^2 * 5^3
      512,   // 2^9
      600,   // 2^3 * 3 * 5^2
      709,   // prime
      1024,  // 2^10
      2048,  // 2^11
  };

  const u32 batchs[] = {1, 10, 100, 1000, 2000, 4000, 8000};

  for (auto BATCH : batchs) {
    io::println(" === Batch = {} === ", BATCH);
    for (const auto N : lens) {
      auto X = math::array<c32>({N, BATCH}, MemKind::GPU);
      auto Y = math::array<c32>({N, BATCH}, MemKind::GPU);
      X.bzero();
      cuda::device_sync().unwrap();
      const auto t1 = time::Instant::now();
      auto fft_c2c = cuda::cufft<c32, c32>(N, BATCH);
      cuda::device_sync().unwrap();
      const auto t2 = time::Instant::now();
      fft_c2c(X, X, -1).unwrap();
      cuda::device_sync().unwrap();
      const auto t3 = time::Instant::now();
      fft_c2c(X, X, +1).unwrap();
      cuda::device_sync().unwrap();
      const auto t4 = time::Instant::now();
      fft_c2c = {};
      cuda::device_sync().unwrap();
      const auto t5 = time::Instant::now();
      io::println("fft_c2c: N={:4}, plan={:5.3m}, drop={:5.3m}, inplace={:5.3m}, outplace={:5.3m}",
                  N,
                  t2.duration_since(t1),
                  t5.duration_since(t4),
                  t3.duration_since(t2),
                  t4.duration_since(t3));
    }
  }
}

SFC_TEST(fft_r2c_perf) {
  const u32 lens[] = {
      200,   // 2^3 * 5^2
      256,   // 2^8
      500,   // 2^2 * 5^3
      512,   // 2^9
      600,   // 2^3 * 3 * 5^2
      709,   // prime
      1024,  // 2^10
      2048,  // 2^11
  };

  const u32 batchs[] = {1, 10, 100, 1000, 2000, 4000, 8000};

  for (auto BATCH : batchs) {
    io::println(" === Batch = {} === ", BATCH);
    for (const auto N : lens) {
      auto X = math::array<f32>({N, BATCH}, {MemKind::GPU});
      auto Y = math::array<c32>({N / 2 + 1, BATCH}, {MemKind::GPU});
      X.bzero();
      cuda::device_sync().unwrap();
      const auto t1 = time::Instant::now();
      auto fft_r2c = cuda::cufft<f32, c32>(N, BATCH);
      auto fft_c2r = cuda::cufft<c32, f32>(N, BATCH);
      cuda::device_sync().unwrap();
      const auto t2 = time::Instant::now();
      fft_r2c(X, Y, -1).unwrap();
      cuda::device_sync().unwrap();
      const auto t3 = time::Instant::now();
      fft_c2r(Y, X, +1).unwrap();
      cuda::device_sync().unwrap();
      const auto t4 = time::Instant::now();
      fft_r2c = {};
      fft_c2r = {};
      cuda::device_sync().unwrap();
      const auto t5 = time::Instant::now();
      io::println("fft_r2c: N={:4}, plan={:5.3m}, drop={:5.3m}, r2c={:5.3m}, c2r={:5.3m}",
                  N,
                  t2.duration_since(t1),
                  t4.duration_since(t3),
                  t3.duration_since(t2),
                  t5.duration_since(t4));
    }
  }
}

}  // namespace sfc::fft::test
