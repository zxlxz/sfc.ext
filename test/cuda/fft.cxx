#include "sfc/test.h"
#include "sfc/time.h"
#include "sfc/io.h"
#include "sfc/cuda.h"
#include "sfc/math.h"

namespace sfc::cuda::test {

SFC_TEST(fft_c2c_perf) {
  cuda::device_set(0);

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
      auto X = NdArray<c32, 2>::with_shape({N, BATCH}, MemType::Device);
      auto Y = NdArray<c32, 2>::with_shape({N, BATCH}, MemType::Device);
      X.buf().zero();
      cuda::device_sync();
      const auto t1 = time::Instant::now();
      auto plan = cuda::fft_plan_c2c(N, BATCH);
      cuda::device_sync();
      const auto t2 = time::Instant::now();
      cuda::fft_exec_c2c(plan, X->_data, X->_data, -1);
      cuda::device_sync();
      const auto t3 = time::Instant::now();
      cuda::fft_exec_c2c(plan, X->_data, Y->_data, -1);
      cuda::device_sync();
      const auto t4 = time::Instant::now();
      cuda::fft_drop(plan);
      cuda::device_sync();
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
  cuda::device_set(0);

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
      auto X = NdArray<f32, 2>::with_shape({N, BATCH}, MemType::Device);
      auto Y = NdArray<c32, 2>::with_shape({N / 2 + 1, BATCH}, MemType::Device);
      X.buf().zero();
      cuda::device_sync();
      const auto t1 = time::Instant::now();
      auto plan_r2c = cuda::fft_plan_r2c(N, BATCH);
      auto plan_c2r = cuda::fft_plan_c2r(N, BATCH);
      cuda::device_sync();
      const auto t2 = time::Instant::now();
      cuda::fft_exec_r2c(plan_r2c, X->_data, Y->_data);
      cuda::device_sync();
      const auto t3 = time::Instant::now();
      cuda::fft_exec_c2r(plan_c2r, Y->_data, X->_data);
      cuda::device_sync();
      const auto t4 = time::Instant::now();
      cuda::fft_drop(plan_r2c);
      cuda::fft_drop(plan_c2r);
      cuda::device_sync();
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

}  // namespace sfc::cuda::test
