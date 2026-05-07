#include "sfc/test.h"
#include "sfc/time.h"
#include "sfc/io.h"
#include "sfc/cuda.h"
#include "sfc/math/ndarray.h"

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
      io::println("fft_c2c: N={:4}, plan={:6.2f}, drop={:6.2f}, inplace={:6.2f}, outplace={:6.2f}",
                  N,
                  t2.duration_since(t1),
                  t5.duration_since(t4),
                  t3.duration_since(t2),
                  t4.duration_since(t3));
    }
  }
}

}  // namespace sfc::cuda::test
