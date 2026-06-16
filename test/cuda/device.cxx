#include "sfc/test.h"
#include "sfc/io.h"
#include "sfc/cuda/device.h"

namespace sfc::cuda::test {

SFC_TEST(device_query) {
  const auto dev_cnt = cuda::dev_count();

  io::println("cuda.dev: count={}", dev_cnt);
  for (auto i = 0U; i < dev_cnt; ++i) {
    const auto dev = Device{i};
    io::println("cuda.dev[{}]", i);
    io::println("  name = {}", dev.name());
    io::println("  compute_capability = {}", dev.compute_capability());
    io::println("  sm_count = {}", dev.sm_count());
    io::println("  global_memory = {}", dev.global_memory());
    io::println("  l2_cache_size = {}", dev.l2_cache_size());
    io::println("  async_engine_count = {}", dev.async_engine_count());
  }
}

}  // namespace sfc::cuda::test
