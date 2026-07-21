#include "sfc/test.h"
#include "sfc/io.h"
#include "sfc/cuda/device.h"

namespace sfc::cuda::device::test {

SFC_TEST(device_query) {
  const auto dev_cnt = cuda::Device::count();

  io::println("cuda.dev: count={}", dev_cnt);
  for (auto i = 0U; i < dev_cnt; ++i) {
    const auto dev = Device{i};
    const auto info  = dev.info();
    io::println("cuda.dev[{}]", i);
    io::println("  name = {}", info.name);
    io::println("  compute_capability = {}", info.compute_capability);
    io::println("  sm_count = {}", info.sm_count);
    io::println("  global_memory = {}", info.global_memory);
    io::println("  l2_cache_size = {}", info.l2_cache_size);
    io::println("  async_engine_count = {}", info.async_engine_count);
  }
}

}  // namespace sfc::cuda::test
