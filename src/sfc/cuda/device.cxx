#include "sfc/test.h"
#include "sfc/io.h"
#include "sfc/cuda/device.h"

namespace sfc::cuda::test {

SFC_TEST(device_query) {
  const auto dev_cnt = cuda::Device::count().unwrap();

  io::println("cuda.dev: count={}", dev_cnt);
  for (auto i = 0U; i < dev_cnt; ++i) {
    const auto dev = Device::of(i).unwrap();
    const auto info = dev.info().unwrap();
    io::println("cuda.dev[{}] = {#}", i, info);
  }
}

}  // namespace sfc::cuda::test
