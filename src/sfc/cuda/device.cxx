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
    io::println("cuda.dev[{}], info={s}", i, info);
  }
}

}  // namespace sfc::cuda::test
