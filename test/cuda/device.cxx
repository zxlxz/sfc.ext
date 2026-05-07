#include "sfc/test.h"
#include "sfc/io.h"
#include "sfc/cuda/device.h"

namespace sfc::cuda::test {

SFC_TEST(device_query) {
  const auto dev_cnt = cuda::dev_count();

  io::println("cuda.dev: count={}", dev_cnt);
  for (auto i = 0; i < dev_cnt; ++i) {
    const auto dev = Device{i};
    io::println("cuda.dev[{}]: id={}, name={:?}, total_memory={}GB",
                i,
                dev.id,
                dev.name(),
                dev.total_memory() >> 30);
  }
}

}  // namespace sfc::cuda::test
