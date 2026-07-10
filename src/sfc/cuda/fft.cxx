#include "sfc/test.h"
#include "sfc/io.h"

#include "sfc/math/ndarray.h"
#include "sfc/cuda/fft.h"
#include "sfc/cuda/device.h"

namespace sfc::cuda::fft::test {

SFC_TEST(fft_c2c) {
  cuda::device_set(0).unwrap();

  const auto fft_len = 8U;
  const auto batch = 1U;
  auto fft = cufft<c32, c32>(fft_len, batch);

  u32 cnts[] = {1 * batch, 2 * batch};
  for (auto cnt : cnts) {
    auto in = math::array<c32>({cnt, fft_len}, MemKind::RAM);
    auto out = math::array<c32>({cnt, fft_len}, MemKind::RAM);
    in.for_each([](auto i, auto j, auto& x) { x = c32{f32(i), f32(j)}; });
    io::println("in = \n {:+5.2}", in);
    fft(in, out).unwrap();
    io::println("out = \n {:+5.2}", out);
    cuda::device_sync().unwrap();
  }
}

}  // namespace sfc::cuda::fft::test
