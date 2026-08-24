#include "sfc/test.h"
#include "sfc/io.h"

#include "sfc/cuda.h"
#include "sfc/cuda/fft.h"

namespace sfc::cuda::test {

SFC_TEST(fft_c2c) {
  const auto fft_len = 8U;
  const auto batch = 1U;
  auto fft = CFFT::new_(fft_len, batch);

  u32 cnts[] = {1 * batch, 2 * batch};
  for (auto cnt : cnts) {
    auto in = cuda::empty<c32>({cnt, fft_len}, MemLocation::Host());
    auto out = cuda::empty<c32>({cnt, fft_len}, MemLocation::Host());
    in.for_each_mut([](auto& y, auto... i) { y = c32{f32(i)...}; });
    io::println("in = \n {:+5.2}", in);
    fft.fft(in, out).unwrap();
    io::println("out = \n {:+5.2}", out);
    cuda::Device::sync().unwrap();
  }
}

}  // namespace sfc::cuda::test
