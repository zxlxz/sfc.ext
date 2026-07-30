#include "sfc/test.h"
#include "sfc/dcm/dcm_file.h"
#include "sfc/math/ndarray.h"

namespace sfc::dcm::test {

auto make_vol(const u32 (&shape)[3]) {
  const auto [nz, ny, nx] = shape;
  auto vol = math::zero<f32>(shape);
  vol.for_each([&](u32 n, u32 y, u32 x, f32& val) {
    (void)n;
    const auto fx = 2 * (f32(x) / f32(nx) - 0.5f);
    const auto fy = 2 * (f32(y) / f32(ny) - 0.5f);
    const auto r = 1 - math::sqrtf(fx * fx + fy * fy);
    val = r;
  });
  return vol;
}

SFC_TEST(dcm_file_read_write) {
  auto vol = make_vol({16, 64, 64});
  auto dcm = DcmFile::create("test.dcm").unwrap();
  dcm.write_meta(DcmMeta{});
  dcm.write_data(vol);
}

}  // namespace sfc::dcm::test
