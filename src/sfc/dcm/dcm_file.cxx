#include "sfc/test.h"
#include "nct/dcm/dcm_file.h"
#include "sfc/math/ndarray.h"

namespace nct::dcm::test {

auto make_vol(const u32 (&shape)[3]) {
  const auto [nz, ny, nx] = shape;
  auto vol = math::zero<f32>(shape);
  vol.for_each_mut([&](f32& val, u32 i, u32 j, [[maybe_unused]] u32 k) {
    const auto fx = 2 * (f32(i) / f32(nx) - 0.5f);
    const auto fy = 2 * (f32(j) / f32(ny) - 0.5f);
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

}  // namespace nct::dcm::test
