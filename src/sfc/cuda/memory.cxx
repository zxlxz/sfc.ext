#include "sfc/test.h"
#include "sfc/cuda/memory.h"

namespace sfc::cuda::memory::test {

SFC_TEST(heap) {
  const auto n = 16U;
  auto p = cuda::mem_allocate(n * sizeof(u32), MemKind::CPU);
  sfc::assert_ne(p, nullptr);

  const auto loc = cuda::mem_location(p);
  sfc::assert_eq(loc.kind, MemKind::CPU);
  cuda::mem_deallocate(p, loc);
}

SFC_TEST(host) {
  const auto n = 16U;

  auto p = cuda::mem_allocate(n * sizeof(u32), MemKind::RAM);
  sfc::assert_ne(p, nullptr);

  const auto loc = cuda::mem_location(p);
  sfc::assert_eq(loc.kind, MemKind::RAM);
  cuda::mem_deallocate(p, loc);
}

SFC_TEST(device) {
  const auto n = 16U;

  auto p = cuda::mem_allocate(n * sizeof(u32), MemKind::GPU);
  sfc::assert_ne(p, nullptr);

  const auto loc = cuda::mem_location(p);
  sfc::assert_eq(loc.kind, MemKind::GPU);

  cuda::mem_deallocate(p, loc);
}

SFC_TEST(managed) {
  const auto n = 16U;

  auto p = cuda::mem_allocate(n * sizeof(u32), MemKind::UVA);
  sfc::assert_ne(p, nullptr);

  const auto loc = cuda::mem_location(p);
  sfc::assert_eq(loc.kind, MemKind::UVA);
  cuda::mem_deallocate(p, loc);
}

}  // namespace sfc::cuda::memory::test
