#include "sfc/test.h"
#include "sfc/cuda/memory.h"

namespace sfc::cuda::memory::test {

SFC_TEST(heap) {
  const auto n = 16U;
  auto p = cuda::allocate(n * sizeof(u32), {MemKind::CPU});
  sfc::assert_ne(p, nullptr);

  const auto loc = cuda::location(p);
  sfc::assert_eq(loc.kind, MemKind::CPU);
  cuda::deallocate(p, n * sizeof(u32), loc);
}

SFC_TEST(host) {
  const auto n = 16U;

  auto p = cuda::allocate(n * sizeof(u32), {MemKind::RAM});
  sfc::assert_ne(p, nullptr);

  const auto loc = cuda::location(p);
  sfc::assert_eq(loc.kind, MemKind::RAM);
  cuda::deallocate(p, n * sizeof(u32), loc);
}

SFC_TEST(device) {
  const auto n = 16U;

  auto p = cuda::allocate(n * sizeof(u32), {MemKind::GPU});
  sfc::assert_ne(p, nullptr);

  const auto loc = cuda::location(p);
  sfc::assert_eq(loc.kind, MemKind::GPU);

  cuda::deallocate(p, n * sizeof(u32), loc);
}

SFC_TEST(managed) {
  const auto n = 16U;

  auto p = cuda::allocate(n * sizeof(u32), {MemKind::UVA});
  sfc::assert_ne(p, nullptr);

  const auto loc = cuda::location(p);
  sfc::assert_eq(loc.kind, MemKind::UVA);
  cuda::deallocate(p, n * sizeof(u32), loc);
}

}  // namespace sfc::cuda::memory::test
