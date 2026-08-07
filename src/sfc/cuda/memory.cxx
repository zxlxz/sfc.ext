#include "sfc/test.h"
#include "sfc/cuda/memory.h"

namespace sfc::cuda::memory::test {

using mem::Layout;

SFC_TEST(heap) {
  auto a = Allocator{MemKind::CPU, 0};

  const auto n = 16U;
  auto p = a.allocate(Layout::array<u32>(n));
  sfc::assert_ne(p, nullptr);

  a.deallocate(p, Layout::array<u32>(n));
}

SFC_TEST(host) {
  auto a = Allocator{MemKind::RAM, 0};

  const auto n = 16U;
  auto p = a.allocate(Layout::array<u32>(n));
  sfc::assert_ne(p, nullptr);

  a.deallocate(p, Layout::array<u32>(n));
}

SFC_TEST(device) {
  auto a = Allocator{MemKind::GPU, 0};

  const auto n = 16U;
  auto p = a.allocate(Layout::array<u32>(n));
  sfc::assert_ne(p, nullptr);
  a.deallocate(p, Layout::array<u32>(n));
}

}  // namespace sfc::cuda::memory::test
