#include "sfc/math/ndarray.h"

namespace sfc::math {

class GlobalAllocator : public Allocator {
  alloc::Global _inn;

 public:
 public:
  void* allocate(Layout layout) override {
    return _inn.allocate(layout);
  }

  void* allocate_zeroed(Layout layout) override {
    return _inn.allocate_zeroed(layout);
  }

  void deallocate(void* ptr, Layout layout) override {
    _inn.deallocate(ptr, layout);
  }
};

auto Allocator::global() -> Allocator& {
  static auto alloc = GlobalAllocator{};
  return alloc;
}

}  // namespace sfc::math
