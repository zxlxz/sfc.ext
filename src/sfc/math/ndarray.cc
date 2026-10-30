#include "sfc/math/ndarray.h"

namespace sfc::math {

class DefaultAlloc : public IAlloc {
  alloc::Global _inn;

 public:
  static auto instance() -> DefaultAlloc& {
    static DefaultAlloc res{};
    return res;
  }

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

auto Allocator::allocate(Layout layout) -> void* {
  if (!_alloc) return nullptr;
  return _alloc->allocate(layout);
}

auto Allocator::allocate_zeroed(Layout layout) -> void* {
  if (!_alloc) return nullptr;
  return _alloc->allocate_zeroed(layout);
}

void Allocator::deallocate(void* ptr, Layout layout) {
  if (!_alloc) return;
  _alloc->deallocate(ptr, layout);
}

}  // namespace sfc::math
