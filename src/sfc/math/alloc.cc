#include "sfc/alloc.h"
#include "sfc/sync.h"
#include "sfc/math/alloc.h"
#include "sfc/cuda/memory.h"
#include "sfc/cuda/device.h"

namespace sfc::math {

class MemBucket {
  struct Info {
    void* ptr;
    usize seq;
  };

  const MemLocation _location;
  const usize _blk_size;
  usize _blk_cnt{0};
  List<Info> _free_list{};

 public:
  explicit MemBucket(MemLocation location, usize size) : _location{location}, _blk_size{size} {}

  ~MemBucket() {
    this->clear();
  }

  MemBucket(MemBucket&&) noexcept = default;

  auto total_count() const -> usize {
    return _blk_cnt;
  }

  auto used_count() const -> usize {
    return _blk_cnt - _free_list.len();
  }

  auto free_count() const -> usize {
    return _free_list.len();
  }

  auto block_size() const -> usize {
    return _blk_size;
  }

 public:
  void* fast_allocate() {
    const auto ptr = _free_list.pop().unwrap_or({nullptr, 0}).ptr;
    return ptr;
  }

  void* slow_allocate() {
    const auto ptr = SysAllocator::allocate(_blk_size, _location);
    if (ptr != nullptr) {
      _blk_cnt += 1;
    }
    return ptr;
  }

  void prepare_slow_alloc(usize seq) {
    if (this->free_count() == 0) {
      return;
    }

    if (this->try_dealloc_all(seq)) {
      return;
    }

    this->try_dealloc_some();
  }

  void deallocate(void* ptr, usize seq) {
    if (ptr == nullptr) {
      return;
    }

    this->fast_deallocate(ptr, seq);
  }

  void clear() {
    for (auto& info : _free_list.as_mut_slice()) {
      this->slow_deallocate(info.ptr);
    }
    _free_list.clear();
  }

 private:
  auto max_unused_age() const -> usize {
    struct Conf {
      usize size;
      usize age;
    };
    static const Conf cfgs[] = {
        {1U << 20, 64},   // 1MB
        {8U << 20, 32},   // 8MB
        {64U << 20, 8},   // 64MB
        {256U << 20, 4},  // 256MB
        {512U << 20, 2},  // 512MB
    };
    for (auto& cfg : cfgs) {
      if (_blk_size <= cfg.size) {
        return cfg.age;
      }
    }
    return 1;
  }

  auto max_hold_seq_range() const -> usize {
    if (_free_list.is_empty()) {
      return 0;
    }

    return _free_list.len() + this->max_unused_age();
  }

  void fast_deallocate(void* ptr, usize seq) {
    _free_list.push({ptr, seq});
  }

  void slow_deallocate(void* ptr) {
    SysAllocator::deallocate(ptr, _blk_size, _location);
    _blk_cnt -= 1;
  }

  auto try_dealloc_all(usize seq) -> bool {
    if (this->used_count() != 0) {
      return false;
    }

    const auto last = _free_list.last().unwrap();
    const auto max_age = this->max_unused_age();
    if (seq - last.seq < max_age) {
      return false;
    }
    this->clear();
    return true;
  }

  auto try_dealloc_some() -> usize {
    if (_free_list.is_empty()) {
      return 0;
    }

    const auto max_cnt = 1 + _free_list.len() / 4;
    for (auto idx = 0U; idx < max_cnt; ++idx) {
      if (!this->try_dealloc_one()) {
        return idx;
      }
    }
    return max_cnt;
  }

  auto try_dealloc_one() -> bool {
    if (this->free_count() == 0) {
      return false;
    }

    const auto max_range = this->max_hold_seq_range();
    const auto first = _free_list.first().unwrap();
    const auto last = _free_list.last().unwrap();
    if (first.seq + max_range >= last.seq) {
      return false;
    }
    this->slow_deallocate(first.ptr);
    _free_list.drain({0, 1});
    return true;
  }
};

class MemPool {
  sync::Mutex _mutex{};
  MemLocation _location;
  List<MemBucket> _free_list;
  usize _seq{0};

 public:
  MemPool(MemLocation location) : _location{location} {}

  ~MemPool() {}

 public:
  auto allocate(usize size) -> void* {
    if (size == 0) {
      return nullptr;
    }

    auto lock = _mutex.lock();

    auto& bucket = this->bucket(size);
    if (auto ptr = bucket.fast_allocate()) {
      return ptr;
    }

    this->prepare_slow_alloc();
    const auto ptr = bucket.slow_allocate();
    return ptr;
  }

  void deallocate(void* ptr, usize size) {
    if (ptr == nullptr) {
      return;
    }

    auto lock = _mutex.lock();
    auto& bucket = this->bucket(size);

    _seq += 1;
    bucket.deallocate(ptr, _seq);
  }

  void shutdown() {
    auto lock = _mutex.lock();
    _free_list.clear();
    _seq = 0;
  }

 private:
  auto bucket(usize size) -> MemBucket& {
    for (auto& b : _free_list.as_mut_slice()) {
      if (b.block_size() == size) {
        return b;
      }
    }
    auto& bucket = _free_list.push(MemBucket{_location, size});
    return bucket;
  }

  void prepare_slow_alloc() {
    for (auto& bucket : _free_list.as_mut_slice()) {
      bucket.prepare_slow_alloc(_seq);
    }
  }
};

auto SysAllocator::allocate(usize size, MemLocation location) -> void* {
  if (size == 0) {
    return nullptr;
  }

  switch (location.kind) {
    case MemKind::CPU: return cuda::HeapAllocator::allocate(size);
    case MemKind::GPU: return cuda::DeviceAllocator::allocate(size);
    case MemKind::UVA: return cuda::ManagedAllocator::allocate(size);
  }
  return nullptr;
}

void SysAllocator::deallocate(void* ptr, [[maybe_unused]] usize size, MemLocation location) {
  if (ptr == nullptr) {
    return;
  }

  switch (location.kind) {
    case MemKind::CPU: return cuda::HeapAllocator::deallocate(ptr);
    case MemKind::GPU: return cuda::DeviceAllocator::deallocate(ptr);
    case MemKind::UVA: return cuda::ManagedAllocator::deallocate(ptr);
  }
}

auto PoolAllocator::pool(MemLocation location) -> MemPool& {
  static MemPool cpu_pool{{MemKind::CPU, 0}};
  static MemPool uva_pool{{MemKind::UVA, 0}};

  static MemPool gpu_pools[] = {
      {{MemKind::GPU, 0}},
      {{MemKind::GPU, 1}},
      {{MemKind::GPU, 2}},
      {{MemKind::GPU, 3}},
      {{MemKind::GPU, 4}},
      {{MemKind::GPU, 5}},
      {{MemKind::GPU, 6}},
      {{MemKind::GPU, 7}},
  };

  switch (location.kind) {
    default:           return cpu_pool;
    case MemKind::CPU: return cpu_pool;
    case MemKind::GPU: return gpu_pools[location.device];
    case MemKind::UVA: return uva_pool;
  }
}

auto PoolAllocator::allocate(usize size, MemLocation location) -> void* {
  auto& pool = PoolAllocator::pool(location);
  return pool.allocate(size);
}

void PoolAllocator::deallocate(void* ptr, usize size, MemLocation location) {
  auto& pool = PoolAllocator::pool(location);
  pool.deallocate(ptr, size);
}

}  // namespace sfc::math
