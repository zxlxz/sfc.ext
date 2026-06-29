#include "sfc/sync.h"
#include "sfc/math/alloc.h"
#include "sfc/cuda/memory.h"

namespace sfc::math {

class MemBucket {
  struct Info {
    void* ptr;
    usize seq;
  };

  const MemType _mem_type;
  const usize _blk_size;
  usize _blk_cnt{0};
  List<Info> _free_list{};

 public:
  explicit MemBucket(MemType type, usize size) : _mem_type{type}, _blk_size{size} {}

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
    const auto ptr = SysAllocator::allocate(_blk_size, _mem_type);
    if (ptr != nullptr) {
      _blk_cnt += 1;
    }
    return ptr;
  }

  void prepare_slow_alloc(usize seq) {
    if (this->free_count() == 0) {
      return;
    }

    this->try_dealloc_all(seq);
    this->try_dealloc_one();
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
    if (_blk_size <= 1U << 20) return 64;   // 1MB
    if (_blk_size <= 4U << 20) return 32;   // 4MB
    if (_blk_size <= 32U << 20) return 8;   // 32M
    if (_blk_size <= 128U << 20) return 4;  // 128MB
    if (_blk_size <= 512U << 20) return 2;  // 512MB
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
    SysAllocator::deallocate(ptr, _blk_size, _mem_type);
    _blk_cnt -= 1;
  }

  void try_dealloc_all(usize seq) {
    if (this->used_count() != 0) {
      return;
    }

    const auto last = _free_list.last().unwrap();
    const auto max_age = this->max_unused_age();
    if (seq - last.seq < max_age) {
      return;
    }
    this->clear();
  }

  void try_dealloc_one() {
    if (this->free_count() == 0) {
      return;
    }

    const auto max_range = this->max_hold_seq_range();
    const auto first = _free_list.first().unwrap();
    const auto last = _free_list.last().unwrap();
    if (first.seq + max_range >= last.seq) {
      return;
    }
    this->slow_deallocate(first.ptr);
    _free_list.drain({0, 1});
  }
};

class MemPool {
  MemType _mem_type;
  usize _dealloc_seq{0};
  List<MemBucket> _free_list;
  sync::Mutex _mutex;

 public:
  MemPool(MemType mtype) : _mem_type{mtype} {}

  ~MemPool() {}

  static auto instance(MemType mtype) -> MemPool& {
    static MemPool cpu_pool{MemType::CPU};
    static MemPool gpu_pool{MemType::GPU};
    static MemPool uva_pool{MemType::UVA};

    switch (mtype) {
      default:           return cpu_pool;
      case MemType::CPU: return cpu_pool;
      case MemType::GPU: return gpu_pool;
      case MemType::UVA: return uva_pool;
    }
  }

 public:
  auto allocate(usize size) -> void* {
    auto lock = _mutex.lock();

    auto& bucket = this->bucket(size);
    if (auto ptr = bucket.fast_allocate()) {
      return ptr;
    }

    this->prepare_slow_alloc();
    const auto ptr = bucket.slow_allocate();
    return ptr;
  }

  auto deallocate(void* ptr, usize size) {
    if (ptr == nullptr) {
      return;
    }

    auto lock = _mutex.lock();
    auto& bucket = this->bucket(size);

    _dealloc_seq += 1;
    bucket.deallocate(ptr, _dealloc_seq);
  }

  void shutdown() {
    auto lock = _mutex.lock();
    _free_list.clear();
    _dealloc_seq = 0;
  }

 private:
  auto bucket(usize size) -> MemBucket& {
    for (auto& b : _free_list.as_mut_slice()) {
      if (b.block_size() == size) {
        return b;
      }
    }
    auto& bucket = _free_list.push(MemBucket{_mem_type, size});
    return bucket;
  }

  void prepare_slow_alloc() {
    for (auto& bucket : _free_list.as_mut_slice()) {
      bucket.prepare_slow_alloc(_dealloc_seq);
    }
  }
};

auto SysAllocator::allocate(usize size, MemType mtype) -> void* {
  if (size == 0) return nullptr;

  switch (mtype) {
    default:           return nullptr;
    case MemType::CPU: return cuda::heap_alloc(size);
    case MemType::GPU: return cuda::device_alloc(size);
    case MemType::UVA: return cuda::managed_alloc(size);
  }
}

void SysAllocator::deallocate(void* ptr, usize size, MemType mtype) {
  if (ptr == nullptr) return;

  switch (mtype) {
    default:           return;
    case MemType::CPU: return cuda::heap_free(ptr);
    case MemType::GPU: return cuda::device_free(ptr);
    case MemType::UVA: return cuda::managed_free(ptr);
  }
}

auto PoolAllocator::allocate(usize size, MemType mtype) -> void* {
  auto& pool = MemPool::instance(mtype);
  return pool.allocate(size);
}

void PoolAllocator::deallocate(void* ptr, usize size, MemType mtype) {
  auto& pool = MemPool::instance(mtype);
  pool.deallocate(ptr, size);
}

}  // namespace sfc::math
