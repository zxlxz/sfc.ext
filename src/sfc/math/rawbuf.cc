#include "sfc/math/rawbuf.h"
#include "sfc/alloc.h"
#include "sfc/sync.h"
#include "sfc/time.h"

namespace sfc::math {

class Bucket {
  MemType _mtype;
  usize _block_size;
  List<void*> _free_list;

 public:
  Bucket(MemType mtype, usize blk_size) : _mtype{mtype}, _block_size{blk_size} {}

  ~Bucket() {
    this->truncate(0);
  }

  Bucket(Bucket&&) noexcept = default;
  Bucket& operator=(Bucket&&) noexcept = default;

 public:
  auto len() const -> usize {
    return _free_list.len();
  }

  auto block_size() const -> usize {
    return _block_size;
  }
  void truncate(usize len) {
    auto& a = DefaultAlloc::instance();
    while (_free_list.len() > len) {
      const auto p = _free_list.pop().unwrap();
      a.deallocate(p, _block_size, _mtype);
    }
  }

  auto alloc() -> void* {
    auto& a = DefaultAlloc::instance();
    const auto ptr = _free_list.pop().unwrap_or_else([&] { return a.allocate(_block_size, _mtype); });
    return ptr;
  }

  void dealloc(void* ptr) {
    if (ptr == nullptr) {
      return;
    }
    _free_list.push(ptr);
  }
};

class MemPool {
  MemType _mtype;
  List<Bucket> _buckets;
  sync::Mutex _mutex;

 public:
  explicit MemPool(MemType mtype) : _mtype{mtype} {}

  ~MemPool() {}

 public:
  static auto instance(MemType mtype) -> MemPool& {
    static auto cpu_pool = MemPool{MemType::CPU};
    static auto gpu_pool = MemPool{MemType::GPU};
    static auto uva_pool = MemPool{MemType::UVA};

    switch (mtype) {
      default:
      case MemType::CPU: return cpu_pool;
      case MemType::GPU: return gpu_pool;
      case MemType::UVA: return uva_pool;
    }
  }

  auto alloc(usize size) -> void* {
    auto guard = _mutex.lock();

    auto& bucket = this->bucket(size);
    while (true) {
      if (auto ptr = bucket.alloc()) {
        return ptr;
      }
      this->truncate_free_list();
    }
    return nullptr;
  }

  void dealloc(void* ptr, usize size) {
    if (ptr == nullptr) {
      return;
    }

    auto guard = _mutex.lock();
    auto& bucket = this->bucket(size);
    bucket.dealloc(ptr);
  }

 private:
  auto bucket(usize size) -> Bucket& {
    auto match = [&](auto& bucket) { return bucket.block_size() == size; };
    auto create_new = [&] -> Bucket& { return _buckets.push(Bucket{_mtype, size}); };
    return _buckets.iter_mut().find(match).unwrap_or_else(create_new);
  }

  void truncate_free_list() {
    for (auto& bucket : _buckets.as_mut_slice()) {
      const auto old_len = bucket.len();
      const auto new_len = old_len / 2;
      bucket.truncate(new_len);
    }
  }
};

auto DefaultAlloc::allocate(usize size, MemType mtype) -> void* {
  switch (mtype) {
    default:
    case MemType::CPU: return cuda::host_alloc(size);
    case MemType::GPU: return cuda::device_alloc(size);
    case MemType::UVA: return cuda::managed_alloc(size);
  }
}

void DefaultAlloc::deallocate(void* ptr, usize size, MemType mtype) {
  switch (mtype) {
    default:
    case MemType::CPU: return cuda::host_free(ptr);
    case MemType::GPU: return cuda::device_free(ptr);
    case MemType::UVA: return cuda::managed_free(ptr);
  }
}

auto PoolAlloc::allocate(usize size, MemType mtype) -> void* {
  auto& pool = MemPool::instance(mtype);
  return pool.alloc(size);
}

void PoolAlloc::deallocate(void* ptr, usize size, MemType mtype) {
  auto& pool = MemPool::instance(mtype);
  pool.dealloc(ptr, size);
}

}  // namespace sfc::math
