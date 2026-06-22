#include "sfc/math/rawbuf.h"
#include "sfc/alloc.h"
#include "sfc/sync.h"
#include "sfc/time.h"

namespace sfc::math {

struct Bucket {
  MemType _mem_type;
  usize _blk_size;
  List<void*> _free_list;
  time::Instant _update_time;

 public:
  auto total_size() const -> usize {
    return _blk_size * _free_list.len();
  }

  auto alloc() -> void* {
    _update_time = time::Instant::now();
    if (auto ptr = _free_list.pop().unwrap_or(nullptr)) {
      return ptr;
    }

    return Alloc{_mem_type}.alloc(_blk_size);
  }

  void dealloc(void* ptr) {
    if (ptr == nullptr) return;

    _update_time = time::Instant::now();
    _free_list.push(ptr);
  }
};

class MemPool {
  MemType _mtype;
  List<Bucket> _buckets;
  sync::Mutex _mutex;

 public:
  MemPool(MemType mtype) : _mtype{mtype} {}

  static auto get(MemType mtype) -> MemPool& {
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
    auto& bucket = this->get_bucket(size);
    while (true) {
      if (auto ptr = bucket.alloc()) {
        return ptr;
      }
      auto& bucket = this->get_oldest_bucket();
      if (bucket.total_size() == 0) {
        break;
      }
    }
    return nullptr;
  }

  void dealloc(void* ptr, usize size) {
    if (ptr == nullptr) return;

    auto guard = _mutex.lock();
    auto& bucket = this->get_bucket(size);
    bucket.dealloc(ptr);
  }

 private:
  auto get_oldest_bucket() -> Bucket& {
    return _buckets.as_mut_slice().iter_mut().max_by_key([](auto& bucket) { return bucket._update_time; }).unwrap();
  }

  auto get_bucket(usize size) -> Bucket& {
    for (auto& bucket : _buckets.as_mut_slice()) {
      if (bucket._blk_size >= size) {
        return bucket;
      }
    }

    return _buckets.push(Bucket{_mtype, size, {}});
  }
};

auto Alloc::alloc(usize size) -> void* {
  switch (mtype) {
    default:
    case MemType::CPU: return cuda::host_alloc(size);
    case MemType::GPU: return cuda::device_alloc(size);
    case MemType::UVA: return cuda::managed_alloc(size);
  }
}

void Alloc::dealloc(void* ptr, usize size) {
  switch (mtype) {
    default:
    case MemType::CPU: return cuda::host_free(ptr);
    case MemType::GPU: return cuda::device_free(ptr);
    case MemType::UVA: return cuda::managed_free(ptr);
  }
}

auto PoolAlloc::alloc(usize size) -> void* {
  auto& pool = MemPool::get(mtype);
  return pool.alloc(size);
}

void PoolAlloc::dealloc(void* ptr, usize size) {
  auto& pool = MemPool::get(mtype);
  pool.dealloc(ptr, size);
}

}  // namespace sfc::math
