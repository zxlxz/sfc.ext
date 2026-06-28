#include "sfc/sync.h"
#include "sfc/math/rawbuf.h"
#include "sfc/cuda/memory.h"

namespace sfc::math {

class Bucket {
  struct Info {
    void* _ptr;
    usize _seq;
  };

  usize _size;
  List<Info> _free_list;

 public:
  explicit Bucket(usize size) : _size{size} {}

  auto size() const -> usize {
    return _size;
  }

  void push(void* ptr, usize seq) {
    _free_list.push({ptr, seq});
  }

  auto pop() -> void* {
    const auto info = _free_list.pop().unwrap_or({nullptr, 0});
    return info._ptr;
  }

  void remove_with(usize seq, auto&& dealloc) {
    const auto f = [&](const auto& info) { return info._seq > seq; };
    const auto cnt = _free_list.iter().position(f).unwrap_or(_free_list.len());
    _free_list.drain({0, cnt}).for_each([&](const auto& info) { dealloc(info._ptr, _size); });
  }
};

class MemPool {
  MemType _mem_type;
  usize _oldest_seq{0};
  usize _current_seq{0};
  List<Bucket> _free_list;
  sync::Mutex _mutex;

 public:
  explicit MemPool(MemType mtype) : _mem_type{mtype} {}

  ~MemPool() {
    this->free_some_memory(_current_seq);
  }

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
    if (auto ptr = this->fast_allocate(size)) {
      return ptr;
    }

    const auto ptr = this->slow_allocate(size);
    return ptr;
  }

  auto deallocate(void* ptr, usize size) {
    if (ptr == nullptr) return;

    auto lock = _mutex.lock();
    auto& bucket = this->bucket(size);
    bucket.push(ptr, _current_seq++);
  }

 private:
  auto bucket(usize size) -> Bucket& {
    for (auto& b : _free_list.as_mut_slice()) {
      if (b.size() >= size) {
        return b;
      }
    }
    auto& bucket = _free_list.push(Bucket{size});
    return bucket;
  }

  auto fast_allocate(usize size) -> void* {
    auto lock = _mutex.lock();
    auto& bucket = this->bucket(size);
    return bucket.pop();
  }

  auto slow_allocate(usize size) -> void* {
    const auto kMaxTryCnt = 3U;
    auto lock = _mutex.lock();
    for (auto try_cnt = 0U; try_cnt < kMaxTryCnt; ++try_cnt) {
      auto ptr = Allocator::allocate(size, _mem_type);
      if (ptr != nullptr) {
        return ptr;
      }

      const auto mid_seq = (_oldest_seq + _current_seq) / 2;
      this->free_some_memory(mid_seq);
    }
    return nullptr;
  }

  void free_some_memory(usize seq) {
    auto dealloc_imp = [&](void* ptr, usize size) {
      if (ptr == 0) return;
      return Allocator::deallocate(ptr, size, _mem_type);
    };

    for (auto& bucket : _free_list.as_mut_slice()) {
      bucket.remove_with(seq, dealloc_imp);
    }
    _oldest_seq = seq;
  }
};

auto Allocator::allocate(usize size, MemType mtype) -> void* {
  if (size == 0) return nullptr;

  switch (mtype) {
    default:           return nullptr;
    case MemType::CPU: return cuda::heap_alloc(size);
    case MemType::GPU: return cuda::device_alloc(size);
    case MemType::UVA: return cuda::managed_alloc(size);
  }
}

void Allocator::deallocate(void* ptr, usize size, MemType mtype) {
  if (ptr == nullptr) return;
  switch (mtype) {
    default:           return;
    case MemType::CPU: return cuda::heap_free(ptr);
    case MemType::GPU: return cuda::device_free(ptr);
    case MemType::UVA: return cuda::managed_free(ptr);
  }
}

void Allocator::bzero(void* ptr, usize size, MemType mtype) {
  if (ptr == nullptr || size == 0) return;

  switch (mtype) {
    default:           break;
    case MemType::CPU: __builtin_memset(ptr, 0, size); break;
    case MemType::GPU: cuda::fill_bytes(ptr, 0, size); break;
    case MemType::UVA: cuda::fill_bytes(ptr, 0, size); break;
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
