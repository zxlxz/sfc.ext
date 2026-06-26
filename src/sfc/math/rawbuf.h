#pragma once

#include "sfc/cuda/memory.h"

namespace sfc::math {

enum class MemType {
  CPU,
  GPU,
  UVA,
};

class RawBuf {
  void* _ptr{nullptr};
  usize _size{0};
  MemType _mtype{MemType::CPU};

 public:
  explicit RawBuf() noexcept;
  ~RawBuf();

  RawBuf(RawBuf&& other) noexcept;
  RawBuf& operator=(RawBuf&& other) noexcept;

  static auto with_capacity(usize capacity, MemType mtype);

 public:
  auto ptr() const -> void* {
    return _ptr;
  }

  auto size() const -> usize {
    return _size;
  }

  auto mtype() const -> MemType {
    return _mtype;
  }
};

}  // namespace sfc::math
