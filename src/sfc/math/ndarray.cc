#include "sfc/math/ndarray.h"
#include "sfc/cuda/memory.h"

namespace sfc::math {

RawBuf::RawBuf() noexcept {}

RawBuf::~RawBuf() {
  if (_ptr == nullptr) return;
  _alloc.deallocate(_ptr, _size, _location);
}

RawBuf::RawBuf(RawBuf&& other) noexcept
    : _ptr{other._ptr}, _size{other._size}, _location{other._location}, _alloc{other._alloc} {
  other._ptr = nullptr;
  other._size = 0;
  other._location = MemLocation{MemKind::CPU, 0};
}

RawBuf& RawBuf::operator=(RawBuf&& other) noexcept {
  if (this != &other) {
    mem::swap(_ptr, other._ptr);
    mem::swap(_size, other._size);
    mem::swap(_location, other._location);
    mem::swap(_alloc, other._alloc);
  }

  return *this;
}

auto RawBuf::xnew(usize size, MemLocation location) -> RawBuf {
  auto alloc = A{};

  auto res = RawBuf{};
  res._ptr = ptr::cast<u8>(alloc.allocate(size, location));
  res._size = size;
  res._location = location;
  res._alloc = alloc;
  return res;
}

void RawBuf::bzero() {
  if (_ptr == nullptr) {
    return;
  }

  auto p = ptr::cast<u8>(_ptr);
  switch (_location.kind) {
    default:           break;
    case MemKind::CPU: ptr::write_bytes(p, 0, _size); break;
    case MemKind::GPU: cuda::fill_bytes(p, 0, _size); break;
    case MemKind::UVA: cuda::fill_bytes(p, 0, _size); break;
  }
}

void RawBuf::copy_from(const RawBuf& src) {
  sfc::assert_(_size == src._size, "RawBuf::copy_from: size mismatch");

  const auto psrc = ptr::cast<const u8>(src._ptr);
  const auto pdst = ptr::cast<u8>(_ptr);
  const auto size = src._size;

  switch (_location.kind) {
    default:           break;
    case MemKind::CPU: ptr::copy_nonoverlapping(psrc, pdst, size); break;
    case MemKind::GPU: cuda::copy_bytes(psrc, pdst, size); break;
    case MemKind::UVA: cuda::copy_bytes(psrc, pdst, size); break;
  }
}

}  // namespace sfc::math
