#include "sfc/math/ndarray.h"
#include "sfc/cuda/memory.h"

namespace sfc::math {

RawBuf::RawBuf() noexcept {}

RawBuf::~RawBuf() {
  if (_ptr == nullptr) return;
  _alloc.deallocate(_ptr, _size, _mtype);
}

RawBuf::RawBuf(RawBuf&& other) noexcept
    : _ptr{other._ptr}, _size{other._size}, _mtype{other._mtype}, _alloc{other._alloc} {
  other._ptr = nullptr;
  other._size = 0;
  other._mtype = MemType::CPU;
}

RawBuf& RawBuf::operator=(RawBuf&& other) noexcept {
  if (this != &other) {
    mem::swap(_ptr, other._ptr);
    mem::swap(_size, other._size);
    mem::swap(_mtype, other._mtype);
    mem::swap(_alloc, other._alloc);
  }

  return *this;
}

auto RawBuf::xnew(usize size, MemType mtype) -> RawBuf {
  auto alloc = A{};

  auto res = RawBuf{};
  res._ptr = ptr::cast<u8>(alloc.allocate(size, mtype));
  res._size = size;
  res._mtype = mtype;
  res._alloc = alloc;
  return res;
}

void RawBuf::bzero() {
  if (_ptr == nullptr) {
    return;
  }

  auto p = ptr::cast<u8>(_ptr);
  switch (_mtype) {
    default:           break;
    case MemType::CPU: ptr::write_bytes(p, 0, _size); break;
    case MemType::GPU: cuda::fill_bytes(p, 0, _size); break;
    case MemType::UVA: cuda::fill_bytes(p, 0, _size); break;
  }
}

void RawBuf::copy_from(const RawBuf& src) {
  sfc::assert_(_size == src._size, "RawBuf::copy_from: size mismatch");

  const auto psrc = ptr::cast<const u8>(src._ptr);
  const auto pdst = ptr::cast<u8>(_ptr);
  const auto size = src._size;

  switch (_mtype) {
    default:           break;
    case MemType::CPU: ptr::copy_nonoverlapping(psrc, pdst, size); break;
    case MemType::GPU: cuda::copy_bytes(psrc, pdst, size); break;
    case MemType::UVA: cuda::copy_bytes(psrc, pdst, size); break;
  }
}

}  // namespace sfc::math
