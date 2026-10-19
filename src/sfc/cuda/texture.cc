#include <cuda.h>

#include "sfc/cuda/mod.h"
#include "sfc/cuda/stream.h"
#include "sfc/cuda/texture.h"

namespace sfc::cuda {

namespace detail {

using arr_t = CUarray;

auto texture_new(arr_t arr, TexFilt tex_filt, TexAddr tex_addr) -> Result<u64> {
  auto res_desc = CUDA_RESOURCE_DESC{};
  res_desc.resType = CU_RESOURCE_TYPE_ARRAY;
  res_desc.res.array.hArray = arr;
  auto tex_desc = CUDA_TEXTURE_DESC{};
  tex_desc.addressMode[0] = CUaddress_mode(tex_addr);
  tex_desc.addressMode[1] = CUaddress_mode(tex_addr);
  tex_desc.addressMode[2] = CUaddress_mode(tex_addr);
  tex_desc.filterMode = CUfilter_mode(tex_filt);
  auto tex = CUtexObject{};
  if (auto err = cuTexObjectCreate(&tex, &res_desc, &tex_desc, nullptr)) {
    return Error(err);
  }

  return u64{tex};
}

auto texture_del(u64 tex) -> Result<> {
  if (auto err = cuTexObjectDestroy(CUtexObject(tex))) {
    return Error(err);
  }
  return Ok{};
}

}  // namespace detail

TexObject::TexObject() noexcept : _tex{0} {}

TexObject::~TexObject() noexcept {
  if (_tex == 0) {
    return;
  }

  (void)detail::texture_del(_tex);
}

TexObject::TexObject(TexObject&& other) noexcept : _tex{mem::take(other._tex)} {}

TexObject& TexObject::operator=(TexObject&& other) noexcept {
  if (this != &other) {
    mem::swap(_tex, other._tex);
  }
  return *this;
}

auto TexObject::new_(CUarray_st* arr, TexFilt filt, TexAddr addr) -> TexObject {
  auto tex = detail::texture_new(arr, filt, addr).unwrap();
  auto res = TexObject{};
  res._tex = tex;
  return res;
}

}  // namespace sfc::cuda
