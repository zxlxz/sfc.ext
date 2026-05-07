#include <cuda.h>

#include "sfc/cuda/texture.h"
#include "sfc/cuda/error.h"

#define CU_TRY(expr)       \
  if (auto err = (expr)) { \
    throw Error{err};      \
  }

namespace sfc::cuda {

auto texture_new(buf_t arr, TexFilt filt_mode, TexAddr addr_mode) -> tex_t {
  if (!arr) {
    throw Error{CUDA_ERROR_INVALID_VALUE};
  }

  const auto cu_filt = static_cast<CUfilter_mode>(filt_mode);
  const auto cu_addr = static_cast<CUaddress_mode>(addr_mode);

  const auto res_desc = CUDA_RESOURCE_DESC_st{
      .resType = CU_RESOURCE_TYPE_ARRAY,
      .res = {.array = {.hArray = arr}},
  };

  const auto tex_desc = CUDA_TEXTURE_DESC_st{
      .addressMode = {cu_addr, cu_addr, cu_addr},
      .filterMode = cu_filt,
      .flags = 0,
  };

  const auto view_desc = nullptr;

  auto tex_obj = tex_t{0};
  CU_TRY(::cuTexObjectCreate(&tex_obj, &res_desc, &tex_desc, view_desc));

  return tex_obj;
}

void texture_del(tex_t obj) {
  if (!obj) {
    return;
  }
  CU_TRY(::cuTexObjectDestroy(obj));
}

}  // namespace sfc::cuda
