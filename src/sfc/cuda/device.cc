#include <cuda.h>

#include "sfc/cuda/device.h"
#include "sfc/cuda/stream.h"
#include "sfc/cuda/error.h"

namespace sfc::cuda {

#define CU_TRY(expr)       \
  if (auto err = (expr)) { \
    throw Error{err};      \
  }

void init() {
  static auto err = ::cuInit(0);
  if (err != CUDA_SUCCESS) {
    throw Error{err};
  }
}

auto dev_count() -> int {
  cuda::init();

  auto cnt = 0;
  CU_TRY(::cuDeviceGetCount(&cnt));
  return cnt;
}

auto device_get() -> int {
  cuda::init();

  auto dev = 0;
  CU_TRY(::cuCtxGetDevice(&dev));
  return dev;
}

void device_set(int dev) {
  static auto _tls_dev = -1;
  if (_tls_dev == dev) {
    return;
  }

  cuda::init();

  // get primary context
  auto context = CUcontext{nullptr};
  CU_TRY(::cuDevicePrimaryCtxRetain(&context, dev));

  // set context current
  CU_TRY(::cuCtxSetCurrent(context));
  _tls_dev = dev;
}

void device_sync() {
  CU_TRY(::cuCtxSynchronize());
}

auto device_attribute(int dev, CUdevice_attribute attr_id) -> int {
  auto attr_val = int{0};
  CU_TRY(::cuDeviceGetAttribute(&attr_val, attr_id, dev));
  return attr_val;
}

auto Device::current() -> Device {
  const auto dev = cuda::device_get();
  return Device{dev};
}

auto Device::name() const -> const char* {
  static thread_local char buf[64] = {};
  CU_TRY(::cuDeviceGetName(buf, sizeof(buf), id));
  return buf;
}

auto Device::total_memory() const -> usize {
  auto bytes = usize{0};
  CU_TRY(::cuDeviceTotalMem_v2(&bytes, id));
  return bytes;
}
}  // namespace sfc::cuda
