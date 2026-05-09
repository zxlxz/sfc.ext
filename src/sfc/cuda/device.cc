#include <cuda.h>

#include "sfc/core.h"
#include "sfc/cuda/device.h"
#include "sfc/cuda/stream.h"
#include "sfc/cuda/error.h"

namespace sfc::cuda {

void init() {
  static auto err = ::cuInit(0);
  if (err != CUDA_SUCCESS) {
    panic::panic_fmt("cuInit failed, err={}", Error{err});
  }

}

auto dev_count() -> int {
  cuda::init();

  auto cnt = 0;
  if (auto e = ::cuDeviceGetCount(&cnt)) {
    panic::panic_fmt("cuDeviceGetCount failed, err={}", Error{e});
  }
  return cnt;
}

auto device_get() -> int {
  cuda::init();

  auto dev = 0;
  if (auto e = ::cuCtxGetDevice(&dev)) {
    panic::panic_fmt("cuCtxGetDevice failed, err={}", Error{e});
  }
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
  if (auto e = ::cuDevicePrimaryCtxRetain(&context, dev)) {
    panic::panic_fmt("cuDevicePrimaryCtxRetain failed, err={}", Error{e});
  }

  // set context current
  if (auto e = ::cuCtxSetCurrent(context)) {
    panic::panic_fmt("cuCtxSetCurrent failed, err={}", Error{e});
  }
  _tls_dev = dev;
}

void device_sync() {
  if (auto e = ::cuCtxSynchronize()) {
    panic::panic_fmt("cuCtxSynchronize failed, err={}", Error{e});
  }
}

auto device_attribute(int dev, CUdevice_attribute attr_id) -> int {
  auto attr_val = int{0};
  if (auto e = ::cuDeviceGetAttribute(&attr_val, attr_id, dev)) {
    panic::panic_fmt("cuDeviceGetAttribute failed, err={}", Error{e});
  }
  return attr_val;
}

auto Device::current() -> Device {
  const auto dev = cuda::device_get();
  return Device{dev};
}

auto Device::name() const -> const char* {
  static thread_local char buf[64] = {};
  if (auto e = ::cuDeviceGetName(buf, sizeof(buf), id)) {
    panic::panic_fmt("cuDeviceGetName failed, err={}", Error{e});
  }
  return buf;
}

auto Device::total_memory() const -> usize {
  auto bytes = usize{0};
  if (auto e = ::cuDeviceTotalMem_v2(&bytes, id)) {
    panic::panic_fmt("cuDeviceTotalMem_v2 failed, err={}", Error{e});
  }
  return bytes;
}
}  // namespace sfc::cuda
